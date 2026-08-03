"""실패 감지기.

RingLogger.to_arrays()의 원본 배열로 판정한다. dump_text()에는 의존하지 않는다.

세 종류를 감지한다 (토크 임계치 하나로는 절반을 놓친다 — 이전 단계에서 확인:
이미 목표에 도달해 정지한 관절이 걸리면 PD 오차가 없어 토크가 안 튄다):

- TORQUE_SPIKE : |tau| > threshold 가 min_duration 이상 지속 (움직이다 걸림)
- STALL        : |q_des - q| 큰데 |qd| ≈ 0 이 min_duration 이상 지속
                 (정지 중 고착을 잡는 유일한 수단)
- OSCILLATION  : 유의미한 진폭의 qd 부호 반전이 고주파로 반복 (게인 과다)

오탐 방지 장치:
- 최소 지속 시간: 판정 조건이 트레일링 윈도우 전체에서 성립해야 발동
- 히스테리시스: 발동 후 조건이 release 수준 밑으로 내려가야 재무장
- refractory: 같은 (타입, 관절)은 최소 간격 안에 재발동 금지
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_core.recovery.events import FailureEvent, FailureType


@dataclass
class DetectorConfig:
    # TORQUE_SPIKE
    torque_threshold: float | np.ndarray = 15.0  # [Nm] 관절별 또는 스칼라
    torque_min_duration_s: float = 0.02
    torque_release_frac: float = 0.7   # 재무장: |tau|가 threshold*frac 밑으로
    # 충격 과도신호는 짧다 (kd 댐핑이 ms 단위로 상쇄). 폴링 주기와 무관하게
    # 잡으려면 "지금 이 순간 초과 중"이 아니라 "최근 lookback 안에 min_duration
    # 이상 지속된 초과 구간이 있었나"를 봐야 한다.
    torque_lookback_s: float = 0.15

    # STALL
    stall_err_threshold: float = 0.1   # [rad] 지령-실제 오차
    stall_qd_eps: float = 0.05         # [rad/s] "안 움직임" 판정
    stall_min_duration_s: float = 0.25
    stall_release_frac: float = 0.5    # 재무장: 오차가 threshold*frac 밑으로

    # OSCILLATION
    osc_qd_amp_eps: float = 0.3        # [rad/s] 이 진폭 넘는 반전만 센다
    osc_min_flips_hz: float = 8.0      # 초당 부호 반전 수
    osc_window_s: float = 0.3
    osc_release_frac: float = 0.5

    # CONTINUOUS_OVERLOAD — 지속 과부하 (열 예산)
    # 순간 상한이 아니라 이동평균 예산이다. 신호는 tau_cmd(지령 토크)를 쓴다:
    # 외란과 맞서는 홀드는 출력측 순토크가 0이라 tau_measured로는 안 보이지만
    # 모터 전류(발열)는 지령 쪽에 흐른다. tau_cmd가 없으면 tau로 폴백.
    cont_budget: float | np.ndarray | None = None   # [Nm] None = 채널 비활성
    cont_window_s: float = 1.0
    cont_release_frac: float = 0.9     # 평균이 budget*frac 밑이면 해제

    # 공통
    refractory_s: float = 1.0          # 같은 (타입,관절) 재발동 최소 간격 (로봇 클록)


@dataclass
class _ChannelState:
    armed: bool = True
    last_fire_t: float = -1e9


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True인 연속 구간을 (start_idx, end_idx) 리스트로 (양끝 포함)."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


class FailureDetector:
    """check(arrays)를 주기적으로 (예: 20Hz) 불러 이벤트 리스트를 받는다."""

    def __init__(self, n_joints: int, config: DetectorConfig | None = None) -> None:
        self.n = int(n_joints)
        self.cfg = config or DetectorConfig()
        thr = np.asarray(self.cfg.torque_threshold, dtype=float)
        self._tau_thr = np.full(self.n, float(thr)) if thr.ndim == 0 else thr.copy()
        if self._tau_thr.shape != (self.n,):
            raise ValueError(f"torque_threshold shape {thr.shape}, expected ({self.n},)")
        self._state: dict[tuple[str, int], _ChannelState] = {}
        if self.cfg.cont_budget is not None:
            cb = np.asarray(self.cfg.cont_budget, dtype=float)
            self._cont_budget = np.full(self.n, float(cb)) if cb.ndim == 0 else cb.copy()
        else:
            self._cont_budget = None
        self._in_overload = np.zeros(self.n, dtype=bool)

    def _chan(self, ftype: FailureType, j: int) -> _ChannelState:
        return self._state.setdefault((ftype.value, j), _ChannelState())

    def reset(self) -> None:
        self._state.clear()

    # ------------------------------------------------------------------ 판정
    def check(self, arrays: dict) -> list[FailureEvent]:
        """RingLogger.to_arrays() 결과를 받아 새로 발동한 이벤트를 돌려준다."""
        t = arrays["t"]
        if len(t) < 3:
            return []
        events: list[FailureEvent] = []
        events += self._check_torque(arrays)
        events += self._check_stall(arrays)
        events += self._check_oscillation(arrays)
        events += self._check_continuous_overload(arrays)
        return events

    def _trailing(self, t: np.ndarray, duration: float) -> np.ndarray:
        """끝에서 duration만큼의 샘플 마스크."""
        return t >= (t[-1] - duration)

    def _check_continuous_overload(self, a: dict) -> list[FailureEvent]:
        """지령 토크 1초 이동평균 > 연속 예산 → CONTINUOUS_OVERLOAD.
        예산 밑으로 회복(release_frac)하면 OVERLOAD_CLEARED를 짝으로 발행.

        신호: tau_cmd (클램프 적용 후 지령 — 로거가 잘라서 기록). 클램프에
        눌린 구간에서 지령 원값을 쓰면 실제 발열을 과대평가해 불필요한
        slow-down이 걸린다. tau_cmd가 없으면(명령 미기록) tau로 폴백.
        """
        if self._cont_budget is None:
            return []
        t = a["t"]
        if t[-1] - t[0] < self.cfg.cont_window_s:   # 창이 덜 찼으면 판정 보류
            return []
        sig = a.get("tau_cmd")
        if sig is None or not np.any(np.isfinite(sig)):
            sig = a["tau"]
        m = self._trailing(t, self.cfg.cont_window_s)
        window = np.abs(sig[m])
        # 명령이 안 찍힌 행(nan)은 평균에서 제외 (all-nan 관절은 판정 보류)
        finite = np.isfinite(window)
        cnt = finite.sum(axis=0)
        avg = np.where(cnt > 0,
                       np.where(finite, window, 0.0).sum(axis=0) / np.maximum(cnt, 1),
                       np.nan)

        out: list[FailureEvent] = []
        now = float(t[-1])
        for j in range(self.n):
            budget = self._cont_budget[j]
            if not np.isfinite(avg[j]):
                continue
            if not self._in_overload[j]:
                ev = self._try_fire(
                    FailureType.CONTINUOUS_OVERLOAD, j, now,
                    condition=bool(avg[j] > budget),
                    release=bool(avg[j] < budget * self.cfg.cont_release_frac),
                    severity=min(1.0, (avg[j] / budget - 1.0) * 2 + 0.3),
                    snapshot={"tau_avg_1s": float(avg[j]), "budget": float(budget),
                              "window_s": float(self.cfg.cont_window_s)},
                )
                if ev:
                    self._in_overload[j] = True
                    out.append(ev)
            else:
                if avg[j] < budget * self.cfg.cont_release_frac:
                    self._in_overload[j] = False
                    out.append(FailureEvent(
                        type=FailureType.OVERLOAD_CLEARED, joint_idx=j,
                        severity=0.0, t=now,
                        snapshot={"tau_avg_1s": float(avg[j]),
                                  "budget": float(budget)}))
        return out

    def _try_fire(
        self, ftype: FailureType, j: int, now: float,
        condition: bool, release: bool, severity: float, snapshot: dict,
    ) -> FailureEvent | None:
        ch = self._chan(ftype, j)
        if not ch.armed:
            if release:
                ch.armed = True
            return None
        if not condition:
            return None
        if now - ch.last_fire_t < self.cfg.refractory_s:
            return None
        ch.armed = False
        ch.last_fire_t = now
        return FailureEvent(
            type=ftype, joint_idx=j, severity=float(np.clip(severity, 0.0, 1.0)),
            t=now, snapshot=snapshot,
        )

    def _check_torque(self, a: dict) -> list[FailureEvent]:
        """lookback 창 안에서 min_duration 이상 지속된 초과 '구간'을 찾는다.

        말미 창만 보면 짧은 충격 과도신호(수십 ms)를 폴링 위상에 따라 놓친다.
        이미 발동에 쓰인 과거 구간의 재발동은 last_fire_t 이후에 끝난 구간만
        인정하는 것으로 막는다.
        """
        t, tau = a["t"], a["tau"]
        m = self._trailing(t, max(self.cfg.torque_lookback_s,
                                  self.cfg.torque_min_duration_s))
        if m.sum() < 2:
            return []
        t_win = t[m]
        out = []
        for j in range(self.n):
            thr = self._tau_thr[j]
            ch = self._chan(FailureType.TORQUE_SPIKE, j)
            win = np.abs(tau[m, j])
            above = win > thr

            # last_fire 이후에 끝난, min_duration 이상 지속된 초과 구간이 있는가
            condition = False
            peak = 0.0
            run_dur = 0.0
            for start, end in _runs(above):
                dur = float(t_win[end] - t_win[start])
                if dur >= self.cfg.torque_min_duration_s and t_win[end] > ch.last_fire_t:
                    condition = True
                    run_dur = max(run_dur, dur)
                    peak = max(peak, float(win[start:end + 1].max()))

            # 재무장: 최근 min_duration 동안 조용해졌는가
            tail = t_win >= t_win[-1] - self.cfg.torque_min_duration_s
            release = bool(win[tail].max() < thr * self.cfg.torque_release_frac)

            ev = self._try_fire(
                FailureType.TORQUE_SPIKE, j, float(t[-1]), condition, release,
                severity=min(1.0, (peak / thr - 1.0) * 2 + 0.5) if peak else 0.0,
                snapshot={
                    "tau_now": float(tau[-1, j]), "tau_peak_window": peak,
                    "threshold": float(thr), "duration_s": run_dur,
                },
            )
            if ev:
                out.append(ev)
        return out

    def _check_stall(self, a: dict) -> list[FailureEvent]:
        t, q, qd, q_des = a["t"], a["q"], a["qd"], a["q_des"]
        if not np.any(np.isfinite(q_des)):
            return []  # 지령이 기록 안 됐으면 판정 불가
        m = self._trailing(t, self.cfg.stall_min_duration_s)
        if m.sum() < 3:
            return []
        window_span = float(t[m][-1] - t[m][0])
        # |qd| < eps 인 창에서 관절이 실제로 진행할 수 있는 최대 거리.
        # 이보다 많이 줄었다면 (천천히라도) 가고 있는 것이므로 스톨이 아니다.
        max_progress = self.cfg.stall_qd_eps * window_span
        out = []
        for j in range(self.n):
            err = q_des[m, j] - q[m, j]
            if not np.all(np.isfinite(err)):
                continue
            abs_err = np.abs(err)
            speed = np.abs(qd[m, j])
            condition = bool(
                np.all(abs_err > self.cfg.stall_err_threshold)
                and np.all(speed < self.cfg.stall_qd_eps)
                and abs_err[-1] >= abs_err[0] - max_progress
            )
            release = bool(abs_err[-1] < self.cfg.stall_err_threshold * self.cfg.stall_release_frac)
            ev = self._try_fire(
                FailureType.STALL, j, float(t[-1]), condition, release,
                severity=min(1.0, float(abs_err[-1]) / (self.cfg.stall_err_threshold * 4)),
                snapshot={
                    "q_now": float(q[-1, j]), "q_des_now": float(q_des[-1, j]),
                    "err": float(err[-1]), "qd_mean_abs": float(speed.mean()),
                    "err_threshold": self.cfg.stall_err_threshold,
                    "duration_s": self.cfg.stall_min_duration_s,
                },
            )
            if ev:
                out.append(ev)
        return out

    def _check_oscillation(self, a: dict) -> list[FailureEvent]:
        t, qd = a["t"], a["qd"]
        m = self._trailing(t, self.cfg.osc_window_s)
        if m.sum() < 4:
            return []
        window_span = float(t[m][-1] - t[m][0])
        if window_span <= 0:
            return []
        out = []
        for j in range(self.n):
            v = qd[m, j]
            # 진폭이 작은 잔떨림은 무시: |qd| > amp_eps인 구간의 부호만 센다
            sign = np.where(np.abs(v) > self.cfg.osc_qd_amp_eps, np.sign(v), 0.0)
            nz = sign[sign != 0]
            flips = int(np.count_nonzero(np.diff(nz) != 0)) if nz.size >= 2 else 0
            flips_hz = flips / window_span
            condition = flips_hz >= self.cfg.osc_min_flips_hz
            release = flips_hz < self.cfg.osc_min_flips_hz * self.cfg.osc_release_frac
            ev = self._try_fire(
                FailureType.OSCILLATION, j, float(t[-1]), condition, release,
                severity=min(1.0, flips_hz / (self.cfg.osc_min_flips_hz * 3)),
                snapshot={
                    "flips_hz": float(flips_hz), "qd_amp_max": float(np.abs(v).max()),
                    "flips_hz_threshold": self.cfg.osc_min_flips_hz,
                    "window_s": self.cfg.osc_window_s,
                },
            )
            if ev:
                out.append(ev)
        return out
