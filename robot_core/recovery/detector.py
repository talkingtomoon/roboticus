"""실패 감지기 — phorce 피드백 프레임(FeedbackCache.to_arrays()) 기준.

네 종류를 감지한다:
- PLAYBACK_STALL : 재생 중인데 전 축이 정지 + 전류는 밀고 있음 (물체에 막힘)
- IMPACT         : dob_a(외란 관측기) 스파이크 — 자체 추정 코드는 제거했다,
                   하드웨어 DOB가 정본이다
- OVERHEAT       : temp_c 임계 초과 지속 (연속 과부하의 phorce 버전).
                   해제 시 OVERHEAT_CLEARED를 짝으로 발행 (복귀 트리거)
- AXIS_FAULT     : fault 비트 발화

모든 판정의 전제: valid 마스크. valid=False(또는 stale) 축은 수치를 신뢰할 수
없으므로 **판정에서 제외**한다 (없는 셈 친다 — 오검출도 미검출도 아닌 '모름').

오탐 방지 구조는 구 감지기에서 그대로 가져왔다:
- 최소 지속 시간 / lookback 구간 스캔 (폴링 위상 무관)
- 히스테리시스 (release 수준 밑으로 내려가야 재무장)
- refractory (같은 (타입,축) 재발동 최소 간격)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_core.recovery.events import FailureEvent, FailureType


@dataclass
class DetectorConfig:
    # IMPACT — dob_a 스파이크 (임계 기본값은 목 스케일. 실물은
    # scripts/field_smoke.py가 30초 분포를 재서 제안하는 값으로 교체)
    impact_dob_threshold: float | np.ndarray = 3.0   # [A]
    impact_min_duration_s: float = 0.008
    impact_release_frac: float = 0.7
    # lookback은 **판단 루프 주기(~0.5s)보다 길어야 한다** — 2Hz 폴 사이에
    # 끝나버린 짧은 충격도 다음 폴에서 구간 스캔으로 잡는다 (last_fire_t
    # 가드가 같은 구간의 재발동을 막는다)
    impact_lookback_s: float = 0.7

    # PLAYBACK_STALL — 재생 중 + 정지 + 전류로 밀고 있음
    stall_vel_eps: float = 0.05        # [rad/s] "안 움직임"
    stall_current_floor: float = 1.0   # [A] "밀고 있음" (없으면 의도된 정지 동작)
    stall_min_duration_s: float = 0.30

    # OVERHEAT — 지속 판정 (스파이크가 아니라 열)
    overheat_temp_c: float = 70.0
    overheat_min_duration_s: float = 0.5
    overheat_release_c: float = 62.0   # 이 밑으로 내려와야 해제(CLEARED 발행)

    # AXIS_FAULT — 즉시 (비트가 곧 판정)
    fault_min_duration_s: float = 0.003

    # 공통
    refractory_s: float = 1.0
    valid_min_frames: int = 5          # 창 안에서 이보다 적게 valid면 그 축 판정 보류


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
    """check(arrays)를 느린 루프(~2Hz)에서 불러 이벤트 리스트를 받는다."""

    def __init__(self, n_axes: int = 12, config: DetectorConfig | None = None) -> None:
        self.n = int(n_axes)
        self.cfg = config or DetectorConfig()
        thr = np.asarray(self.cfg.impact_dob_threshold, dtype=float)
        self._dob_thr = np.full(self.n, float(thr)) if thr.ndim == 0 else thr.copy()
        if self._dob_thr.shape != (self.n,):
            raise ValueError(f"impact_dob_threshold shape {thr.shape} != ({self.n},)")
        self._state: dict[tuple[str, int], _ChannelState] = {}
        self._in_overheat = np.zeros(self.n, dtype=bool)

    def _chan(self, ftype: FailureType, j: int) -> _ChannelState:
        return self._state.setdefault((ftype.value, j), _ChannelState())

    def reset(self) -> None:
        self._state.clear()
        self._in_overheat[:] = False

    # ------------------------------------------------------------------ 판정
    def check(self, arrays: dict) -> list[FailureEvent]:
        t = arrays["t"]
        if len(t) < 3:
            return []
        events: list[FailureEvent] = []
        events += self._check_impact(arrays)
        events += self._check_playback_stall(arrays)
        events += self._check_overheat(arrays)
        events += self._check_axis_fault(arrays)
        return events

    def _trailing(self, t: np.ndarray, duration: float) -> np.ndarray:
        return t >= (t[-1] - duration)

    def _axis_ok(self, valid_col: np.ndarray) -> bool:
        """창 안에서 이 축을 판정에 써도 되는가 (valid 전제)."""
        return int(valid_col.sum()) >= self.cfg.valid_min_frames

    def _try_fire(self, ftype: FailureType, j: int, now: float,
                  condition: bool, release: bool, severity: float,
                  snapshot: dict) -> FailureEvent | None:
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
        return FailureEvent(type=ftype, joint_idx=j, severity=min(1.0, severity),
                            t=now, snapshot=snapshot)

    # ------------------------------------------------------------- IMPACT
    def _check_impact(self, a: dict) -> list[FailureEvent]:
        """lookback 창에서 min_duration 이상 지속된 |dob| 초과 구간을 찾는다."""
        t, dob, valid = a["t"], a["dob"], a["valid"]
        m = self._trailing(t, max(self.cfg.impact_lookback_s,
                                  self.cfg.impact_min_duration_s))
        t_win = t[m]
        out = []
        for j in range(self.n):
            v = valid[m, j]
            if not self._axis_ok(v):
                continue
            thr = self._dob_thr[j]
            ch = self._chan(FailureType.IMPACT, j)
            sig = np.where(v, np.abs(dob[m, j]), 0.0)   # invalid 샘플은 0 취급
            above = sig > thr

            condition, peak, run_dur = False, 0.0, 0.0
            for s, e in _runs(above):
                dur = float(t_win[e] - t_win[s])
                if dur >= self.cfg.impact_min_duration_s and t_win[e] > ch.last_fire_t:
                    condition = True
                    run_dur = max(run_dur, dur)
                    peak = max(peak, float(sig[s:e + 1].max()))

            tail = t_win >= t_win[-1] - self.cfg.impact_min_duration_s
            release = bool(sig[tail].max() < thr * self.cfg.impact_release_frac)

            ev = self._try_fire(
                FailureType.IMPACT, j, float(t[-1]), condition, release,
                severity=(peak / thr - 1.0) * 2 + 0.5 if peak else 0.0,
                snapshot={"dob_peak": peak, "threshold": float(thr),
                          "duration_s": run_dur,
                          "dob_now": float(dob[-1, j]) if valid[-1, j] else float("nan")})
            if ev:
                out.append(ev)
        return out

    # ------------------------------------------------------ PLAYBACK_STALL
    def _check_playback_stall(self, a: dict) -> list[FailureEvent]:
        """재생 중인데 valid 전 축이 정지 + 전류는 밀고 있음 → 막힘.

        축 단위가 아니라 '프레임 단위' 판정이다: 모션에는 의도된 개별 축 정지가
        흔하므로, "전류가 흐르는데 아무 축도 안 움직인다"가 막힘의 신호다.
        발화는 전류가 가장 큰 축(=미는 축)에 귀속시킨다.
        """
        t = a["t"]
        m = self._trailing(t, self.cfg.stall_min_duration_s)
        if m.sum() < 5 or not bool(a["playing"][m].all()):
            return []   # 창 전체가 재생 중이어야 판정
        vel, cur, valid = a["velocity"][m], a["current"][m], a["valid"][m]

        # invalid 축 제외한 프레임별 최대 속도/전류
        vel_ok = np.where(valid, np.abs(vel), 0.0)
        cur_ok = np.where(valid, np.abs(cur), 0.0)
        if not self._axis_ok(valid.any(axis=1)):
            return []
        frozen = bool((vel_ok.max(axis=1) < self.cfg.stall_vel_eps).all())
        pushing = bool((cur_ok.max(axis=1) > self.cfg.stall_current_floor).all())

        j = int(cur_ok[-1].argmax())
        release = not frozen
        ev = self._try_fire(
            FailureType.PLAYBACK_STALL, j, float(t[-1]),
            condition=frozen and pushing, release=release,
            severity=0.5 + 0.5 * min(1.0, cur_ok[-1].max()
                                     / max(self.cfg.stall_current_floor, 1e-9) - 1.0),
            snapshot={"vel_max": float(vel_ok.max()),
                      "current_max": float(cur_ok.max()),
                      "vel_eps": self.cfg.stall_vel_eps,
                      "current_floor": self.cfg.stall_current_floor,
                      "duration_s": self.cfg.stall_min_duration_s})
        return [ev] if ev else []

    # ------------------------------------------------------------ OVERHEAT
    def _check_overheat(self, a: dict) -> list[FailureEvent]:
        t, temp, valid = a["t"], a["temp"], a["valid"]
        m = self._trailing(t, self.cfg.overheat_min_duration_s)
        if m.sum() < 3:
            return []
        out = []
        now = float(t[-1])
        for j in range(self.n):
            v = valid[m, j]
            if not self._axis_ok(v):
                continue
            col = temp[m, j][v]
            hot = bool((col > self.cfg.overheat_temp_c).all())
            cooled = bool(col[-1] < self.cfg.overheat_release_c)

            if not self._in_overheat[j]:
                ev = self._try_fire(
                    FailureType.OVERHEAT, j, now, condition=hot, release=True,
                    severity=0.5 + (float(col[-1]) - self.cfg.overheat_temp_c) / 20.0,
                    snapshot={"temp_now": float(col[-1]),
                              "threshold_c": self.cfg.overheat_temp_c,
                              "release_c": self.cfg.overheat_release_c})
                if ev:
                    self._in_overheat[j] = True
                    out.append(ev)
            elif cooled:
                self._in_overheat[j] = False
                out.append(FailureEvent(
                    type=FailureType.OVERHEAT_CLEARED, joint_idx=j, severity=0.0,
                    t=now, snapshot={"temp_now": float(col[-1]),
                                     "release_c": self.cfg.overheat_release_c}))
        return out

    def overheat_active(self) -> bool:
        return bool(self._in_overheat.any())

    # ---------------------------------------------------------- AXIS_FAULT
    def _check_axis_fault(self, a: dict) -> list[FailureEvent]:
        t, fault, valid = a["t"], a["fault"], a["valid"]
        m = self._trailing(t, max(self.cfg.fault_min_duration_s, 0.003))
        out = []
        for j in range(self.n):
            # fault 비트는 valid와 무관하게 신뢰 (드라이버가 직접 세우는 비트)
            active = bool(fault[m, j].all()) and m.sum() >= 2
            release = not bool(fault[-1, j])
            ev = self._try_fire(
                FailureType.AXIS_FAULT, j, float(t[-1]), condition=active,
                release=release, severity=1.0,
                snapshot={"fault": True, "valid_now": bool(valid[-1, j])})
            if ev:
                out.append(ev)
        return out
