"""링버퍼 로거.

최근 N초의 상태/명령/타이밍을 메모리에 들고 있다가,
dump_text()로 사람(과 LLM)이 읽는 요약 텍스트를 뱉는다.

출력 포맷은 "LLM에 그대로 붙여넣어 원인 분석을 시키는 용도"를 1순위로 잡았다:
  - 섹션 헤더가 대괄호로 구분되어 파싱/스캔이 쉽다
  - 숫자는 고정 소수점, 단위 명시
  - 임계치 초과 구간은 별도 섹션에 시작/끝 타임스탬프로 요약
  - 같은 입력이면 같은 출력 (벽시계 시각을 본문에 넣지 않는다)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from robot_core.hal.interface import JointCommand, JointState


@dataclass
class _Record:
    t: float                  # 상태 타임스탬프 [s]
    wall: float               # 루프 시작 기준 경과 시간 [s] (없으면 t와 동일)
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    q_des: np.ndarray | None
    tau_cmd: np.ndarray | None  # 지령 토크 kp·e + kd·ė + tau_ff (열 예산 신호)
    loop_dt: float | None     # 스텝 계산 소요 [s]
    overrun: bool


@dataclass
class _Event:
    t: float
    text: str


class RingLogger:
    """최근 window_sec 구간만 유지하는 링버퍼.

    Parameters
    ----------
    n_joints:
        관절 수.
    window_sec:
        유지할 시간 창 [s]. 이보다 오래된 샘플은 log() 때 버린다.
    torque_threshold:
        관절별 |tau| 경보 임계치 [Nm]. 스칼라/배열/None.
        None이면 임계치 섹션에서 "미설정"으로 표시한다.
    max_samples:
        메모리 상한(안전장치). 시간 창보다 먼저 걸리면 오래된 것부터 버린다.
    """

    def __init__(
        self,
        n_joints: int,
        window_sec: float = 5.0,
        torque_threshold=None,
        torque_clamp=None,
        max_samples: int = 200_000,
    ) -> None:
        self.n_joints = int(n_joints)
        self.window_sec = float(window_sec)
        self.max_samples = int(max_samples)
        self.torque_threshold = self._as_threshold(torque_threshold)
        # tau_cmd 기록용 클램프 (하드웨어/소프트웨어 포화 한계). 지령이 잘리는
        # 구간에선 지령 ≠ 실제 전류이므로, 열 예산 신호는 클램프 후 값이어야
        # 과대평가(불필요한 slow-down)를 피한다.
        self.torque_clamp = self._as_threshold(torque_clamp)

        self._records: deque[_Record] = deque()
        self._events: deque[_Event] = deque(maxlen=200)

    @classmethod
    def for_hal(cls, hal, window_sec: float = 5.0, threshold_frac: float = 0.8) -> "RingLogger":
        """HAL의 torque_limits * threshold_frac 을 임계치로, limits를 클램프로."""
        limits = np.asarray(hal.torque_limits, dtype=float)
        return cls(
            n_joints=hal.n_joints,
            window_sec=window_sec,
            torque_threshold=limits * float(threshold_frac),
            torque_clamp=limits,
        )

    def _as_threshold(self, value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n_joints, float(arr))
        if arr.shape != (self.n_joints,):
            raise ValueError(f"torque_threshold shape {arr.shape}, expected ({self.n_joints},)")
        return arr.copy()

    # ------------------------------------------------------------------ 기록
    def log(
        self,
        state: JointState,
        cmd: JointCommand | None = None,
        loop_dt: float | None = None,
        overrun: bool = False,
        wall_time: float | None = None,
    ) -> None:
        # 지령 토크 = kp(q_des-q) + kd(qd_des-qd) + tau_ff.
        # 지속 과부하(열 예산) 감시는 이 신호로 한다 — 외란과 맞서는 홀드에서
        # 출력측 순토크(tau_measured)는 0이지만 모터 전류(발열)는 지령 쪽에
        # 그대로 흐른다. 실물에서도 지령은 항상 알고 있으므로 동일하게 동작.
        tau_cmd = None
        if cmd is not None:
            tau_cmd = (np.asarray(cmd.kp, dtype=float) * (cmd.q_des - state.q)
                       + np.asarray(cmd.kd, dtype=float) * (cmd.qd_des - state.qd)
                       + np.asarray(cmd.tau_ff, dtype=float))
            if self.torque_clamp is not None:
                # 클램프 초과분은 실제 전류로 흐르지 않는다 — 잘린 뒤 값이 발열
                np.clip(tau_cmd, -self.torque_clamp, self.torque_clamp, out=tau_cmd)
        rec = _Record(
            t=float(state.timestamp),
            wall=float(state.timestamp if wall_time is None else wall_time),
            q=np.asarray(state.q, dtype=float).copy(),
            qd=np.asarray(state.qd, dtype=float).copy(),
            tau=np.asarray(state.tau_measured, dtype=float).copy(),
            q_des=None if cmd is None else np.asarray(cmd.q_des, dtype=float).copy(),
            tau_cmd=tau_cmd,
            loop_dt=None if loop_dt is None else float(loop_dt),
            overrun=bool(overrun),
        )
        self._records.append(rec)
        self._prune()

    def mark_event(self, text: str, t: float | None = None) -> None:
        """타임라인에 사람이 읽을 메모를 남긴다 (예: 'jam injected on joint 2')."""
        if t is None:
            t = self._records[-1].t if self._records else 0.0
        self._events.append(_Event(t=float(t), text=str(text)))

    def events(self) -> list:
        """mark_event()로 기록된 (t, text) 이벤트 목록 (통합 타임라인용)."""
        return list(self._events)

    def clear(self) -> None:
        self._records.clear()
        self._events.clear()

    def __len__(self) -> int:
        return len(self._records)

    def _prune(self) -> None:
        if not self._records:
            return
        t_new = self._records[-1].t
        cutoff = t_new - self.window_sec
        while self._records and self._records[0].t < cutoff:
            self._records.popleft()
        while len(self._records) > self.max_samples:
            self._records.popleft()

    # ------------------------------------------------------------------ 조회
    def _window(self, window_sec: float | None) -> list[_Record]:
        if not self._records:
            return []
        if window_sec is None:
            return list(self._records)
        cutoff = self._records[-1].t - float(window_sec)
        return [r for r in self._records if r.t >= cutoff]

    def to_arrays(self, window_sec: float | None = None) -> dict:
        """분석용 원본 배열. keys: t, wall, q, qd, tau, q_des, tau_cmd, loop_dt, overrun."""
        recs = self._window(window_sec)
        if not recs:
            empty2 = np.zeros((0, self.n_joints))
            return {
                "t": np.zeros(0), "wall": np.zeros(0), "q": empty2, "qd": empty2,
                "tau": empty2, "q_des": empty2, "tau_cmd": empty2,
                "loop_dt": np.zeros(0), "overrun": np.zeros(0, dtype=bool),
            }
        nanrow = np.full(self.n_joints, np.nan)
        return {
            "t": np.array([r.t for r in recs]),
            "wall": np.array([r.wall for r in recs]),
            "q": np.stack([r.q for r in recs]),
            "qd": np.stack([r.qd for r in recs]),
            "tau": np.stack([r.tau for r in recs]),
            "q_des": np.stack([r.q_des if r.q_des is not None else nanrow for r in recs]),
            "tau_cmd": np.stack([r.tau_cmd if r.tau_cmd is not None else nanrow for r in recs]),
            "loop_dt": np.array([np.nan if r.loop_dt is None else r.loop_dt for r in recs]),
            "overrun": np.array([r.overrun for r in recs], dtype=bool),
        }

    # ------------------------------------------------------------------ 출력
    def dump_text(self, window_sec: float | None = None, max_excursions: int = 20) -> str:
        """사람/LLM이 읽는 요약. 같은 입력이면 항상 같은 출력."""
        recs = self._window(window_sec)
        rule = "=" * 78
        if not recs:
            return f"{rule}\nROBOT STATE DUMP | (no samples)\n{rule}"

        d = self._arrays_from(recs)
        t0, t1 = d["t"][0], d["t"][-1]
        span = t1 - t0
        lines: list[str] = []

        lines.append(rule)
        lines.append(
            f"ROBOT STATE DUMP | {len(recs)} samples | {self.n_joints} joints | "
            f"window {span:.3f} s"
        )
        lines.append(f"time span: t = {t0:.3f} s .. {t1:.3f} s   (robot clock)")
        lines.append("units: q,q_des [rad]  qd [rad/s]  tau [Nm]  time [s]")
        lines.append(rule)

        lines.append("")
        lines.extend(self._section_timing(d, span))
        lines.append("")
        lines.extend(self._section_torque(d))
        lines.append("")
        lines.extend(self._section_tracking(d))
        lines.append("")
        lines.extend(self._section_excursions(d, max_excursions))
        lines.append("")
        lines.extend(self._section_events(t0, t1))
        lines.append("")
        lines.extend(self._section_last(d))
        lines.append(rule)
        return "\n".join(lines)

    def _arrays_from(self, recs: list[_Record]) -> dict:
        return {
            "t": np.array([r.t for r in recs]),
            "wall": np.array([r.wall for r in recs]),
            "q": np.stack([r.q for r in recs]),
            "qd": np.stack([r.qd for r in recs]),
            "tau": np.stack([r.tau for r in recs]),
            "q_des": np.stack(
                [r.q_des if r.q_des is not None else np.full(self.n_joints, np.nan) for r in recs]
            ),
            "loop_dt": np.array([np.nan if r.loop_dt is None else r.loop_dt for r in recs]),
            "overrun": np.array([r.overrun for r in recs], dtype=bool),
        }

    def _section_timing(self, d: dict, span: float) -> list[str]:
        out = ["[TIMING]"]
        n = len(d["t"])
        wall = d["wall"]
        monotonic = n >= 2 and np.all(np.isfinite(wall)) and np.all(np.diff(wall) > 0)
        if n >= 2 and np.all(np.isfinite(wall)) and not monotonic:
            # 시계가 뒤로 갔다 = 로그 소스가 섞였다는 뜻. 엉터리 통계를 내느니 그렇다고 말한다.
            out.append(
                "  rate       : (unavailable — wall clock not monotonic; "
                "여러 러너/세션의 로그가 섞였는지 확인할 것)"
            )
        if monotonic:
            per = np.diff(wall) * 1e3
            hz = (n - 1) / (wall[-1] - wall[0]) if wall[-1] > wall[0] else float("nan")
            out.append(
                f"  rate       : {hz:8.1f} Hz achieved   "
                f"(period mean {per.mean():.3f} ms, jitter(std) {per.std():.3f} ms)"
            )
            out.append(
                f"  period     : min {per.min():.3f} | p50 {np.percentile(per, 50):.3f} | "
                f"p95 {np.percentile(per, 95):.3f} | max {per.max():.3f}  [ms]"
            )
        elif not (n >= 2 and np.all(np.isfinite(wall))):
            out.append(f"  rate       : {n} samples over {span:.3f} s (period stats unavailable)")

        dt = d["loop_dt"]
        if np.any(np.isfinite(dt)):
            v = dt[np.isfinite(dt)] * 1e3
            out.append(f"  compute    : mean {v.mean():.3f} ms | max {v.max():.3f} ms")
        n_over = int(d["overrun"].sum())
        pct = 100.0 * n_over / n if n else 0.0
        flag = "  <-- CHECK" if pct > 1.0 else ""
        out.append(f"  overruns   : {n_over} / {n} ({pct:.2f}%){flag}")
        return out

    def _section_torque(self, d: dict) -> list[str]:
        tau, t = d["tau"], d["t"]
        thr = self.torque_threshold
        out = ["[PER-JOINT TORQUE]  tau_measured  (peak = |tau|가 최대인 순간의 부호 있는 값)"]
        out.append(
            "  jnt |    mean |     rms |    peak |  at t   | thresh |  status"
        )
        out.append("  " + "-" * 68)
        for j in range(self.n_joints):
            col = tau[:, j]
            k = int(np.argmax(np.abs(col)))
            peak = col[k]
            rms = float(np.sqrt(np.mean(col**2)))
            if thr is None:
                thr_s, status = "     -- ", "no-threshold"
            else:
                thr_s = f"{thr[j]:7.2f} "
                over = int(np.sum(np.abs(col) > thr[j]))
                if over:
                    status = f"EXCEEDED ({over} samples, {100.0 * over / len(col):.1f}%)"
                else:
                    status = f"ok ({100.0 * abs(peak) / thr[j]:.0f}% of thresh)" if thr[j] > 0 else "ok"
            out.append(
                f"  {j:3d} | {col.mean():7.3f} | {rms:7.3f} | {peak:7.3f} | "
                f"{t[k]:7.3f} |{thr_s}| {status}"
            )
        return out

    def _section_tracking(self, d: dict) -> list[str]:
        q, qd, q_des = d["q"], d["qd"], d["q_des"]
        has_cmd = bool(np.any(np.isfinite(q_des)))
        out = ["[PER-JOINT TRACKING]"]
        if not has_cmd:
            out.append("  (no commands logged — q_des unavailable)")
            out.append("  jnt |   q_now |  qd_now |  |qd|max")
            out.append("  " + "-" * 40)
            for j in range(self.n_joints):
                out.append(
                    f"  {j:3d} | {q[-1, j]:7.3f} | {qd[-1, j]:7.3f} | {np.abs(qd[:, j]).max():8.3f}"
                )
            return out

        err = q_des - q
        out.append("  jnt |   q_now | qdes_now | err_now | err_max |  |qd|max")
        out.append("  " + "-" * 58)
        for j in range(self.n_joints):
            e = err[:, j]
            e_fin = e[np.isfinite(e)]
            e_max = float(np.abs(e_fin).max()) if e_fin.size else float("nan")
            out.append(
                f"  {j:3d} | {q[-1, j]:7.3f} | {q_des[-1, j]:8.3f} | "
                f"{e[-1]:7.3f} | {e_max:7.3f} | {np.abs(qd[:, j]).max():9.3f}"
            )
        return out

    def _section_excursions(self, d: dict, max_excursions: int) -> list[str]:
        out = ["[THRESHOLD EXCURSIONS]  |tau| > threshold"]
        if self.torque_threshold is None:
            out.append("  (threshold not set — RingLogger(torque_threshold=...) 로 지정)")
            return out

        tau, t = d["tau"], d["t"]
        segments = []
        for j in range(self.n_joints):
            mask = np.abs(tau[:, j]) > self.torque_threshold[j]
            for start, end in _mask_segments(mask):
                seg_tau = tau[start : end + 1, j]
                k = int(np.argmax(np.abs(seg_tau)))
                segments.append(
                    {
                        "joint": j,
                        "t0": t[start],
                        "t1": t[end],
                        "n": end - start + 1,
                        "peak": seg_tau[k],
                        "t_peak": t[start + k],
                        "ongoing": end == len(t) - 1,
                    }
                )
        if not segments:
            out.append("  (none)")
            return out

        segments.sort(key=lambda s: (-abs(s["peak"]), s["joint"], s["t0"]))
        for s in segments[:max_excursions]:
            tag = "  [ONGOING]" if s["ongoing"] else ""
            out.append(
                f"  joint {s['joint']}: t={s['t0']:.3f}..{s['t1']:.3f} s "
                f"({s['t1'] - s['t0']:.3f} s, {s['n']} samples) "
                f"peak {s['peak']:+.3f} Nm @ t={s['t_peak']:.3f}"
                f" (thresh {self.torque_threshold[s['joint']]:.2f}){tag}"
            )
        if len(segments) > max_excursions:
            out.append(f"  ... and {len(segments) - max_excursions} more (peak 순 정렬, 상위만 표시)")
        return out

    def _section_events(self, t0: float, t1: float) -> list[str]:
        out = ["[EVENTS]"]
        evs = [e for e in self._events if t0 <= e.t <= t1]
        if not evs:
            out.append("  (none)")
            return out
        for e in evs:
            out.append(f"  t={e.t:.3f}  {e.text}")
        return out

    def _section_last(self, d: dict) -> list[str]:
        fmt = lambda v: " ".join(f"{x:8.3f}" for x in v)
        out = [f"[LAST SAMPLE]  t={d['t'][-1]:.3f} s"]
        out.append(f"  q      : {fmt(d['q'][-1])}")
        out.append(f"  qd     : {fmt(d['qd'][-1])}")
        out.append(f"  tau    : {fmt(d['tau'][-1])}")
        if np.any(np.isfinite(d["q_des"][-1])):
            out.append(f"  q_des  : {fmt(d['q_des'][-1])}")
        return out


def _mask_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """True인 연속 구간을 (start_idx, end_idx) 리스트로 (양끝 포함)."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))
