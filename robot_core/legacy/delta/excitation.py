"""여기(excitation) 궤적 생성기 — 마찰·백래시를 드러내는 안전한 탐침 궤적.

세 종류의 세그먼트를 시간 예산 안에 자동 배분한다:
- 사인 스윕      : 여러 진폭×주파수 — 마찰의 속도 의존성(점성 항) 커버
- 방향 반전 반복  : 백래시는 반전 순간에만 보인다. 이게 핵심 데이터
- 준정적 램프    : 일정 저속 왕복 — Coulomb 상수(tau_c) 추출용

전부 관절/속도 한계의 보수적 비율(safety_frac, 기본 50%) 안에서만 만들어지고,
build 직후 validate()가 해석적 피크값으로 재검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ------------------------------------------------------------------ 세그먼트
class Segment:
    """단일 관절 여기 프리미티브. rest를 중심으로 한 상대 궤적."""

    kind = "base"
    duration: float

    def sample(self, t: float | np.ndarray):
        """(dq, dqd): rest 대비 오프셋과 속도."""
        raise NotImplementedError

    @property
    def peak_offset(self) -> float:
        raise NotImplementedError

    @property
    def peak_speed(self) -> float:
        raise NotImplementedError

    @property
    def n_reversals(self) -> int:
        return 0


class SineSegment(Segment):
    kind = "sine_sweep"

    def __init__(self, amp: float, freq: float, n_cycles: int) -> None:
        self.amp = float(amp)
        self.freq = float(freq)
        self.n_cycles = int(n_cycles)
        self.duration = self.n_cycles / self.freq

    def sample(self, t):
        w = 2 * np.pi * self.freq
        return self.amp * np.sin(w * t), self.amp * w * np.cos(w * t)

    @property
    def peak_offset(self):
        return self.amp

    @property
    def peak_speed(self):
        return self.amp * 2 * np.pi * self.freq

    @property
    def n_reversals(self):
        return 2 * self.n_cycles


class ReversalSegment(SineSegment):
    """소진폭 사인 반복 — 데드존 폭의 몇 배만 왕복하며 반전을 다량 생산."""

    kind = "reversal"


class RampCycleSegment(Segment):
    """준정적 삼각파: |qd| = speed 일정, ±amp 왕복. Coulomb 상수 추출용."""

    kind = "quasi_static_ramp"

    def __init__(self, amp: float, speed: float, n_cycles: int) -> None:
        self.amp = float(amp)
        self.speed = float(speed)
        self.n_cycles = int(n_cycles)
        self._period = 4.0 * self.amp / self.speed  # 0→+A→-A→0
        self.duration = self._period * self.n_cycles

    def sample(self, t):
        t = np.asarray(t, dtype=float)
        ph = np.mod(t, self._period) / self._period      # 0..1
        # 0→A (0..1/4), A→-A (1/4..3/4), -A→0 (3/4..1)
        tri = np.where(ph < 0.25, 4 * ph,
                       np.where(ph < 0.75, 2 - 4 * ph, 4 * ph - 4))
        sign = np.where(ph < 0.25, 1.0, np.where(ph < 0.75, -1.0, 1.0))
        dq = self.amp * tri
        dqd = self.speed * sign
        if np.ndim(t) == 0:
            return float(dq), float(dqd)
        return dq, dqd

    @property
    def peak_offset(self):
        return self.amp

    @property
    def peak_speed(self):
        return self.speed

    @property
    def n_reversals(self):
        return 2 * self.n_cycles


# ---------------------------------------------------------------- 계획/설정
@dataclass
class ExcitationConfig:
    safety_frac: float = 0.5          # 관절/속도 한계 대비 사용 비율 (보수적)
    qd_limit: float = 4.0             # [rad/s] 관절 속도 한계 (실물 스펙으로 교체)
    mix: dict = field(default_factory=lambda: {
        "sine_sweep": 0.4, "reversal": 0.3, "quasi_static_ramp": 0.3})
    sweep_freqs: tuple = (0.3, 0.7, 1.4)   # [Hz]
    sweep_amp_fracs: tuple = (0.35, 0.7)   # 허용 진폭 대비
    reversal_amp: float = 0.06             # [rad] 데드존(수 mrad~수십 mrad)의 수 배
    reversal_freq: float = 1.0             # [Hz]
    quasi_static_speed: float = 0.1        # [rad/s]
    ramp_amp_frac: float = 0.5
    settle_s: float = 0.5                  # 세그먼트 사이 정지 시간


@dataclass
class _Window:
    t_start: float
    t_end: float
    joint: int
    segment: Segment


class ExcitationPlan:
    """sample(t) -> (q_des, qd_des) 전체 벡터. 비활성 관절은 rest 홀드."""

    def __init__(self, n_joints: int, rest: np.ndarray, windows: list[_Window],
                 config: ExcitationConfig) -> None:
        self.n_joints = n_joints
        self.rest = rest
        self.windows = windows
        self.config = config
        self.duration = max((w.t_end for w in windows), default=0.0) + config.settle_s

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        q_des = self.rest.copy()
        qd_des = np.zeros(self.n_joints)
        for w in self.windows:
            if w.t_start <= t < w.t_end:
                dq, dqd = w.segment.sample(t - w.t_start)
                q_des[w.joint] = self.rest[w.joint] + dq
                qd_des[w.joint] = dqd
        return q_des, qd_des

    # ------------------------------------------------------------- 검증/요약
    def validate(self, joint_limits: np.ndarray, qd_limit: float | np.ndarray,
                 margin: float = 1.0) -> list[str]:
        """해석적 피크값으로 한계 검사. margin<1이면 그만큼 더 보수적으로 본다."""
        jl = np.asarray(joint_limits, dtype=float)
        vmax = np.broadcast_to(np.asarray(qd_limit, dtype=float), (self.n_joints,))
        problems = []
        for w in self.windows:
            j = w.joint
            lo, hi = jl[j]
            top = self.rest[j] + w.segment.peak_offset
            bot = self.rest[j] - w.segment.peak_offset
            # margin<1 이면 rest 기준으로 그 비율만큼만 허용 (더 보수적 검사)
            hi_allow = self.rest[j] + margin * (hi - self.rest[j])
            lo_allow = self.rest[j] - margin * (self.rest[j] - lo)
            if top > hi_allow or bot < lo_allow:
                problems.append(
                    f"joint {j} {w.segment.kind}: range [{bot:.3f}, {top:.3f}] vs "
                    f"allowed [{lo_allow:.3f}, {hi_allow:.3f}] (margin {margin})")
            if w.segment.peak_speed > vmax[j] * margin:
                problems.append(
                    f"joint {j} {w.segment.kind}: peak speed {w.segment.peak_speed:.3f} "
                    f"> {vmax[j] * margin:.3f}")
        return problems

    def summary(self) -> str:
        lines = [f"[EXCITATION PLAN] {len(self.windows)} segments, "
                 f"total {self.duration:.1f} s"]
        for w in self.windows:
            s = w.segment
            lines.append(
                f"  t={w.t_start:7.1f}..{w.t_end:7.1f}  joint {w.joint}  {s.kind:18s}"
                f" peak_dq={s.peak_offset:.3f} rad  peak_qd={s.peak_speed:.3f} rad/s"
                f"  reversals={s.n_reversals}")
        return "\n".join(lines)

    def expected_reversals(self, joint: int) -> int:
        return sum(w.segment.n_reversals for w in self.windows if w.joint == joint)


def build_excitation(
    joints: list[int],
    joint_limits: np.ndarray,
    budget_s: float = 480.0,
    rest: np.ndarray | None = None,
    config: ExcitationConfig | None = None,
    n_joints: int | None = None,
    mode: str = "sequential",
) -> ExcitationPlan:
    """시간 예산을 세그먼트 믹스에 자동 배분해 계획을 만든다.

    mode="sequential": 관절을 하나씩 (예산을 관절 수로 분할). 안전 기본값.
    mode="parallel"  : 전 관절 동시 (각 관절이 전체 예산 사용. 빠르지만 과감)
    """
    cfg = config or ExcitationConfig()
    jl = np.asarray(joint_limits, dtype=float)
    n = n_joints or len(jl)
    rest = np.zeros(n) if rest is None else np.asarray(rest, dtype=float)
    if mode not in ("sequential", "parallel"):
        raise ValueError('mode must be "sequential" or "parallel"')
    if not joints:
        raise ValueError("no joints selected")

    per_joint_budget = budget_s / len(joints) if mode == "sequential" else budget_s
    windows: list[_Window] = []
    t_cursor = 0.0

    for joint in joints:
        lo, hi = jl[joint]
        # rest에서 양쪽으로 안전하게 쓸 수 있는 진폭
        amp_allow = cfg.safety_frac * min(hi - rest[joint], rest[joint] - lo)
        if amp_allow <= 0:
            raise ValueError(f"joint {joint}: rest pose leaves no safe amplitude")
        qd_allow = cfg.safety_frac * cfg.qd_limit

        t_j = t_cursor if mode == "sequential" else 0.0
        segs = _build_joint_program(per_joint_budget, amp_allow, qd_allow, cfg)
        for seg in segs:
            windows.append(_Window(t_j, t_j + seg.duration, joint, seg))
            t_j += seg.duration + cfg.settle_s
        if mode == "sequential":
            t_cursor = t_j

    plan = ExcitationPlan(n, rest, windows, cfg)
    problems = plan.validate(jl, cfg.qd_limit, margin=1.0)
    if problems:  # 생성기 버그를 조기에 잡는 자기 검증
        raise AssertionError("generated plan violates limits:\n  " + "\n  ".join(problems))
    return plan


def _build_joint_program(budget: float, amp_allow: float, qd_allow: float,
                         cfg: ExcitationConfig) -> list[Segment]:
    segs: list[Segment] = []

    # --- 사인 스윕: 진폭×주파수 그리드, 각 조합에 균등 시간 ---
    t_sweep = budget * cfg.mix["sine_sweep"]
    combos = []
    for af in cfg.sweep_amp_fracs:
        for f in cfg.sweep_freqs:
            amp = min(af * amp_allow, qd_allow / (2 * np.pi * f))
            if amp > 1e-3:
                combos.append((amp, f))
    if combos:
        per = t_sweep / len(combos)
        for amp, f in combos:
            n_cyc = max(1, int(per * f))
            segs.append(SineSegment(amp, f, n_cyc))

    # --- 방향 반전: 소진폭 사인 다회 반복 ---
    t_rev = budget * cfg.mix["reversal"]
    amp = min(cfg.reversal_amp, amp_allow)
    f = min(cfg.reversal_freq, qd_allow / (2 * np.pi * amp))
    n_cyc = max(2, int(t_rev * f))
    segs.append(ReversalSegment(amp, f, n_cyc))

    # --- 준정적 램프 ---
    t_ramp = budget * cfg.mix["quasi_static_ramp"]
    speed = min(cfg.quasi_static_speed, qd_allow)
    amp = min(cfg.ramp_amp_frac * amp_allow, speed * t_ramp / 4.0)
    if amp > 1e-3:
        period = 4.0 * amp / speed
        n_cyc = max(1, int(t_ramp / period))
        segs.append(RampCycleSegment(amp, speed, n_cyc))

    return segs
