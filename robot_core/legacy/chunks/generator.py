"""청크 생성기 — 현장에서 미션 보고 빠르게 딕셔너리를 채우는 도구.

전부 순수 기하: min-jerk 점대점, 경유점 우회, 시간/진폭 스케일링, 후퇴 청크.
생성 직후 validate_chunk()로 한계 검사를 돌리는 것을 습관화할 것
(생성 함수들에 limits를 넘기면 위반 시 바로 ValueError).
"""

from __future__ import annotations

import numpy as np

from robot_core.legacy.chunks.format import MotionChunk


# ------------------------------------------------------------------ 한계 검사
def validate_chunk(
    chunk: MotionChunk,
    joint_limits: np.ndarray | None = None,   # (n, 2) [min, max]
    qd_max: np.ndarray | float | None = None, # (n,) 또는 스칼라 [rad/s]
    dt: float = 2e-3,
) -> list[str]:
    """위반 사항 리스트를 돌려준다 (빈 리스트 = 통과). 조밀 샘플링 검사."""
    ts = np.arange(0.0, chunk.duration + dt, dt)
    q, qd, _ = chunk.sample(ts)
    problems: list[str] = []

    if joint_limits is not None:
        jl = np.asarray(joint_limits, dtype=float)
        for j in range(chunk.n_joints):
            lo, hi = jl[j]
            qmin, qmax = q[:, j].min(), q[:, j].max()
            if qmin < lo or qmax > hi:
                problems.append(
                    f"joint {j}: position range [{qmin:.3f}, {qmax:.3f}] exceeds "
                    f"limits [{lo:.3f}, {hi:.3f}]"
                )
    if qd_max is not None:
        vmax = np.broadcast_to(np.asarray(qd_max, dtype=float), (chunk.n_joints,))
        for j in range(chunk.n_joints):
            peak = np.abs(qd[:, j]).max()
            if peak > vmax[j]:
                problems.append(
                    f"joint {j}: peak speed {peak:.3f} rad/s exceeds limit {vmax[j]:.3f}"
                )
    return problems


def _check_or_raise(chunk: MotionChunk, joint_limits, qd_max) -> MotionChunk:
    if joint_limits is not None or qd_max is not None:
        problems = validate_chunk(chunk, joint_limits, qd_max)
        if problems:
            raise ValueError(f"chunk {chunk.name!r} violates limits:\n  " + "\n  ".join(problems))
    return chunk


# ------------------------------------------------------------- 점대점 min-jerk
def _min_jerk_s(tau: np.ndarray) -> np.ndarray:
    """정규화 min-jerk 프로파일 s(τ) = 10τ³ - 15τ⁴ + 6τ⁵ (양끝 속도·가속 0)."""
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def min_jerk(
    name: str, q_start, q_end, duration: float, *,
    n_knots: int = 17, tags: list[str] | None = None,
    joint_limits=None, qd_max=None,
) -> MotionChunk:
    """min-jerk 점대점 궤적. 양끝 위치 고정, 속도 0."""
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_end, dtype=float)
    tau = np.linspace(0.0, 1.0, n_knots)
    wp = q0[None, :] + _min_jerk_s(tau)[:, None] * (q1 - q0)[None, :]
    chunk = MotionChunk.from_waypoints(name, tau * duration, wp, tags=tags)
    return _check_or_raise(chunk, joint_limits, qd_max)


def via_points(
    name: str, q_start, vias: list, q_end, duration: float, *,
    tags: list[str] | None = None, n_knots_per_leg: int = 8,
    joint_limits=None, qd_max=None,
) -> MotionChunk:
    """경유점들을 지나는 점대점 궤적. 구간별 시간은 경로 길이에 비례 배분."""
    pts = [np.asarray(q_start, dtype=float)] + \
          [np.asarray(v, dtype=float) for v in vias] + \
          [np.asarray(q_end, dtype=float)]
    legs = np.array([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)])
    legs = np.maximum(legs, 1e-6)
    leg_T = duration * legs / legs.sum()

    # 각 구간을 min-jerk 시간매핑으로 채워 급출발을 피한다
    times = [0.0]
    wp = [pts[0]]
    t0 = 0.0
    for i, T in enumerate(leg_T):
        tau = np.linspace(0.0, 1.0, n_knots_per_leg + 1)[1:]
        s = _min_jerk_s(tau)
        for tk, sk in zip(tau, s):
            times.append(t0 + tk * T)
            wp.append(pts[i] + sk * (pts[i + 1] - pts[i]))
        t0 += T
    chunk = MotionChunk.from_waypoints(name, np.array(times), np.array(wp), tags=tags)
    return _check_or_raise(chunk, joint_limits, qd_max)


# ------------------------------------------------------------------ 변주 생성
def time_scaled(chunk: MotionChunk, factor: float, name: str | None = None) -> MotionChunk:
    """재생 속도 변경. factor=2.0 → 2배 느리게 (소요시간 2배). 기하는 동일."""
    if factor <= 0:
        raise ValueError("factor must be > 0")
    # q_new(t) = q_old(t / factor): dt^k 항 계수는 factor^-k 배
    scale = np.array([1.0, 1 / factor, 1 / factor**2, 1 / factor**3])
    coeffs = chunk.coeffs * scale[None, None, :]
    return MotionChunk(name or f"{chunk.name}_x{factor:g}", chunk.times * factor,
                       coeffs, tags=list(chunk.tags))


def amplitude_scaled(
    chunk: MotionChunk, factor: float, center=None, name: str | None = None,
) -> MotionChunk:
    """center 기준 진폭 스케일: q' = center + factor*(q - center). 기본 center=시작 자세."""
    c = chunk.q_start if center is None else np.asarray(center, dtype=float)
    coeffs = chunk.coeffs * factor
    coeffs[:, :, 0] += (1.0 - factor) * c[None, :]
    return MotionChunk(name or f"{chunk.name}_amp{factor:g}", chunk.times.copy(),
                       coeffs, tags=list(chunk.tags))


def with_detour(
    chunk: MotionChunk, t_via: float, offset, *,
    width: float | None = None, name: str | None = None,
    tags: list[str] | None = None, n_knots: int = 25,
    joint_limits=None, qd_max=None,
) -> MotionChunk:
    """기존 궤적에 우회 범프 추가 (장애물 회피 변주).

    t_via 시점 주변 width 구간 동안 offset만큼 벗어났다가 돌아온다.
    범프는 raised-cosine이라 양끝에서 위치/속도 영향이 0이다.
    """
    offset = np.asarray(offset, dtype=float)
    if not 0.0 < t_via < chunk.duration:
        raise ValueError("t_via must be inside (0, duration)")
    width = width if width is not None else chunk.duration * 0.5
    ts = np.linspace(0.0, chunk.duration, n_knots)
    q, _, _ = chunk.sample(ts)

    phase = np.clip((ts - (t_via - width / 2)) / width, 0.0, 1.0)
    bump = 0.5 * (1.0 - np.cos(2 * np.pi * phase))  # 0→1→0, 양끝 기울기 0
    wp = q + bump[:, None] * offset[None, :]

    out = MotionChunk.from_waypoints(
        name or f"{chunk.name}_detour", ts, wp,
        qd_start=chunk.qd_start, qd_end=chunk.qd_end,
        tags=tags if tags is not None else list(chunk.tags) + ["detour"],
    )
    return _check_or_raise(out, joint_limits, qd_max)


# ------------------------------------------------------------------ 후퇴 청크
def retreat(
    name: str, q_from, q_safe, duration: float, *,
    qd_from=None, tags: list[str] | None = None,
    joint_limits=None, qd_max=None,
) -> MotionChunk:
    """임의 상태(위치+속도)에서 안전 자세로 복귀. 진입 속도를 경계조건으로 흡수한다."""
    q0 = np.asarray(q_from, dtype=float)
    q1 = np.asarray(q_safe, dtype=float)
    v0 = np.zeros_like(q0) if qd_from is None else np.asarray(qd_from, dtype=float)

    tau = np.linspace(0.0, 1.0, 13)
    wp = q0[None, :] + _min_jerk_s(tau)[:, None] * (q1 - q0)[None, :]
    chunk = MotionChunk.from_waypoints(
        name, tau * duration, wp, qd_start=v0, qd_end=np.zeros_like(q0),
        tags=tags if tags is not None else ["retreat"],
    )
    return _check_or_raise(chunk, joint_limits, qd_max)
