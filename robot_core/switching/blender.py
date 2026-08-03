"""블렌딩 스위처 — C1 연속(위치+속도) 전이 스플라인.

스위칭 순간의 불연속은 실물에서 기어를 갈아먹는다. 그래서:
- 전이는 양끝 (위치, 속도, 가속도) 경계조건을 만족하는 5차 다항식
  (C1은 타협 불가 요구사항, 가속도까지 맞추므로 실제로는 C2)
- 전이 시간은 상태 거리에 비례해 자동 결정 (급할수록 짧게, 최소값 보장)
- 블렌더는 무상태(stateless) — 전이 중 재스위칭이 와도 "현재 참조 상태에서
  다시 blend()"만 하면 되므로 재귀적으로 안전하다
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_core.chunks.format import MotionChunk


@dataclass
class BlendConfig:
    min_duration_s: float = 0.15   # 아무리 급해도 이보다 짧은 전이는 안 만든다
    max_duration_s: float = 1.0
    k_pos: float = 1.2             # 전이시간 ≈ k_pos*||Δq|| + k_vel*||Δqd||
    k_vel: float = 0.25


def quintic_coeffs(p0, v0, a0, pf, vf, af, T: float) -> np.ndarray:
    """양끝 (위치,속도,가속) 경계조건을 만족하는 5차 다항식 계수, shape=(n, 6)."""
    p0, v0, a0 = (np.asarray(x, dtype=float) for x in (p0, v0, a0))
    pf, vf, af = (np.asarray(x, dtype=float) for x in (pf, vf, af))
    h = pf - p0
    c = np.empty((len(p0), 6))
    c[:, 0] = p0
    c[:, 1] = v0
    c[:, 2] = a0 / 2.0
    c[:, 3] = (20 * h - (8 * vf + 12 * v0) * T - (3 * a0 - af) * T**2) / (2 * T**3)
    c[:, 4] = (-30 * h + (14 * vf + 16 * v0) * T + (3 * a0 - 2 * af) * T**2) / (2 * T**4)
    c[:, 5] = (12 * h - 6 * (vf + v0) * T + (af - a0) * T**2) / (2 * T**5)
    return c


def _eval_quintic(c: np.ndarray, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
    q = (c[:, 0] + c[:, 1] * t_arr + c[:, 2] * t_arr**2 + c[:, 3] * t_arr**3
         + c[:, 4] * t_arr**4 + c[:, 5] * t_arr**5)
    qd = (c[:, 1] + 2 * c[:, 2] * t_arr + 3 * c[:, 3] * t_arr**2
          + 4 * c[:, 4] * t_arr**3 + 5 * c[:, 5] * t_arr**4)
    qdd = (2 * c[:, 2] + 6 * c[:, 3] * t_arr + 12 * c[:, 4] * t_arr**2
           + 20 * c[:, 5] * t_arr**3)
    if np.isscalar(t) or np.ndim(t) == 0:
        return q[0], qd[0], qdd[0]
    return q, qd, qdd


class TransitionPlan:
    """전이 스플라인(0..blend_s) 뒤에 청크가 이어지는 합성 궤적.

    sample(t):
      t < blend_s          : 5차 전이
      blend_s <= t < end   : chunk.sample(t - blend_s)
      t >= end             : 종점 홀드 (qd=qdd=0)
    """

    def __init__(self, coeffs: np.ndarray, blend_s: float, chunk: MotionChunk) -> None:
        self._c = coeffs
        self.blend_s = float(blend_s)
        self.chunk = chunk
        self.duration = self.blend_s + chunk.duration

    @property
    def name(self) -> str:
        return f"blend->{self.chunk.name}"

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if t >= self.duration:
            q = self.chunk.q_end
            return q.copy(), np.zeros_like(q), np.zeros_like(q)
        if t < self.blend_s:
            return _eval_quintic(self._c, max(0.0, t))
        return self.chunk.sample(t - self.blend_s)


class ChunkPlan:
    """청크 하나를 그대로 따르는 계획 (전이 없음). 끝나면 종점 홀드."""

    def __init__(self, chunk: MotionChunk) -> None:
        self.chunk = chunk
        self.blend_s = 0.0
        self.duration = chunk.duration

    @property
    def name(self) -> str:
        return self.chunk.name

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if t >= self.duration:
            q = self.chunk.q_end
            return q.copy(), np.zeros_like(q), np.zeros_like(q)
        return self.chunk.sample(max(0.0, t))


class Blender:
    def __init__(self, config: BlendConfig | None = None) -> None:
        self.cfg = config or BlendConfig()

    def blend_duration(self, q, qd, chunk: MotionChunk) -> float:
        dq = np.linalg.norm(chunk.q_start - np.asarray(q, dtype=float))
        dv = np.linalg.norm(chunk.qd_start - np.asarray(qd, dtype=float))
        T = self.cfg.k_pos * dq + self.cfg.k_vel * dv
        return float(np.clip(T, self.cfg.min_duration_s, self.cfg.max_duration_s))

    def blend(self, q, qd, chunk: MotionChunk, qdd=None) -> TransitionPlan:
        """현재 참조 상태 (q, qd[, qdd])에서 chunk 시작점으로 C1 연속 전이.

        전이 중 재스위칭: 진행 중 플랜을 현재 시점에서 sample한 (q,qd,qdd)로
        다시 blend()를 부르면 된다. 이 함수는 아무 상태도 저장하지 않는다.
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        a0 = np.zeros_like(q) if qdd is None else np.asarray(qdd, dtype=float)

        T = self.blend_duration(q, qd, chunk)
        q1, qd1, qdd1 = chunk.sample(0.0)
        coeffs = quintic_coeffs(q, qd, a0, q1, qd1, qdd1, T)
        return TransitionPlan(coeffs, T, chunk)
