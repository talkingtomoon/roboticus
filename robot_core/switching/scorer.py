"""실시간 청크 채점기 — 전 과정 numpy 벡터화. 후보 100개를 5ms 안에.

채점 항목 (가중합, 낮을수록 좋음):
  (a) connection : 현재 상태에서 청크 진입점까지의 위치/속도 거리
  (b) resistance : 청크의 초기 이동 방향이 추정 외란과 얼마나 '맞서는가'.
                   (1 - cos)/2 — 순응 0, 무관(비켜감) 0.5, 정면 대항 1.
                   순응이 중립보다 항상 싸다: 밀리면 밀리는 쪽 우회를 고른다
  (c) progress   : 청크 종점이 원래 목표에서 얼마나 먼가 (후퇴만 반복하지 않게)

"Dream" 요소: 각 후보의 첫 T_horizon 구간을 관절별 1자유도 모델로 순전파해
예상 토크 지령을 계산하고, 토크 한계 초과가 예측되는 후보는 탈락(veto)시킨다.
동역학 파라미터는 DreamModel로 주입 — 목이든 실물 추정치든 갈아끼울 수 있다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from robot_core.chunks.format import MotionChunk


# ---------------------------------------------------------- 외란 방향 추정
def estimate_disturbance(arrays: dict, window_s: float = 0.08,
                         baseline_s: float = 0.5,
                         stiffness: float | np.ndarray = 40.0) -> np.ndarray:
    """관절별 부호 있는 외란 토크 추정, shape=(n,).

    FailureDetector 이벤트를 스위칭 트리거로 쓸 때 필요한 '어느 관절에 어느
    부호로 힘이 걸렸나'만 뽑는 얇은 함수다. 두 신호를 합친다:

    1. 토크 변화:  mean(tau 최근) - mean(tau 기준구간).
       충격 직후 과도 구간에서만 살아있다 — PD가 외란을 상쇄하고 나면
       tau_measured(=구동+외란)는 0 근처로 돌아간다.
    2. 처짐 변화:  stiffness * [mean(q - q_des 최근) - median(q - q_des 기준)].
       외란은 관절을 힘의 방향으로 밀어 지령에서 처지게 하고, 이건
       정상상태에서도 지속된다. stiffness에 실제 kp를 주면 크기도 대략 맞는다
       (정상상태 처짐 = 외란/kp). 방향 추정이 목적이라 정확도는 부차적.

    기준구간은 평균이 아니라 **중앙값**을 쓴다 — 감지가 늦어 기준구간의 일부가
    이미 외란에 오염됐어도, 오염이 절반 미만이면 기준값이 무너지지 않는다.
    (그래도 감지 직후에 부르는 것이 정석이다.)
    """
    t, tau = arrays["t"], arrays["tau"]
    n = tau.shape[1] if tau.ndim == 2 else 0
    if len(t) < 4:
        return np.zeros(n)
    t_end = t[-1]
    recent = t >= t_end - window_s
    base = (t < t_end - window_s) & (t >= t_end - window_s - baseline_s)
    if not base.any():
        base = ~recent
    if not base.any():
        return np.zeros(n)

    est = tau[recent].mean(axis=0) - np.median(tau[base], axis=0)

    q, q_des = arrays.get("q"), arrays.get("q_des")
    if q is not None and q_des is not None and np.all(np.isfinite(q_des)):
        defl = q - q_des
        est = est + np.asarray(stiffness, dtype=float) * (
            defl[recent].mean(axis=0) - np.median(defl[base], axis=0))
    return est


# ------------------------------------------------------------------ 설정
@dataclass
class ScoreWeights:
    connection: float = 1.0     # [1/rad] 위치·속도 거리
    connection_vel: float = 0.3 # connection 안에서 속도 거리의 상대 비중
    resistance: float = 2.0    # [무차원 0..1] 외란과 맞서는 정도
    progress: float = 0.6      # [1/rad] 목표까지 남는 거리


@dataclass
class DreamModel:
    """Dream 롤아웃용 관절별 1자유도 모델 + 추종 제어기 게인. 전부 (n,) 또는 스칼라.

    MockRobotHAL에서 뽑아 쓰되(from_mock_hal), 실물에서는 추정 파라미터로 교체.
    """

    inertia: np.ndarray
    viscous: np.ndarray
    coulomb: np.ndarray
    torque_limit: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self):
        n = np.broadcast_shapes(*(np.shape(np.atleast_1d(v)) for v in
                                  (self.inertia, self.torque_limit)))[0]
        for name in ("inertia", "viscous", "coulomb", "torque_limit", "kp", "kd"):
            setattr(self, name, np.broadcast_to(
                np.asarray(getattr(self, name), dtype=float), (n,)).copy())

    @classmethod
    def from_mock_hal(cls, hal, kp, kd) -> "DreamModel":
        return cls(
            inertia=hal.inertia.copy(),
            viscous=hal.viscous_friction.copy() if hal.enable_viscous_friction else 0.0,
            coulomb=hal.coulomb_friction.copy() if hal.enable_coulomb_friction else 0.0,
            torque_limit=hal.torque_limits,
            kp=kp, kd=kd,
        )


@dataclass
class ScorerConfig:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    dream_horizon_s: float = 0.15
    dream_dt: float = 5e-3
    torque_veto_frac: float = 0.95   # 예측 |tau지령| > limit*frac → 탈락
    entry_dir_window_s: float = 0.25 # 초기 이동 방향을 잴 구간 (duration의 상한으로 클램프)


# ------------------------------------------------------------------ 결과
@dataclass
class CandidateScore:
    name: str
    connection: float
    resistance: float
    progress: float
    total: float
    vetoed: bool
    peak_tau_frac: float   # 예측 피크 토크 / 한계 (관절 최악값)

    def line(self) -> str:
        flag = " VETO(torque)" if self.vetoed else ""
        return (f"{self.name:24s} total={self.total:7.3f}  "
                f"conn={self.connection:6.3f} resist={self.resistance:5.3f} "
                f"prog={self.progress:6.3f}  peak_tau={100 * self.peak_tau_frac:5.1f}%{flag}")


@dataclass
class ScoreReport:
    entries: list[CandidateScore]
    best_index: int | None      # None = 전 후보 탈락
    elapsed_ms: float

    @property
    def best(self) -> CandidateScore | None:
        return None if self.best_index is None else self.entries[self.best_index]

    def table(self, top: int | None = None) -> str:
        order = sorted(range(len(self.entries)),
                       key=lambda i: (self.entries[i].vetoed, self.entries[i].total))
        lines = [f"[SCORE] {len(self.entries)} candidates in {self.elapsed_ms:.2f} ms"]
        for rank, i in enumerate(order[: top or len(order)], 1):
            marker = " <-- chosen" if i == self.best_index else ""
            lines.append(f"  {rank:2d}. {self.entries[i].line()}{marker}")
        return "\n".join(lines)


# ------------------------------------------------------------------ 채점기
class ChunkScorer:
    """후보 목록은 생성 시 고정 — 정적 특징(진입점·방향·Dream 참조 궤적)을
    전부 미리 계산해 두고, score() 호출에서는 벡터 연산만 한다."""

    def __init__(
        self,
        chunks: list[MotionChunk],
        dream: DreamModel | None = None,
        config: ScorerConfig | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("need at least one candidate chunk")
        self.chunks = list(chunks)
        self.dream = dream
        self.cfg = config or ScorerConfig()

        n = chunks[0].n_joints
        if any(c.n_joints != n for c in chunks):
            raise ValueError("all chunks must have the same n_joints")
        self.n_joints = n
        M = len(chunks)

        # --- 정적 특징 미리 계산 ---
        self._entry_q = np.stack([c.q_start for c in chunks])       # (M, n)
        self._entry_qd = np.stack([c.qd_start for c in chunks])     # (M, n)
        self._end_q = np.stack([c.q_end for c in chunks])           # (M, n)

        # 초기 이동 방향: 처음 entry_dir_window 동안의 변위 (정지 출발도 방향이 나온다)
        disp = np.stack([
            c.sample(min(self.cfg.entry_dir_window_s, c.duration))[0] - c.q_start
            for c in chunks
        ])
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        self._entry_dir = np.where(norms > 1e-9, disp / np.maximum(norms, 1e-9), 0.0)

        # Dream 참조 궤적: (M, T, n) — 후보가 horizon보다 짧으면 끝점 홀드(sample 클램프)
        steps = max(2, int(round(self.cfg.dream_horizon_s / self.cfg.dream_dt)))
        self._dream_ts = np.arange(1, steps + 1) * self.cfg.dream_dt
        q_ref = np.empty((M, steps, n))
        qd_ref = np.empty((M, steps, n))
        for i, c in enumerate(chunks):
            q, qd, _ = c.sample(self._dream_ts)
            q_ref[i], qd_ref[i] = q, qd
        self._dream_q_ref = q_ref
        self._dream_qd_ref = qd_ref

    # ---------------------------------------------------------------- 채점
    def score(self, q, qd, disturbance, goal) -> ScoreReport:
        """현재 상태 + 추정 외란 + 원래 목표 → 전체 후보 채점 리포트."""
        t0 = time.perf_counter()
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        d = np.asarray(disturbance, dtype=float)
        goal = np.asarray(goal, dtype=float)
        w = self.cfg.weights

        # (a) 연결 비용
        conn = (np.linalg.norm(self._entry_q - q, axis=1)
                + w.connection_vel * np.linalg.norm(self._entry_qd - qd, axis=1))

        # (b) 저항 비용: (1 - cos(초기방향, 외란방향)) / 2 ∈ [0, 1]
        #     순응=0 < 중립(외란과 직교)=0.5 < 정면 대항=1
        d_norm = np.linalg.norm(d)
        if d_norm > 1e-9:
            cos = self._entry_dir @ (d / d_norm)
            resist = (1.0 - cos) / 2.0
        else:
            resist = np.zeros(len(self.chunks))

        # (c) 진행 비용
        prog = np.linalg.norm(self._end_q - goal, axis=1)

        total = w.connection * conn + w.resistance * resist + w.progress * prog

        # Dream 롤아웃 → 토크 한계 veto
        if self.dream is not None:
            peak_frac = self._dream_rollout(q, qd, d)
            vetoed = peak_frac > self.cfg.torque_veto_frac
        else:
            peak_frac = np.zeros(len(self.chunks))
            vetoed = np.zeros(len(self.chunks), dtype=bool)

        entries = [
            CandidateScore(
                name=c.name, connection=float(conn[i]), resistance=float(resist[i]),
                progress=float(prog[i]), total=float(total[i]),
                vetoed=bool(vetoed[i]), peak_tau_frac=float(peak_frac[i]),
            )
            for i, c in enumerate(self.chunks)
        ]
        alive = np.flatnonzero(~vetoed)
        best = int(alive[np.argmin(total[alive])]) if alive.size else None
        return ScoreReport(entries=entries, best_index=best,
                           elapsed_ms=(time.perf_counter() - t0) * 1e3)

    def _dream_rollout(self, q0: np.ndarray, qd0: np.ndarray,
                       disturbance: np.ndarray) -> np.ndarray:
        """전 후보를 현재 상태에서 동시에 순전파. 외란은 horizon 내내 지속 가정.

        반환: (M,) 후보별 max_j (예측 |tau지령| / torque_limit).
        """
        dm = self.dream
        M = len(self.chunks)
        dt = self.cfg.dream_dt
        q = np.tile(q0, (M, 1))     # (M, n)
        qd = np.tile(qd0, (M, 1))
        peak = np.zeros((M, self.n_joints))

        for k in range(self._dream_q_ref.shape[1]):
            tau_cmd = (dm.kp * (self._dream_q_ref[:, k] - q)
                       + dm.kd * (self._dream_qd_ref[:, k] - qd))
            np.maximum(peak, np.abs(tau_cmd), out=peak)
            tau = np.clip(tau_cmd, -dm.torque_limit, dm.torque_limit)
            qdd = (tau + disturbance - dm.viscous * qd
                   - dm.coulomb * np.sign(qd)) / dm.inertia
            qd = qd + qdd * dt
            q = q + qd * dt

        return (peak / dm.torque_limit).max(axis=1)
