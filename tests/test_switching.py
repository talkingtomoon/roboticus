"""채점기(예산·저항·veto) + 블렌더(연속성·재스위칭) + 외란 추정."""

import time

import numpy as np
import pytest

from robot_core import MockRobotHAL, RingLogger
from robot_core.chunks import generator as G
from robot_core.switching import (
    BlendConfig, Blender, ChunkScorer, DreamModel, ScorerConfig, ScoreWeights,
    estimate_disturbance,
)

from .conftest import pd_command, run_steps

N = 3


@pytest.fixture
def dream():
    return DreamModel(inertia=0.05, viscous=0.02, coulomb=0.1,
                      torque_limit=20.0, kp=40.0, kd=2.0)


def rand_chunks(m, rng=None):
    rng = rng or np.random.default_rng(0)
    return [G.min_jerk(f"c{i}", rng.uniform(-0.5, 0.5, N), rng.uniform(-0.5, 0.5, N),
                       rng.uniform(0.5, 2.0)) for i in range(m)]


# ------------------------------------------------------------------ 채점기
def test_scorer_meets_5ms_budget_with_100_candidates(dream):
    scorer = ChunkScorer(rand_chunks(100), dream=dream)
    q, qd = np.zeros(N), np.zeros(N)
    d, goal = np.array([0.0, 6.0, 0.0]), np.array([0.8, 0.0, 0.4])

    scorer.score(q, qd, d, goal)  # 워밍업 (첫 호출 캐시/할당 제외)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        scorer.score(q, qd, d, goal)
        times.append(time.perf_counter() - t0)
    median_ms = sorted(times)[len(times) // 2] * 1e3
    assert median_ms < 5.0, f"채점 {median_ms:.2f} ms > 5 ms 예산"


def test_resistance_cost_filters_chunks_fighting_the_disturbance(dream):
    """외란이 관절 1을 +방향으로 미는 상황: -방향(맞서는) 청크가 걸러져야 한다."""
    start = np.zeros(N)
    with_force = G.min_jerk("against", start, [0.0, -0.6, 0.0], 1.0)   # 힘에 맞섬
    yielding = G.min_jerk("comply", start, [0.0, +0.6, 0.0], 1.0)      # 힘에 순응
    sideways = G.min_jerk("avoid", start, [0.6, 0.0, 0.0], 1.0)        # 힘을 비켜감

    scorer = ChunkScorer([with_force, yielding, sideways], dream=None,
                         config=ScorerConfig(weights=ScoreWeights(progress=0.0)))
    d = np.array([0.0, 6.0, 0.0])
    report = scorer.score(start, np.zeros(N), d, goal=np.zeros(N))

    by_name = {e.name: e for e in report.entries}
    assert by_name["against"].resistance == pytest.approx(1.0, abs=1e-6)
    assert by_name["comply"].resistance == pytest.approx(0.0, abs=1e-6)
    assert by_name["avoid"].resistance == pytest.approx(0.5, abs=1e-6)  # 중립
    # 서열: 순응 < 비켜감 < 정면 대항
    assert (by_name["comply"].resistance < by_name["avoid"].resistance
            < by_name["against"].resistance)
    assert report.best.name == "comply"


def test_zero_disturbance_means_zero_resistance(dream):
    scorer = ChunkScorer(rand_chunks(10), dream=dream)
    report = scorer.score(np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N))
    assert all(e.resistance == 0.0 for e in report.entries)


def test_connection_cost_prefers_nearby_entry():
    near = G.min_jerk("near", [0.1, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0)
    far = G.min_jerk("far", [2.0, 2.0, 2.0], [0.5, 0.0, 0.0], 1.0)
    scorer = ChunkScorer([near, far], dream=None,
                         config=ScorerConfig(weights=ScoreWeights(progress=0.0)))
    report = scorer.score(np.array([0.1, 0.0, 0.0]), np.zeros(N),
                          np.zeros(N), np.zeros(N))
    assert report.best.name == "near"


def test_progress_cost_penalizes_pure_retreat():
    goal = np.array([0.8, 0.0, 0.0])
    onward = G.min_jerk("onward", [0.4, 0.0, 0.0], goal, 1.0)
    back = G.retreat("back", [0.4, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0)
    scorer = ChunkScorer([onward, back], dream=None)
    report = scorer.score(np.array([0.4, 0.0, 0.0]), np.zeros(N), np.zeros(N), goal)
    by_name = {e.name: e for e in report.entries}
    assert by_name["back"].progress > by_name["onward"].progress
    assert report.best.name == "onward"


def test_dream_veto_rejects_torque_infeasible_candidates():
    """멀리 떨어진 진입점 → kp*오차가 토크 한계를 넘을 것으로 예측 → 탈락."""
    feasible = G.min_jerk("feasible", [0.05, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0)
    infeasible = G.min_jerk("infeasible", [3.0, 0.0, 0.0], [3.5, 0.0, 0.0], 1.0)
    dream = DreamModel(inertia=0.05, viscous=0.02, coulomb=0.1,
                       torque_limit=20.0, kp=40.0, kd=2.0)  # 40*2.95 >> 20
    scorer = ChunkScorer([feasible, infeasible], dream=dream)
    report = scorer.score(np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N))

    by_name = {e.name: e for e in report.entries}
    assert by_name["infeasible"].vetoed
    assert by_name["infeasible"].peak_tau_frac > 1.0
    assert not by_name["feasible"].vetoed
    assert report.best.name == "feasible"


def test_all_vetoed_returns_no_best():
    far = [G.min_jerk(f"f{i}", [5.0 + i, 0, 0], [6.0 + i, 0, 0], 1.0) for i in range(3)]
    dream = DreamModel(inertia=0.05, viscous=0.0, coulomb=0.0,
                       torque_limit=5.0, kp=40.0, kd=2.0)
    scorer = ChunkScorer(far, dream=dream)
    report = scorer.score(np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N))
    assert report.best is None
    assert all(e.vetoed for e in report.entries)


def test_scorer_input_validation(dream):
    with pytest.raises(ValueError):
        ChunkScorer([], dream=dream)
    with pytest.raises(ValueError):
        ChunkScorer([G.min_jerk("a", [0.0], [1.0], 1.0),
                     G.min_jerk("b", [0, 0], [1, 1], 1.0)])


# --------------------------------------------------------------- 외란 추정
@pytest.mark.parametrize("steps_after,baseline_s", [(40, 0.4), (500, 2.0)])
def test_estimate_disturbance_direction_from_mock(steps_after, baseline_s):
    """충격 직후(토크 신호)든 한참 뒤(처짐 신호)든 방향이 나와야 한다.

    PD가 외란을 상쇄하면 tau_measured는 0으로 돌아가므로 토크 변화만으로는
    정상상태 외란을 못 잡는다 — 처짐 항 + 중앙값 기준선이 이를 커버하는지가 핵심.
    (늦게 부를수록 baseline_s를 길게 잡아 오염 비율을 절반 미만으로 유지해야 한다)
    """
    hal = MockRobotHAL(n_joints=N, dt=1e-3, torque_limit=20.0)
    logger = RingLogger.for_hal(hal, window_sec=5.0)
    cmd = pd_command(hal, np.zeros(N), kp=40.0)
    run_steps(hal, cmd, 800, logger=logger)          # 조용한 기준 구간
    hal.inject_disturbance(1, +6.0, duration=5.0)
    run_steps(hal, cmd, steps_after, logger=logger)  # 외란 걸린 최근 구간

    d = estimate_disturbance(logger.to_arrays(), window_s=0.05,
                             baseline_s=baseline_s, stiffness=40.0)
    assert d[1] > 2.5                                 # 관절 1, + 부호가 지배적
    assert d[1] > 3 * max(abs(d[0]), abs(d[2]))       # 다른 관절 대비 지배적


def test_estimate_disturbance_empty_arrays_safe():
    d = estimate_disturbance({"t": np.zeros(0), "tau": np.zeros((0, N))})
    assert d.shape == (0,) or np.allclose(d, 0.0)


# ------------------------------------------------------------------ 블렌더
def test_blend_boundaries_are_continuous_below_1e6():
    """블렌딩 경계(시작/청크 연결점)의 위치·속도 불연속 < 1e-6 (요구사항)."""
    blender = Blender()
    chunk = G.min_jerk("alt", [0.3, 0.2, 0.1], [0.0, 0.5, 0.0], 1.0)
    q0 = np.array([0.1, 0.0, 0.05])
    qd0 = np.array([0.4, -0.1, 0.0])
    plan = blender.blend(q0, qd0, chunk)

    # 시작점: 현재 상태와 일치
    q, qd, _ = plan.sample(0.0)
    assert np.abs(q - q0).max() < 1e-9
    assert np.abs(qd - qd0).max() < 1e-9

    # 전이→청크 경계
    eps = 1e-7
    qa, qda, _ = plan.sample(plan.blend_s - eps)
    qb, qdb, _ = plan.sample(plan.blend_s + eps)
    assert np.abs(qa - qb).max() < 1e-6
    assert np.abs(qda - qdb).max() < 1e-6


def test_blend_duration_scales_with_distance_and_respects_min():
    cfg = BlendConfig(min_duration_s=0.15, max_duration_s=1.0)
    blender = Blender(cfg)
    near = G.min_jerk("near", [0.01, 0, 0], [0.5, 0, 0], 1.0)
    far = G.min_jerk("far", [1.5, 1.5, 1.5], [2.0, 2.0, 2.0], 1.0)
    q0, qd0 = np.zeros(3), np.zeros(3)
    t_near = blender.blend_duration(q0, qd0, near)
    t_far = blender.blend_duration(q0, qd0, far)
    assert t_near == cfg.min_duration_s          # 최소값 보장
    assert cfg.min_duration_s < t_far <= cfg.max_duration_s
    assert t_far > t_near


def test_reswitch_mid_transition_is_recursively_safe():
    """전이 도중 다시 블렌딩해도 새 경계에서 연속성이 유지돼야 한다."""
    blender = Blender()
    first = G.min_jerk("first", [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0)
    second = G.min_jerk("second", [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], 1.0)

    plan1 = blender.blend(np.zeros(3), np.array([0.5, 0.0, 0.0]), first)
    t_mid = plan1.blend_s * 0.4                    # 전이 40% 지점에서 재스위칭
    q_mid, qd_mid, qdd_mid = plan1.sample(t_mid)

    plan2 = blender.blend(q_mid, qd_mid, second, qdd=qdd_mid)
    q, qd, qdd = plan2.sample(0.0)
    assert np.abs(q - q_mid).max() < 1e-9          # 위치 연속
    assert np.abs(qd - qd_mid).max() < 1e-9        # 속도 연속
    assert np.abs(qdd - qdd_mid).max() < 1e-9      # 가속도까지 연속

    # 새 계획도 목표 청크에 정상 도달
    q_end, qd_end, _ = plan2.sample(plan2.duration + 1.0)
    assert np.allclose(q_end, second.q_end, atol=1e-9)
    assert np.allclose(qd_end, 0.0)


def test_plan_holds_end_pose_after_completion():
    blender = Blender()
    chunk = G.min_jerk("c", [0.0, 0, 0], [0.5, 0, 0], 1.0)
    plan = blender.blend(np.zeros(3), np.zeros(3), chunk)
    q, qd, qdd = plan.sample(plan.duration + 10.0)
    assert np.allclose(q, chunk.q_end)
    assert np.allclose(qd, 0.0) and np.allclose(qdd, 0.0)
