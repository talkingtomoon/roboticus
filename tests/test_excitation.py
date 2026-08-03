"""여기 궤적 생성기 — 한계 준수(절대), 예산 배분, 믹스 구성."""

import numpy as np
import pytest

from robot_core.delta import ExcitationConfig, build_excitation

JL = np.tile([-np.pi / 2, np.pi / 2], (3, 1))


def test_all_three_segment_kinds_present():
    plan = build_excitation([0], JL, budget_s=60.0, n_joints=3)
    kinds = {w.segment.kind for w in plan.windows}
    assert kinds == {"sine_sweep", "reversal", "quasi_static_ramp"}


def test_limits_never_violated_analytically():
    """해석적 피크 + 조밀 샘플링 이중 검사. 여유율(margin<1)로도 통과해야 한다."""
    cfg = ExcitationConfig(safety_frac=0.5, qd_limit=4.0)
    plan = build_excitation([0, 1, 2], JL, budget_s=90.0, n_joints=3, config=cfg)

    # 해석적: safety_frac=0.5이므로 margin 0.55에서도 통과해야 한다 (여유 포함)
    assert plan.validate(JL, cfg.qd_limit, margin=1.0) == []
    assert plan.validate(JL, cfg.qd_limit, margin=0.55) == []

    # 샘플링: 실제 sample() 출력도 한계의 55% 안
    ts = np.arange(0.0, plan.duration, 0.01)
    for t in ts:
        q, qd = plan.sample(t)
        assert np.all(q >= JL[:, 0] * 0.55 - 1e-9)
        assert np.all(q <= JL[:, 1] * 0.55 + 1e-9)
        assert np.all(np.abs(qd) <= cfg.qd_limit * 0.5 + 1e-9)


def test_sequential_windows_do_not_overlap_across_joints():
    plan = build_excitation([0, 1], JL, budget_s=40.0, n_joints=3, mode="sequential")
    for a in plan.windows:
        for b in plan.windows:
            if a is b or a.joint == b.joint:
                continue
            assert a.t_end <= b.t_start or b.t_end <= a.t_start, \
                "순차 모드에서 다른 관절 창이 겹쳤다"


def test_parallel_mode_overlaps_and_is_shorter():
    """병렬은 각 관절이 전체 예산을 쓰되 동시에 돈다 — 총 시간은 순차보다 짧고,
    관절당 데이터는 3배 (예산의 의미가 모드에 따라 다름을 문서화하는 테스트)."""
    seq = build_excitation([0, 1, 2], JL, budget_s=60.0, n_joints=3, mode="sequential")
    par = build_excitation([0, 1, 2], JL, budget_s=60.0, n_joints=3, mode="parallel")
    assert par.duration < seq.duration * 0.8
    # 병렬: 서로 다른 관절이 동시에 활성
    t_mid = par.windows[0].t_start + 0.1
    active = {w.joint for w in par.windows if w.t_start <= t_mid < w.t_end}
    assert len(active) >= 2


def test_budget_roughly_respected():
    for budget in (60.0, 240.0):
        plan = build_excitation([0], JL, budget_s=budget, n_joints=3)
        assert budget * 0.6 <= plan.duration <= budget * 1.4


def test_joint_selection_only_moves_selected():
    plan = build_excitation([1], JL, budget_s=30.0, n_joints=3)
    ts = np.arange(0.0, plan.duration, 0.05)
    for t in ts:
        q, qd = plan.sample(t)
        assert q[0] == 0.0 and q[2] == 0.0
        assert qd[0] == 0.0 and qd[2] == 0.0


def test_reversal_segments_produce_many_reversals():
    plan = build_excitation([0], JL, budget_s=120.0, n_joints=3)
    assert plan.expected_reversals(0) >= 20


def test_rest_at_limit_rejected():
    rest = np.array([JL[0, 1], 0.0, 0.0])  # 관절 0이 상한에 붙어 있음
    with pytest.raises(ValueError, match="no safe amplitude"):
        build_excitation([0], JL, budget_s=30.0, rest=rest, n_joints=3)


def test_invalid_args():
    with pytest.raises(ValueError):
        build_excitation([], JL, budget_s=30.0, n_joints=3)
    with pytest.raises(ValueError):
        build_excitation([0], JL, budget_s=30.0, n_joints=3, mode="both")
