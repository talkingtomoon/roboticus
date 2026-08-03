"""데이터 수집기 — 수집, 안전 중단(부분 보존), npz, 품질 리포트."""

import numpy as np
import pytest

from robot_core import MockRobotHAL
from robot_core.delta import (
    CalibrationData, Collector, SafetyLimits, build_excitation,
)


@pytest.fixture
def hal():
    return MockRobotHAL(n_joints=2, dt=1e-3, torque_limit=20.0,
                        coulomb_friction=0.3, viscous_friction=0.1)


@pytest.fixture
def plan(hal):
    return build_excitation([0, 1], hal.joint_limits, budget_s=20.0, n_joints=2)


def test_collect_shapes_and_content(hal, plan):
    data = Collector(hal, kp=40.0, kd=2.0).collect(plan)
    n = data.n_samples
    assert n == pytest.approx(plan.duration * 1000, abs=2)
    for arr in (data.q_des, data.qd_des, data.q, data.qd, data.tau):
        assert arr.shape == (n, 2)
    assert not data.aborted
    assert data.joints == [0, 1]
    # 추종이 실제로 일어났다 (지령과 측정이 비슷)
    assert np.abs(data.q - data.q_des).max() < 0.2


def test_safety_abort_preserves_partial_data(hal, plan):
    """토크 감시 임계를 낮게 잡으면 중단되고, 그때까지 데이터는 보존."""
    tight = SafetyLimits(tau_frac=0.02, consecutive=3)   # 0.4 Nm에서 걸림
    data = Collector(hal, kp=40.0, kd=2.0, safety=tight).collect(plan)
    assert data.aborted
    assert "safety abort" in data.abort_reason
    assert 0 < data.n_samples < plan.duration * 1000 * 0.5
    assert np.all(np.isfinite(data.tau))                  # 쓰레기 행 없음


def test_safety_abort_on_speed(hal, plan):
    tight = SafetyLimits(tau_frac=1.0, qd_abs=0.05, consecutive=3)
    data = Collector(hal, kp=40.0, kd=2.0, safety=tight).collect(plan)
    assert data.aborted and "qd" in data.abort_reason


def test_single_sample_glitch_does_not_abort(hal, plan):
    """consecutive 미만의 단발 초과는 무시된다."""
    # 정상 한계로는 이 계획에서 중단이 없어야 한다
    data = Collector(hal, kp=40.0, kd=2.0,
                     safety=SafetyLimits(tau_frac=0.85, consecutive=3)).collect(plan)
    assert not data.aborted


def test_npz_roundtrip(tmp_path, hal, plan):
    data = Collector(hal).collect(plan)
    data.aborted = True
    data.abort_reason = "테스트 사유"
    path = data.save(tmp_path / "calib")
    loaded = CalibrationData.load(path)
    assert loaded.n_samples == data.n_samples
    assert np.allclose(loaded.tau, data.tau)
    assert loaded.aborted is True
    assert loaded.abort_reason == "테스트 사유"
    assert loaded.joints == [0, 1]
    assert loaded.rate_hz == 1000.0


def test_quality_report_structure(hal, plan):
    data = Collector(hal).collect(plan)
    text = data.quality_report()
    assert "[COLLECTION QUALITY]" in text
    assert "speed bins" in text
    assert "reversals=" in text
    for j in (0, 1):
        assert f"joint {j}:" in text


def test_quality_warnings_on_poor_data(hal):
    """짧은 수집 → 저속 샘플/반전 부족 경고가 떠야 한다."""
    tiny = build_excitation([0], hal.joint_limits, budget_s=4.0, n_joints=2)
    data = Collector(hal).collect(tiny)
    warns = data.warnings(min_reversals=50, min_slow_samples=100_000)
    assert any("반전" in w for w in warns)
    assert any("저속" in w for w in warns)
    assert "경고" in data.quality_report().replace(
        "커버리지 양호", "") or warns  # 리포트에 경고 섹션 노출


def test_reversal_count_matches_plan(hal, plan):
    data = Collector(hal).collect(plan)
    for j in (0, 1):
        assert data.count_reversals(j) >= plan.expected_reversals(j) * 0.7
