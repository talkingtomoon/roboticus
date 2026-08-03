"""델타 모델 — 파라미터 복원, 강건성, 저장/로드, 추론 예산, 비교표."""

import time

import numpy as np
import pytest

from robot_core import MockRobotHAL
from robot_core.delta import (
    Collector, MLPDeltaModel, PhysicsDeltaModel, build_excitation, compare_models,
)

TAU_C, B, M = 0.4, 0.15, 0.05

torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False

needs_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")


@pytest.fixture(scope="module")
def mock_data():
    """백래시 켠 목에서 수집한 공유 데이터 (모듈당 1회 — 수집이 제일 느리다)."""
    hal = MockRobotHAL(n_joints=2, dt=1e-3, torque_limit=20.0,
                       coulomb_friction=TAU_C, viscous_friction=B, inertia=M,
                       backlash_width=0.02, enable_backlash=True)
    plan = build_excitation([0, 1], hal.joint_limits, budget_s=40.0, n_joints=2)
    return Collector(hal, kp=40.0, kd=2.0).collect(plan)


@pytest.fixture(scope="module")
def physics(mock_data):
    return PhysicsDeltaModel(2).fit(mock_data)


# ---------------------------------------------------------------- 물리 모델
def test_physics_recovers_planted_parameters(mock_data, physics):
    """리허설 핵심: 목에 심어둔 tau_c, b (그리고 M)를 10% 이내로 복원."""
    for j in (0, 1):
        p = physics.params[j]
        assert abs(p.coulomb - TAU_C) / TAU_C < 0.10, f"joint {j} tau_c={p.coulomb}"
        assert abs(p.viscous - B) / B < 0.10, f"joint {j} b={p.viscous}"
        assert abs(p.inertia - M) / M < 0.10, f"joint {j} M={p.inertia}"


def test_physics_robust_to_gear_slam_outliers(mock_data):
    """백래시 재체결 슬램(거대 qdd 이상치)이 있어도 피팅이 안 무너진다.

    (트리밍 없는 일반 lstsq는 이 데이터에서 M을 90% 이상 틀린다 — 회귀 테스트)
    """
    p = PhysicsDeltaModel(2).fit(mock_data).params[0]
    assert abs(p.inertia - M) / M < 0.10


def test_physics_fit_on_synthetic_arrays():
    """합성 데이터(관계식 정확)에서는 사실상 0% 오차."""
    from robot_core.delta.collector import CalibrationData
    rng = np.random.default_rng(3)
    dt, n = 1e-3, 6000
    t = np.arange(n) * dt
    qd = 1.2 * np.sin(2 * np.pi * 0.8 * t)[:, None]
    q = np.cumsum(qd, axis=0) * dt
    qdd = np.gradient(qd[:, 0], dt)[:, None]
    tau = M * qdd + B * qd + TAU_C * np.sign(qd)
    data = CalibrationData(t=t, q_des=q, qd_des=qd, q=q, qd=qd, tau=tau,
                           rate_hz=1000.0, joints=[0], kp=40.0, kd=2.0)
    p = PhysicsDeltaModel(1).fit(data, joints=[0]).params[0]
    assert abs(p.coulomb - TAU_C) < 0.02
    assert abs(p.viscous - B) < 0.02


def test_physics_save_load_roundtrip(tmp_path, physics):
    path = physics.save(tmp_path / "phys")
    loaded = PhysicsDeltaModel.load(path)
    q = np.array([0.1, -0.2]); qd = np.array([0.5, -0.3])
    physics.reset(); loaded.reset()
    a = physics.predict(q, qd, q, qd)
    b = loaded.predict(q, qd, q, qd)
    assert np.allclose(a, b, atol=1e-12)


def test_physics_predict_sign_and_scale(physics):
    """예측이 마찰 방향과 크기를 맞추는지 (양의 속도 → 양의 보정 토크)."""
    physics.reset()
    qd = np.array([0.5, -0.5])
    delta = physics.predict(np.zeros(2), qd, np.zeros(2), qd)
    assert delta[0] > 0 and delta[1] < 0
    assert abs(delta[0] - (TAU_C + B * 0.5)) < 0.1


def test_insufficient_data_raises():
    from robot_core.delta.collector import CalibrationData
    n = 50
    z = np.zeros((n, 1))
    data = CalibrationData(t=np.arange(n) * 1e-3, q_des=z, qd_des=z, q=z, qd=z,
                           tau=z, rate_hz=1000.0, joints=[0], kp=1, kd=1)
    with pytest.raises(ValueError, match="데이터 부족"):
        PhysicsDeltaModel(1).fit(data, joints=[0])


# --------------------------------------------------------------------- MLP
@pytest.fixture(scope="module")
def mlp(mock_data, physics):
    if not torch_available:
        pytest.skip("torch not installed")
    inertia = np.array([p.inertia for p in physics.params])
    return MLPDeltaModel(2).fit(mock_data, inertia=inertia, epochs=25, seed=0)


@needs_torch
def test_mlp_trains_and_beats_uncorrected(mock_data, physics, mlp):
    _, result = compare_models(mock_data, physics, mlp)
    for j in (0, 1):
        r = result["per_joint"][j]
        assert r["mlp"] < r["uncorrected"] * 0.9
        assert np.isfinite(mlp.val_rms[j])


@needs_torch
def test_mlp_save_load_roundtrip(tmp_path, mlp):
    path = mlp.save(tmp_path / "mlp")
    loaded = MLPDeltaModel.load(path)
    q = np.array([0.2, -0.1]); qd = np.array([0.7, -0.4])
    a = mlp.predict(q, qd, q + 0.01, qd)
    b = loaded.predict(q, qd, q + 0.01, qd)
    assert np.allclose(a, b, atol=1e-9)


@needs_torch
def test_mlp_packed_predict_matches_per_joint_forward(mock_data, mlp):
    """블록 행렬 일괄 추론 == 관절별 개별 forward (최적화 등가성 검증)."""
    rows = np.arange(100, 200)
    for j in (0, 1):
        per_joint = mlp._predict_rows(mock_data, j, rows)
        packed = np.array([
            mlp.predict(mock_data.q[k - 1], mock_data.qd[k - 1],
                        mock_data.q_des[k - 1], mock_data.qd_des[k - 1])[j]
            for k in rows])
        assert np.allclose(per_joint, packed, atol=1e-9)


def test_untrained_mlp_predicts_zero():
    m = MLPDeltaModel(2)
    d = m.predict(np.zeros(2), np.ones(2), np.zeros(2), np.ones(2))
    assert np.allclose(d, 0.0)


# ------------------------------------------------------------- 추론 예산
def _timing_us(model, n_joints, n_calls=1000):
    """관절당 추론 시간의 중앙값 [µs].

    p95가 아니라 중앙값을 쓴다 — 이 테스트는 '우리 코드'의 예산 준수를 재는
    벤치마크인데, p95는 병렬 테스트/다른 프로세스와의 CPU 경합(OS 선점)에
    지배돼 오탐이 난다. 실제 회귀(코드가 느려짐)는 중앙값에도 그대로 잡힌다.
    운용 중 tail 계측은 DeltaCorrectorNode.timing_report()가 담당한다.
    """
    q = np.zeros(n_joints); qd = np.full(n_joints, 0.3)
    model.reset()
    model.predict(q, qd, q, qd)  # 워밍업
    ts = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        model.predict(q, qd, q, qd)
        ts.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(ts)) / n_joints


def test_physics_inference_budget(physics):
    assert _timing_us(physics, 2) < 50.0


@needs_torch
def test_mlp_inference_budget(mlp):
    assert _timing_us(mlp, 2) < 50.0


# ---------------------------------------------------------------- 비교표
def test_compare_table_format(mock_data, physics):
    table, result = compare_models(mock_data, physics, None)
    assert "무보정" in table and "물리모델" in table
    assert result["overall"] == "physics"  # MLP 없으면 물리
    for j in (0, 1):
        r = result["per_joint"][j]
        assert r["physics"] < r["uncorrected"]  # 물리 보정이 무보정보다 낫다
