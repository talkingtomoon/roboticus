"""phact-401 적응 패치 — 프로파일 상수, AFC 가드, veto 한계 배선."""

import numpy as np
import pytest

from robot_core import MockRobotHAL
from robot_core.hal import PHACT_401, PROFILES, HardwareProfile
from robot_core.delta import (
    CalibrationData, Collector, DeltaCorrectorNode, PhysicsDeltaModel,
    build_excitation,
)
from robot_core.delta.models import _JointParams
from robot_core.integration import FullStack, StackConfig


# ---------------------------------------------------------------- 프로파일
def test_phact_401_profile_numbers():
    """주최측 공개 스펙 × 안전율 0.8 (출처: profiles.py 주석)."""
    p = PHACT_401
    assert p.n_joints == 6
    assert p.tau_limit == pytest.approx(7.2 * 0.8)        # 5.76 ≈ 5.8
    assert p.tau_limit_peak == pytest.approx(27.0 * 0.8)  # 21.6
    assert p.qd_limit == pytest.approx(15.7 * 0.8)        # 12.56
    assert PROFILES["phact-401"] is p


def test_profile_is_frozen():
    with pytest.raises(Exception):
        PHACT_401.tau_continuous = 100.0


def test_stack_config_from_profile_wires_three_tiers():
    """토크 한계 3단이 용도별로 올바른 곳에 배선돼야 한다.

    - HAL 포화        = tau_clamp (21.6, 순간 상한 — 버스트는 설계 의도)
    - 보정기 Δτ 기준   = tau_cont (5.76 — 지속 신호는 연속 예산 안)
    - 감지 임계        = tau_detect (10.8)
    - dream veto      = tau_veto (21.6)
    - 과부하 이동평균  = tau_cont (5.76)
    """
    cfg = StackConfig.from_profile(PHACT_401, n_joints=3)
    assert cfg.torque_limit == pytest.approx(5.76)
    assert cfg.torque_limit_clamp == pytest.approx(21.6)
    assert cfg.tau_detect == pytest.approx(10.8)
    assert cfg.torque_limit_peak == pytest.approx(21.6)

    stack = FullStack.build(cfg)
    assert np.allclose(stack.hal.torque_limits, 21.6)                     # 포화
    assert np.allclose(stack.corrector.torque_limits, 5.76)               # Δτ 기준
    assert np.allclose(stack.detector._tau_thr, 10.8)                     # 감지
    assert np.allclose(stack.chunk_node.scorer.dream.torque_limit, 21.6)  # veto
    assert np.allclose(stack.detector._cont_budget, 5.76)                 # 열 예산


def test_from_profile_defaults_to_profile_joint_count():
    cfg = StackConfig.from_profile(PHACT_401)
    assert cfg.n_joints == 6


# ------------------------------------------------------------- AFC 기록 경로
def test_mock_hal_reports_afc_off():
    assert MockRobotHAL(n_joints=2).afc_state == "off"


def test_collector_records_afc_state_and_roundtrips(tmp_path):
    hal = MockRobotHAL(n_joints=2, dt=1e-3)
    plan = build_excitation([0], hal.joint_limits, budget_s=4.0, n_joints=2)
    data = Collector(hal).collect(plan)
    assert data.afc_state == "off"
    assert "AFC=off" in data.quality_report()

    loaded = CalibrationData.load(data.save(tmp_path / "d"))
    assert loaded.afc_state == "off"


def test_collector_unknown_when_hal_has_no_afc(tmp_path):
    hal = MockRobotHAL(n_joints=2, dt=1e-3)
    del hal.afc_state   # AFC 개념이 없는 HAL 흉내
    plan = build_excitation([0], hal.joint_limits, budget_s=4.0, n_joints=2)
    assert Collector(hal).collect(plan).afc_state == "unknown"


def test_model_inherits_and_persists_afc_state(tmp_path):
    hal = MockRobotHAL(n_joints=2, dt=1e-3, coulomb_friction=0.3,
                       viscous_friction=0.1)
    plan = build_excitation([0, 1], hal.joint_limits, budget_s=12.0, n_joints=2)
    data = Collector(hal).collect(plan)
    model = PhysicsDeltaModel(2).fit(data)
    assert model.afc_state == "off"
    assert "AFC=off" in model.info()

    loaded = PhysicsDeltaModel.load(model.save(tmp_path / "m"))
    assert loaded.afc_state == "off"


# ------------------------------------------------------------- sanity 판정
def _model_with(coulomb: float, viscous: float = 0.1, afc: str = "off"):
    m = PhysicsDeltaModel(1)
    m.params[0] = _JointParams(inertia=0.05, viscous=viscous, coulomb=coulomb)
    m.afc_state = afc
    return m


def test_negative_friction_warns_when_afc_off():
    assert _model_with(coulomb=-0.15, afc="off").sanity_warnings()
    assert _model_with(coulomb=-0.15, afc="unknown").sanity_warnings()


def test_near_zero_friction_is_normal_when_afc_on():
    """AFC on이면 잔여 마찰이 0 근처(살짝 음수 포함)인 것이 정상."""
    assert _model_with(coulomb=-0.15, afc="on").sanity_warnings() == []
    assert _model_with(coulomb=0.02, afc="on").sanity_warnings() == []


def test_large_friction_with_afc_on_is_suspicious():
    warns = _model_with(coulomb=0.9, afc="on").sanity_warnings()
    assert any("AFC on인데" in w for w in warns)


def test_clean_fit_no_warnings():
    assert _model_with(coulomb=0.4, afc="off").sanity_warnings() == []


# --------------------------------------------------------------- 보정 노드 가드
def _corrector(model_afc: str, hw_afc: str):
    model = PhysicsDeltaModel(2)
    for j in range(2):
        model.params[j] = _JointParams(inertia=0.05, viscous=0.1, coulomb=0.3)
    model.afc_state = model_afc
    node = DeltaCorrectorNode(torque_limits=np.full(2, 5.76))
    node.set_model(model, current_afc_state=hw_afc)
    return node


def test_corrector_refuses_on_afc_mismatch():
    node = _corrector(model_afc="off", hw_afc="on")
    with pytest.warns(UserWarning, match="AFC"):
        ok = node.enable_correction()
    assert ok is False
    assert not node.correction_enabled
    assert "refused" in node.last_refusal


def test_corrector_allows_on_afc_match_or_unknown():
    node = _corrector("off", "off")
    assert node.enable_correction() is True
    assert node.last_refusal is None
    assert _corrector("on", "on").enable_correction() is True
    # 모델이 unknown이면 대조 불가 → 조용히 통과
    assert _corrector("unknown", "on").enable_correction() is True


def test_corrector_unverified_when_hardware_afc_unknown():
    """모델은 AFC 상태를 아는데 하드웨어가 unknown이면: 통과하되
    '검증 안 됨' 흔적(last_refusal='unverified: ...') + 경고."""
    node = _corrector("off", "unknown")
    with pytest.warns(UserWarning, match="unverified"):
        assert node.enable_correction() is True
    assert node.correction_enabled
    assert "unverified" in node.last_refusal


def test_corrector_recovers_after_afc_state_fixed():
    node = _corrector(model_afc="off", hw_afc="on")
    with pytest.warns(UserWarning):
        assert node.enable_correction() is False
    node.set_current_afc_state("off")   # AFC를 수집 당시 상태로 되돌림
    assert node.enable_correction() is True
    assert node.correction_enabled
