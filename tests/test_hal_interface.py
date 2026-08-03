"""HAL 계약 테스트: 목이든 실물이든 이 규약은 깨지면 안 된다."""

import numpy as np
import pytest

from robot_core import JointCommand, JointState, MockRobotHAL, RealRobotHAL, RobotHAL


def test_mock_is_a_robot_hal(hal):
    assert isinstance(hal, RobotHAL)


def test_state_shapes_and_types(hal):
    s = hal.read_state()
    assert isinstance(s, JointState)
    for arr in (s.q, s.qd, s.tau_measured):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (hal.n_joints,)
        assert arr.dtype == float
    assert isinstance(s.timestamp, float)


def test_limits_shapes(hal):
    assert hal.joint_limits.shape == (hal.n_joints, 2)
    assert np.all(hal.joint_limits[:, 0] < hal.joint_limits[:, 1])
    assert hal.torque_limits.shape == (hal.n_joints,)
    assert np.all(hal.torque_limits > 0)


def test_limits_are_copies_not_internal_refs(hal):
    """상위 코드가 실수로 한계값을 덮어써도 HAL 내부가 망가지면 안 된다."""
    hal.joint_limits[:] = 0.0
    hal.torque_limits[:] = 0.0
    assert np.all(hal.torque_limits > 0)
    assert np.all(hal.joint_limits[:, 1] > 0)


def test_state_is_a_snapshot_not_a_live_view(hal):
    s1 = hal.read_state()
    hal.send_command(JointCommand.hold(np.full(hal.n_joints, 1.0), kp=50.0))
    s2 = hal.read_state()
    assert s1.q is not s2.q
    assert s1.timestamp < s2.timestamp


def test_joint_command_hold_and_damping_helpers():
    q = np.array([0.1, 0.2, 0.3])
    hold = JointCommand.hold(q, kp=25.0, kd=1.5)
    assert np.allclose(hold.q_des, q)
    assert np.allclose(hold.qd_des, 0.0)
    assert np.allclose(hold.kp, 25.0) and np.allclose(hold.kd, 1.5)
    hold.q_des[0] = 99.0
    assert q[0] == 0.1  # 원본을 건드리지 않는다

    damp = JointCommand.damping_only(3, kd=2.0)
    assert np.allclose(damp.kp, 0.0) and np.allclose(damp.kd, 2.0)
    assert np.allclose(damp.tau_ff, 0.0)


def test_impedance_law_matches_spec():
    """tau = kp*(q_des-q) + kd*(qd_des-qd) + tau_ff 를 목이 실제로 따르는지."""
    hal = MockRobotHAL(
        n_joints=1, dt=1e-3, coulomb_friction=0.0, viscous_friction=0.0, torque_limit=1e6
    )
    cmd = JointCommand(
        q_des=np.array([0.5]), qd_des=np.array([0.0]), tau_ff=np.array([1.0]),
        kp=np.array([10.0]), kd=np.array([0.5]),
    )
    hal.send_command(cmd)  # 첫 스텝: q=0, qd=0 에서 계산됨
    expected = 10.0 * (0.5 - 0.0) + 0.5 * (0.0 - 0.0) + 1.0
    assert hal.read_state().tau_measured[0] == pytest.approx(expected)


# ----------------------------------------------------------------- 실물 stub
def test_real_hal_is_a_robot_hal_subclass():
    assert issubclass(RealRobotHAL, RobotHAL)


def test_real_hal_methods_raise_not_implemented():
    real = RealRobotHAL(channel="can0", motor_ids=[1, 2, 3])
    assert real.n_joints == 3  # 이건 __init__에서 정해지므로 동작한다
    with pytest.raises(NotImplementedError):
        real.read_state()
    with pytest.raises(NotImplementedError):
        real.send_command(JointCommand.hold(np.zeros(3)))
    with pytest.raises(NotImplementedError):
        _ = real.joint_limits
    with pytest.raises(NotImplementedError):
        _ = real.torque_limits
    for method in ("connect", "enable", "disable", "estop", "close"):
        with pytest.raises(NotImplementedError):
            getattr(real, method)()


def test_real_hal_pack_unpack_roundtrip():
    """현장에서 쓸 고정소수점 헬퍼는 지금 검증해 둔다."""
    lo, hi, bits = -12.5, 12.5, 16
    for value in (-12.5, -3.0, 0.0, 4.2, 12.5):
        raw = RealRobotHAL._pack_float(value, lo, hi, bits)
        assert 0 <= raw < (1 << bits)
        assert RealRobotHAL._unpack_float(raw, lo, hi, bits) == pytest.approx(value, abs=1e-3)
