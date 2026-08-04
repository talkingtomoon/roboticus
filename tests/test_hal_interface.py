"""내부 동역학 엔진(MockRobotHAL) 계약 테스트.

주의: 이 타입들은 phorce 피벗 이후 '목의 물리 엔진' 내부용이다.
상위 시스템의 경계는 hal/phorce.py의 PhorceHAL이다 (test_phorce_hal.py).
"""

import numpy as np
import pytest

from robot_core.hal import JointCommand, JointState, MockRobotHAL, RobotHAL


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


# (구 임피던스 RealRobotHAL stub 테스트는 legacy 이동과 함께 제거 —
#  실물 경계는 이제 PhorceHAL이고 test_phorce_hal.py가 담당한다)
