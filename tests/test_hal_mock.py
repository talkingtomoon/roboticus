"""MockRobotHAL 동작 검증 — PD 추종, 마찰/백래시/포화 옵션, 외란·jam 주입."""

import numpy as np
import pytest

from robot_core import JointCommand, MockRobotHAL

from .conftest import pd_command, run_steps


# --------------------------------------------------------------- PD 추종
def test_pd_command_converges_to_target(hal):
    target = np.array([0.5, -0.3, 0.2])
    state = run_steps(hal, pd_command(hal, target), 3000)
    assert np.allclose(state.q, target, atol=5e-3)
    assert np.allclose(state.qd, 0.0, atol=1e-3)


def test_pd_tracking_error_shrinks_over_time(hal):
    """과도응답 세부는 안 보고, 시간이 갈수록 오차가 줄어드는지만 본다.

    정상상태 오차는 0으로 안 간다 — Coulomb 마찰(기본 0.1 Nm)이 만드는
    스틱 구간 때문에 kp=40에서 ~1e-3 rad 수준의 바닥값이 남는다. 정상 동작이다.
    """
    target = np.full(hal.n_joints, 0.4)
    cmd = pd_command(hal, target)
    err = lambda s: float(np.abs(s.q - target).max())

    e_early = err(run_steps(hal, cmd, 20))
    e_mid = err(run_steps(hal, cmd, 280))
    e_late = err(run_steps(hal, cmd, 2000))

    assert e_late < e_mid < e_early
    assert e_late < 0.01 * target.max()  # 목표의 1% 이내로 수렴


def test_higher_kp_gives_tighter_steady_state(hal):
    """Coulomb 마찰이 있으면 게인이 높을수록 정상상태 오차가 작아진다."""
    target = np.full(hal.n_joints, 0.3)
    soft = run_steps(MockRobotHAL(n_joints=3, dt=1e-3), pd_command(hal, target, kp=5.0), 4000)
    stiff = run_steps(MockRobotHAL(n_joints=3, dt=1e-3), pd_command(hal, target, kp=80.0), 4000)
    assert np.abs(stiff.q - target).max() < np.abs(soft.q - target).max()


def test_zero_gain_command_leaves_robot_limp(hal):
    limp = JointCommand.damping_only(hal.n_joints, kd=0.0)
    state = run_steps(hal, limp, 500)
    assert np.allclose(state.q, 0.0)
    assert np.allclose(state.tau_measured, 0.0)


# ------------------------------------------------------------ 마찰 / 포화
def test_viscous_friction_slows_free_spin():
    """tau_ff만 걸고 굴렸을 때 점성 마찰이 있으면 덜 돈다."""
    kwargs = dict(n_joints=1, dt=1e-3, coulomb_friction=0.0, torque_limit=100.0)
    cmd = JointCommand(
        q_des=np.zeros(1), qd_des=np.zeros(1), tau_ff=np.array([1.0]),
        kp=np.zeros(1), kd=np.zeros(1),
    )
    free = run_steps(MockRobotHAL(enable_viscous_friction=False, **kwargs), cmd, 500)
    damped = run_steps(MockRobotHAL(viscous_friction=5.0, **kwargs), cmd, 500)
    assert damped.qd[0] < free.qd[0]
    assert damped.q[0] < free.q[0]


def test_coulomb_friction_blocks_small_torque():
    """정지 마찰보다 작은 토크로는 관절이 안 움직인다."""
    hal = MockRobotHAL(n_joints=1, dt=1e-3, coulomb_friction=1.0, viscous_friction=0.0)
    weak = JointCommand(
        q_des=np.zeros(1), qd_des=np.zeros(1), tau_ff=np.array([0.5]),
        kp=np.zeros(1), kd=np.zeros(1),
    )
    assert run_steps(hal, weak, 1000).q[0] == pytest.approx(0.0)

    hal.reset()
    strong = JointCommand(
        q_des=np.zeros(1), qd_des=np.zeros(1), tau_ff=np.array([2.0]),
        kp=np.zeros(1), kd=np.zeros(1),
    )
    assert run_steps(hal, strong, 1000).q[0] > 0.1


def test_friction_can_be_disabled_entirely():
    hal = MockRobotHAL(
        n_joints=1, dt=1e-3, enable_coulomb_friction=False, enable_viscous_friction=False
    )
    cmd = JointCommand(
        q_des=np.zeros(1), qd_des=np.zeros(1), tau_ff=np.array([0.1]),
        kp=np.zeros(1), kd=np.zeros(1),
    )
    # 마찰이 없으면 아주 작은 토크로도 계속 가속한다.
    assert run_steps(hal, cmd, 1000).qd[0] == pytest.approx(0.1 / 0.05 * 1.0, rel=0.05)


def test_torque_saturation_caps_measured_torque():
    hal = MockRobotHAL(n_joints=1, dt=1e-3, torque_limit=5.0)
    huge = JointCommand(
        q_des=np.array([100.0]), qd_des=np.zeros(1), tau_ff=np.zeros(1),
        kp=np.array([1000.0]), kd=np.zeros(1),
    )
    hal.send_command(huge)
    assert hal.read_state().tau_measured[0] == pytest.approx(5.0)


def test_saturation_can_be_disabled():
    hal = MockRobotHAL(n_joints=1, dt=1e-3, torque_limit=5.0, enable_saturation=False)
    huge = JointCommand(
        q_des=np.array([1.0]), qd_des=np.zeros(1), tau_ff=np.zeros(1),
        kp=np.array([1000.0]), kd=np.zeros(1),
    )
    hal.send_command(huge)
    assert hal.read_state().tau_measured[0] > 5.0


def test_joint_limits_clamp_position():
    hal = MockRobotHAL(n_joints=1, dt=1e-3, joint_limit=(-0.2, 0.2), torque_limit=50.0)
    state = run_steps(hal, pd_command(hal, np.array([5.0]), kp=50.0), 1000)
    assert state.q[0] == pytest.approx(0.2)
    assert state.qd[0] == pytest.approx(0.0)


# ---------------------------------------------------------------- 백래시
def test_backlash_creates_deadzone_between_motor_and_link():
    width = 0.05
    hal = MockRobotHAL(
        n_joints=1, dt=1e-3, backlash_width=width, enable_backlash=True, coulomb_friction=0.0
    )
    run_steps(hal, pd_command(hal, np.array([0.5]), kp=30.0, kd=1.5), 3000)
    slack = abs(hal.q[0] - hal.read_state().q[0])
    assert slack <= width / 2 + 1e-9
    assert slack > 1e-4  # 실제로 데드존이 벌어져 있다


def test_backlash_off_means_link_equals_motor():
    hal = MockRobotHAL(n_joints=1, dt=1e-3, backlash_width=0.05, enable_backlash=False)
    run_steps(hal, pd_command(hal, np.array([0.3]), kp=30.0), 1000)
    assert hal.read_state().q[0] == pytest.approx(hal.q[0])


# ------------------------------------------------- 외란 주입 (지름길 ③ 용)
def test_disturbance_spikes_measured_torque(hal):
    """이게 핵심 기능: 외란을 넣으면 tau_measured가 실제로 튄다."""
    target = np.zeros(hal.n_joints)
    cmd = pd_command(hal, target)
    baseline = np.abs(run_steps(hal, cmd, 500).tau_measured[1])

    hal.inject_disturbance(joint_idx=1, torque=12.0, duration=0.02)
    peak = 0.0
    for _ in range(20):
        hal.send_command(cmd)
        peak = max(peak, abs(hal.read_state().tau_measured[1]))

    assert peak > baseline + 10.0
    assert peak == pytest.approx(12.0, abs=0.5)


def test_disturbance_expires_after_duration(hal):
    cmd = pd_command(hal, np.zeros(hal.n_joints))
    hal.inject_disturbance(joint_idx=0, torque=10.0, duration=0.01)
    run_steps(hal, cmd, 10)
    assert abs(hal.read_state().tau_measured[0]) > 5.0

    run_steps(hal, cmd, 5)  # 외란 종료 후
    assert np.allclose(hal.active_disturbance(), 0.0)
    assert abs(hal.read_state().tau_measured[0]) < 5.0


def test_disturbance_only_affects_target_joint(hal):
    cmd = pd_command(hal, np.zeros(hal.n_joints))
    run_steps(hal, cmd, 200)
    hal.inject_disturbance(joint_idx=2, torque=15.0, duration=0.05)

    peak = np.zeros(hal.n_joints)
    for _ in range(10):
        hal.send_command(cmd)
        peak = np.maximum(peak, np.abs(hal.read_state().tau_measured))

    assert peak[2] > 10.0
    assert peak[[0, 1]].max() < 1.0


def test_disturbance_actually_moves_the_joint(hal):
    """토크만 바뀌는 게 아니라 관절이 실제로 밀려야 한다 (충격 = 상태 교란)."""
    cmd = pd_command(hal, np.zeros(hal.n_joints), kp=5.0, kd=0.2)
    run_steps(hal, cmd, 200)
    q_before = hal.read_state().q[0]
    hal.inject_disturbance(joint_idx=0, torque=8.0, duration=0.05)
    state = run_steps(hal, cmd, 50)
    assert abs(state.q[0] - q_before) > 1e-3


def test_disturbances_accumulate_on_same_joint(hal):
    cmd = pd_command(hal, np.zeros(hal.n_joints))
    hal.inject_disturbance(0, 4.0, 0.05)
    hal.inject_disturbance(0, 3.0, 0.05)
    assert hal.active_disturbance()[0] == pytest.approx(7.0)


def test_inject_disturbance_validates_arguments(hal):
    with pytest.raises(IndexError):
        hal.inject_disturbance(99, 1.0, 0.01)
    with pytest.raises(ValueError):
        hal.inject_disturbance(0, 1.0, 0.0)


# ---------------------------------------------------- jam 주입 (지름길 ① 용)
def test_jam_freezes_joint_and_spikes_torque(hal):
    """jam이면 관절이 안 움직이고 PD 오차가 쌓여 토크가 임계치를 넘는다."""
    threshold = 0.8 * hal.torque_limits[0]
    hal.inject_jam(0)
    state = run_steps(hal, pd_command(hal, np.array([1.0, 0.0, 0.0]), kp=40.0), 200)

    assert hal.is_jammed(0)
    assert state.q[0] == pytest.approx(0.0)
    assert state.qd[0] == pytest.approx(0.0)
    assert abs(state.tau_measured[0]) > threshold


def test_jam_saturates_at_torque_limit(hal):
    hal.inject_jam(1)
    state = run_steps(hal, pd_command(hal, np.array([0.0, 2.0, 0.0]), kp=100.0), 100)
    assert abs(state.tau_measured[1]) == pytest.approx(hal.torque_limits[1])


def test_jam_does_not_affect_other_joints(hal):
    hal.inject_jam(0)
    target = np.array([0.5, 0.5, 0.5])
    state = run_steps(hal, pd_command(hal, target), 3000)
    assert state.q[0] == pytest.approx(0.0)
    assert np.allclose(state.q[1:], target[1:], atol=5e-3)


def test_clear_jam_lets_joint_move_again(hal):
    cmd = pd_command(hal, np.array([0.5, 0.0, 0.0]))
    hal.inject_jam(0)
    run_steps(hal, cmd, 100)
    assert hal.read_state().q[0] == pytest.approx(0.0)

    hal.clear_jam(0)
    assert not hal.is_jammed(0)
    assert run_steps(hal, cmd, 2000).q[0] == pytest.approx(0.5, abs=5e-3)


def test_clear_jam_all(hal):
    hal.inject_jam(0)
    hal.inject_jam(2)
    hal.clear_jam()
    assert not any(hal.is_jammed(j) for j in range(hal.n_joints))


def test_inject_jam_validates_index(hal):
    with pytest.raises(IndexError):
        hal.inject_jam(-1)
    with pytest.raises(IndexError):
        hal.inject_jam(3)


# ------------------------------------------------------------------ 기타
def test_reset_clears_state_and_injections(hal):
    hal.inject_jam(0)
    hal.inject_disturbance(1, 5.0, 1.0)
    run_steps(hal, pd_command(hal, np.full(3, 0.4)), 100)

    hal.reset()
    state = hal.read_state()
    assert state.timestamp == 0.0
    assert np.allclose(state.q, 0.0) and np.allclose(state.qd, 0.0)
    assert np.allclose(state.tau_measured, 0.0)
    assert not hal.is_jammed(0)
    assert np.allclose(hal.active_disturbance(), 0.0)


def test_timestamp_advances_by_dt(hal):
    cmd = pd_command(hal, np.zeros(3))
    hal.send_command(cmd)
    hal.send_command(cmd)
    assert hal.read_state().timestamp == pytest.approx(2 * hal.dt)


def test_scalar_and_vector_params_both_accepted():
    hal = MockRobotHAL(n_joints=2, inertia=[0.1, 0.2], torque_limit=[10.0, 20.0])
    assert np.allclose(hal.inertia, [0.1, 0.2])
    assert np.allclose(hal.torque_limits, [10.0, 20.0])
    with pytest.raises(ValueError):
        MockRobotHAL(n_joints=2, inertia=[0.1, 0.2, 0.3])
