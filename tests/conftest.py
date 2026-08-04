import numpy as np
import pytest

from robot_core.hal import JointCommand, MockRobotHAL


@pytest.fixture
def hal():
    """3관절 목 로봇, 1kHz."""
    return MockRobotHAL(n_joints=3, dt=1e-3, torque_limit=20.0)


def pd_command(hal, target, kp=40.0, kd=2.0) -> JointCommand:
    """target 위치를 잡는 임피던스 지령."""
    n = hal.n_joints
    return JointCommand(
        q_des=np.asarray(target, dtype=float),
        qd_des=np.zeros(n),
        tau_ff=np.zeros(n),
        kp=np.full(n, kp),
        kd=np.full(n, kd),
    )


def run_steps(hal, cmd, n_steps, logger=None):
    """목 로봇을 n_steps만큼 돌리고 마지막 상태를 반환."""
    state = hal.read_state()
    for _ in range(n_steps):
        hal.send_command(cmd)
        state = hal.read_state()
        if logger is not None:
            logger.log(state, cmd=cmd, loop_dt=1e-4, wall_time=state.timestamp)
    return state
