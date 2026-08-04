"""전체 흐름 데모: 러너로 1kHz 홀드 → 충격 외란 → jam → 로그 덤프.

실행:
    python examples/demo_impact_and_jam.py

출력 마지막의 dump_text가 지름길 ①(실패 상태 감지)에서 LLM에 그대로 먹일 텍스트다.
"""

import sys
from pathlib import Path

import numpy as np

# pip install -e . 없이 바로 실행되게 저장소 루트를 경로에 넣는다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core import ControlLoopRunner, JointCommand, MockRobotHAL, RingLogger  # noqa: E402

N = 4
TARGET = np.array([0.3, -0.2, 0.5, 0.0])
# jam 발생 후 관절 0에 새 목표를 준다. 걸린 관절을 계속 밀어야 토크가 치솟는다.
TARGET_AFTER_JAM = np.array([1.0, -0.2, 0.5, 0.0])


def main() -> None:
    hal = MockRobotHAL(
        n_joints=N,
        dt=1e-3,
        torque_limit=25.0,
        coulomb_friction=0.15,
        viscous_friction=0.05,
        backlash_width=0.01,
        enable_backlash=True,
    )
    logger = RingLogger.for_hal(hal, window_sec=3.0, threshold_frac=0.6)
    runner = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger)

    goal = {"q_des": TARGET}

    def hold_policy(state):
        return JointCommand(
            q_des=goal["q_des"],
            qd_des=np.zeros(N),
            tau_ff=np.zeros(N),
            kp=np.full(N, 50.0),
            kd=np.full(N, 2.0),
        )

    runner.set_policy(hold_policy)

    print("1) 목표 자세로 수렴 (0.3 s)")
    runner.run(duration_sec=0.3)

    print("2) 관절 2에 20 Nm 충격 외란 30 ms")
    hal.inject_disturbance(joint_idx=2, torque=20.0, duration=0.03)
    logger.mark_event("impact: 20 Nm / 30 ms on joint 2", t=hal.t)
    runner.run(duration_sec=0.2)

    print("3) 관절 0 jam 발생 + 새 목표 지령 (걸린 관절을 계속 밀게 된다)")
    hal.inject_jam(0)
    goal["q_des"] = TARGET_AFTER_JAM
    logger.mark_event("jam on joint 0; new target q_des[0]=1.0 commanded", t=hal.t)
    stats = runner.run(duration_sec=0.3)

    print()
    print(stats.summary())
    print()
    print(logger.dump_text(window_sec=0.6))


if __name__ == "__main__":
    main()
