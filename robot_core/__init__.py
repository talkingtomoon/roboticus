"""robot_core: 해커톤용 코어 인프라.

- robot_core.hal      : 하드웨어 추상화 (인터페이스 / 목 / 실물 stub)
- robot_core.control  : 제어 루프 러너
- robot_core.logging  : 링버퍼 로거
"""

from robot_core.hal import JointCommand, JointState, MockRobotHAL, RealRobotHAL, RobotHAL
from robot_core.control import ControlLoopRunner, LoopStats
from robot_core.logging import RingLogger

__all__ = [
    "JointState",
    "JointCommand",
    "RobotHAL",
    "MockRobotHAL",
    "RealRobotHAL",
    "ControlLoopRunner",
    "LoopStats",
    "RingLogger",
]
