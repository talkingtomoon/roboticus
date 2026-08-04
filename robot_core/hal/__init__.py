"""하드웨어 추상화 계층 — phorce 인터페이스.

interface.py의 JointState/JointCommand/RobotHAL은 목 동역학 엔진(mock.py)이
쓰는 내부 타입이다 (구 임피던스 HAL의 흔적 — 외부에서 새로 쓰지 말 것).
"""

from robot_core.hal.interface import JointCommand, JointState, RobotHAL
from robot_core.hal.mock import MockRobotHAL
from robot_core.hal.phorce import (
    MOTION_ID_MAX, MOTION_ID_MIN, N_AXES, OPERATOR_CODES, MockMotion,
    MockPhorceHAL, MotionAborted, MotionBusy, MotionRejected, PhorceError,
    PhorceFeedback, PhorceHAL, PlayHandle,
)
from robot_core.hal.profiles import PHACT_401, PROFILES, HardwareProfile

__all__ = [
    "PhorceHAL", "MockPhorceHAL", "MockMotion", "PhorceFeedback", "PlayHandle",
    "PhorceError", "MotionBusy", "MotionRejected", "MotionAborted",
    "N_AXES", "MOTION_ID_MIN", "MOTION_ID_MAX", "OPERATOR_CODES",
    "MockRobotHAL", "JointState", "JointCommand", "RobotHAL",
    "HardwareProfile", "PHACT_401", "PROFILES",
]
