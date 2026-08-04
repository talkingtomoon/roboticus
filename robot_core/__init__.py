"""robot_core: 해커톤 코어 인프라 — phorce 인터페이스 판.

아키텍처: "관찰 → 판단 → 선곡"
- robot_core.hal       : PhorceHAL 경계 (play(motion_id) + 1kHz 피드백) + 목
- robot_core.catalog   : 모션 메타데이터 (교시된 모션의 선곡 정보)
- robot_core.switching : 모션 선택기 (+ 베이스라인)
- robot_core.recovery  : 실패 감지 → LLM/규칙 태그 결정 → 태그 가드
- robot_core.supervisor: 2Hz 판단 루프 + 상태기계 (1kHz는 저장만)
- robot_core.legacy    : 임피던스 인터페이스 시절 스냅샷 (참고용)
"""

from robot_core.hal import (
    MockMotion, MockPhorceHAL, MotionAborted, MotionBusy, MotionRejected,
    PhorceFeedback, PhorceHAL, PlayHandle,
)
from robot_core.catalog import MotionCatalog, MotionMeta
from robot_core.logging import FeedbackCache, RingLogger
from robot_core.switching import MissionPlan, MotionSelector, SelectorWeights
from robot_core.supervisor import Supervisor, SupervisorConfig

__all__ = [
    "PhorceHAL", "MockPhorceHAL", "MockMotion", "PhorceFeedback", "PlayHandle",
    "MotionBusy", "MotionRejected", "MotionAborted",
    "MotionCatalog", "MotionMeta",
    "FeedbackCache", "RingLogger",
    "MotionSelector", "SelectorWeights", "MissionPlan",
    "Supervisor", "SupervisorConfig",
]
