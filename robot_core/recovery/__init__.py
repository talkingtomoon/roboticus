"""회복 루프: 실패 감지 → (LLM | 규칙) 태그 결정 → 태그 가드 → 선택기로.

legacy 이름(SafetyGuard/ParamSpec/RuleBasedRecovery 등)도 계속 내보낸다 —
robot_core/legacy 참고용 코드가 임포트한다.
"""

from robot_core.recovery.events import FailureEvent, FailureType
from robot_core.recovery.detector import DetectorConfig, FailureDetector
from robot_core.recovery.safety import (
    AuditEntry, ParamSpec, SafetyGuard, TagAuditEntry, TagSafetyGuard,
)
from robot_core.recovery.rules import (
    DEFAULT_TAG_MAP, RuleAction, RuleBasedRecovery, TagDecision, TagRuleFallback,
    default_rules,
)
from robot_core.recovery.llm_agent import (
    AnthropicChatClient, LLMConfig, LLMRecoveryAgent, LLMUnavailable, check_model,
)

__all__ = [
    "FailureType", "FailureEvent",
    "FailureDetector", "DetectorConfig",
    "TagSafetyGuard", "TagAuditEntry",
    "TagRuleFallback", "TagDecision", "DEFAULT_TAG_MAP",
    "LLMRecoveryAgent", "LLMConfig", "LLMUnavailable", "AnthropicChatClient",
    "check_model",
    # legacy (robot_core/legacy 참고용)
    "SafetyGuard", "ParamSpec", "AuditEntry",
    "RuleBasedRecovery", "RuleAction", "default_rules",
]
