"""자가 회복 루프: 실패 감지 → (LLM | 규칙) → 검증 → 파라미터 갱신."""

from robot_core.recovery.events import FailureEvent, FailureType
from robot_core.recovery.detector import DetectorConfig, FailureDetector
from robot_core.recovery.safety import AuditEntry, ParamSpec, SafetyGuard
from robot_core.recovery.rules import RuleAction, RuleBasedRecovery, default_rules
from robot_core.recovery.llm_agent import (
    AnthropicChatClient,
    LLMConfig,
    LLMRecoveryAgent,
    LLMUnavailable,
)
from robot_core.recovery.supervisor import RecoverySupervisor

__all__ = [
    "FailureType", "FailureEvent",
    "FailureDetector", "DetectorConfig",
    "SafetyGuard", "ParamSpec", "AuditEntry",
    "RuleBasedRecovery", "RuleAction", "default_rules",
    "LLMRecoveryAgent", "LLMConfig", "LLMUnavailable", "AnthropicChatClient",
    "RecoverySupervisor",
]
