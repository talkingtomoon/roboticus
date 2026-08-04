"""규칙 기반 회복 폴백.

LLM이 없어도(API 죽어도, 타임아웃 나도, 헛소리해도) 시스템이 회복을 시도하게
하는 최소 전략.

phorce 판 (현행): **TagRuleFallback** — 실패 타입 → (의도 태그, 긴급도) 매핑.
LLM과 동일하게 태그만 내고, motion_id는 선택기가 고른다.

파일 하단의 RuleBasedRecovery(파라미터 조정)는 임피던스 시절 —
robot_core/legacy 참고용 코드가 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_core.graph.manager import NodeGraphManager
from robot_core.recovery.events import FailureEvent, FailureType


# ==========================================================================
# phorce 판 (현행): 실패 타입 → (의도 태그, 긴급도)
# ==========================================================================
# 긴급도 의미 (supervisor가 해석):
#   normal = 계획 계속, slow = 저속 변주 선호(선택기 modifier), stop = 휴지 후 관망
DEFAULT_TAG_MAP: dict[FailureType, tuple[str, str]] = {
    FailureType.IMPACT: ("retreat", "normal"),          # 충격 → 일단 물러난다
    FailureType.PLAYBACK_STALL: ("retry", "slow"),      # 막힘 → 천천히 재시도
    FailureType.OVERHEAT: ("rest", "stop"),             # 과열 → 휴지 삽입
    FailureType.AXIS_FAULT: ("rest", "stop"),           # 축 이상 → 안전 휴지
}


@dataclass
class TagDecision:
    """회복 결정: 의도 태그 + 긴급도. motion_id가 아니다 — 선곡은 선택기가."""

    intent_tag: str
    urgency: str          # "normal" | "slow" | "stop"
    source: str           # "llm" | "rules:<이유>"
    reasoning: str = ""
    confidence: float = 1.0


class TagRuleFallback:
    """실패 타입 → 태그 매핑 테이블. LLM의 결정적 대역.

    매핑에 없는 실패 타입은 default로 — 모르는 상황에서 제일 안전한 선택은
    '쉬면서 관망'이다.
    """

    def __init__(self, mapping: dict[FailureType, tuple[str, str]] | None = None,
                 default: tuple[str, str] = ("rest", "stop")) -> None:
        self.mapping = dict(DEFAULT_TAG_MAP if mapping is None else mapping)
        self.default = tuple(default)

    def propose(self, event: FailureEvent) -> tuple[str, str]:
        return self.mapping.get(event.type, self.default)


@dataclass
class RuleAction:
    """현재값 기준의 상대 조정. op: "scale" | "delta" | "set"."""

    node: str
    param: str
    op: str
    operand: float

    def resolve(self, current: float) -> float:
        if self.op == "scale":
            return current * self.operand
        if self.op == "delta":
            return current + self.operand
        if self.op == "set":
            return self.operand
        raise ValueError(f"unknown op {self.op!r}")


def default_rules(
    impedance_node: str = "impedance",
    target_node: str = "target",
    chunk_node: str = "chunk_switch",
) -> dict[FailureType, list[RuleAction]]:
    """표준 그래프 기준 기본 전략. 노드 이름만 바꿔 재사용.

    그래프에 없는 노드를 참조하는 규칙은 propose()가 조용히 건너뛴다 —
    chunk_switch 없는 단순 그래프에서도 나머지 규칙은 그대로 동작한다.
    """
    return {
        FailureType.TORQUE_SPIKE: [
            RuleAction(target_node, "retreat", "delta", +0.05),
            RuleAction(impedance_node, "kp", "scale", 0.5),
        ],
        FailureType.STALL: [
            RuleAction(impedance_node, "kp", "scale", 2.0),
            RuleAction(target_node, "retreat", "delta", -0.02),
        ],
        FailureType.OSCILLATION: [
            RuleAction(impedance_node, "kd", "scale", 1.8),
            RuleAction(impedance_node, "kp", "scale", 0.6),
        ],
        FailureType.CONTINUOUS_OVERLOAD: [
            RuleAction(chunk_node, "time_scale", "scale", 0.6),
        ],
        FailureType.OVERLOAD_CLEARED: [
            RuleAction(chunk_node, "time_scale", "set", 1.0),
        ],
    }


class RuleBasedRecovery:
    def __init__(
        self,
        manager: NodeGraphManager,
        rules: dict[FailureType, list[RuleAction]] | None = None,
    ) -> None:
        self.manager = manager
        self.rules = rules if rules is not None else default_rules()

    def propose(self, event: FailureEvent) -> list[dict]:
        """이벤트에 대한 액션 리스트. 존재하지 않는 노드/파라미터는 조용히 건너뛴다.

        (그래프 구성이 데모와 다르면 규칙 일부가 안 맞을 수 있다 — 그래도
        나머지는 적용되게. 어차피 SafetyGuard가 한 번 더 거른다.)
        """
        actions: list[dict] = []
        for ra in self.rules.get(event.type, []):
            try:
                current = float(self.manager.get_params(ra.node)[ra.param])
            except KeyError:
                continue
            actions.append({"node": ra.node, "param": ra.param, "value": ra.resolve(current)})
        return actions
