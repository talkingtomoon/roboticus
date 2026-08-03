"""규칙 기반 회복 폴백.

LLM이 없어도(API 죽어도, 타임아웃 나도, 헛소리해도) 시스템이 회복을 시도하게
하는 최소 전략. 값은 어차피 SafetyGuard를 다시 통과하므로 여기서는 방향만 잡는다.

기본 전략:
- TORQUE_SPIKE        → 후퇴(target retreat 증가) + kp 절반
- STALL               → kp 증가(지령 진폭/강성 키워 밀어붙임) + 후퇴 살짝 풀기
- OSCILLATION         → kd 증가 + kp 감소
- CONTINUOUS_OVERLOAD → 동작 속도 하향 (chunk_switch.time_scale × 0.6 —
                        열 예산 초과는 게인이 아니라 duty의 문제)
- OVERLOAD_CLEARED    → 속도 복원 (time_scale = 1.0. 내려가는 길만 있고
                        올라오는 길이 없으면 한 번 느려진 로봇이 영원히 긴다)
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_core.graph.manager import NodeGraphManager
from robot_core.recovery.events import FailureEvent, FailureType


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
