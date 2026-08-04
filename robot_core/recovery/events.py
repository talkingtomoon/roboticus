"""실패 이벤트 타입."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureType(str, Enum):
    # ---- phorce 인터페이스 기준 (현행) ----
    PLAYBACK_STALL = "PLAYBACK_STALL"  # 재생 중인데 진행 없음 — 물체에 막힘
    IMPACT = "IMPACT"                  # dob_a 스파이크 — 외부 충격/접촉
    OVERHEAT = "OVERHEAT"              # temp_c 임계 초과 — 지속 과부하의 phorce 판
    OVERHEAT_CLEARED = "OVERHEAT_CLEARED"  # 과열 해제 통지 (실패 아님 — 복귀 트리거)
    AXIS_FAULT = "AXIS_FAULT"          # fault 비트 발화 — 축 이상

    # ---- legacy (임피던스 인터페이스 시절 — robot_core/legacy 참고용 코드가 참조) ----
    TORQUE_SPIKE = "TORQUE_SPIKE"
    STALL = "STALL"
    OSCILLATION = "OSCILLATION"
    CONTINUOUS_OVERLOAD = "CONTINUOUS_OVERLOAD"
    OVERLOAD_CLEARED = "OVERLOAD_CLEARED"


@dataclass
class FailureEvent:
    """감지기가 발행하는 실패 이벤트 하나.

    snapshot에는 판정 근거 수치를 담는다 (LLM 프롬프트와 감사 로그에 그대로 들어간다).
    """

    type: FailureType
    joint_idx: int
    severity: float          # 0..1 (1 = 심각). 판정 지표가 임계치를 얼마나 넘었는지
    t: float                 # 발생 시각 (로봇 클록 = JointState.timestamp 기준)
    snapshot: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        """쿨다운/이력 관리용 식별자."""
        return (self.type.value, self.joint_idx)

    def describe(self) -> str:
        nums = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.snapshot.items()
        )
        return (
            f"{self.type.value} joint={self.joint_idx} severity={self.severity:.2f} "
            f"t={self.t:.3f}s [{nums}]"
        )
