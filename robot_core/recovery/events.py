"""실패 이벤트 타입."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureType(str, Enum):
    TORQUE_SPIKE = "TORQUE_SPIKE"  # 움직이다 걸림 — 토크가 임계치 초과
    STALL = "STALL"                # 지령은 가는데 관절이 안 감 — 정지 중 고착 포함
    OSCILLATION = "OSCILLATION"    # 토크/속도 부호가 고주파로 반전 — 게인 과다
    # 지속 과부하: 지령 토크의 1초 이동평균이 연속 예산(tau_cont) 초과.
    # 순간 스파이크가 아니라 열 예산 문제 — 규칙 폴백은 동작 속도 하향.
    CONTINUOUS_OVERLOAD = "CONTINUOUS_OVERLOAD"
    # 위 과부하의 해제 통지 (실패가 아니라 상태 전이 — 회복 루프가 낮췄던
    # 속도를 복원하는 트리거. LLM/쿨다운을 태우지 않고 규칙에 직행시킨다).
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
