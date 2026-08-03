"""HAL 추상 인터페이스.

모든 알고리즘 모듈은 이 파일의 타입만 보고 개발한다.
실물/목 구현 어느 쪽이든 이 계약만 지키면 상위 코드는 수정 없이 동작해야 한다.

QDD 액추에이터 임피던스 제어 규약:
    tau = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointState:
    """관절 상태 스냅샷. 배열은 전부 shape=(n_joints,)."""

    q: np.ndarray            # 관절 위치 [rad]
    qd: np.ndarray           # 관절 속도 [rad/s]
    tau_measured: np.ndarray # 측정(추정) 토크 [Nm]
    timestamp: float         # [s] (mock: 시뮬레이션 시각, real: 드라이버 타임스탬프)

    def copy(self) -> "JointState":
        return JointState(
            q=self.q.copy(),
            qd=self.qd.copy(),
            tau_measured=self.tau_measured.copy(),
            timestamp=self.timestamp,
        )


@dataclass
class JointCommand:
    """임피던스 제어 지령. 배열은 전부 shape=(n_joints,)."""

    q_des: np.ndarray   # 목표 위치 [rad]
    qd_des: np.ndarray  # 목표 속도 [rad/s]
    tau_ff: np.ndarray  # 피드포워드 토크 [Nm]
    kp: np.ndarray      # 위치 게인 [Nm/rad]
    kd: np.ndarray      # 속도 게인 [Nm·s/rad]

    @classmethod
    def hold(cls, q: np.ndarray, kp: float = 20.0, kd: float = 1.0) -> "JointCommand":
        """현재 위치를 붙잡는 지령 (안전 정지 등에 사용)."""
        n = len(q)
        return cls(
            q_des=np.asarray(q, dtype=float).copy(),
            qd_des=np.zeros(n),
            tau_ff=np.zeros(n),
            kp=np.full(n, float(kp)),
            kd=np.full(n, float(kd)),
        )

    @classmethod
    def damping_only(cls, n_joints: int, kd: float = 2.0) -> "JointCommand":
        """kp=0, kd만 거는 소프트 스톱 지령."""
        return cls(
            q_des=np.zeros(n_joints),
            qd_des=np.zeros(n_joints),
            tau_ff=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.full(n_joints, float(kd)),
        )


class RobotHAL(ABC):
    """하드웨어 추상화 계층.

    구현체 의무:
    - read_state()는 호출 시점의 최신 상태를 반환한다 (블로킹 최소화).
    - send_command()는 지령을 하드웨어(또는 시뮬레이터)에 전달한다.
    - 두 메서드 모두 1kHz 루프 안에서 호출해도 될 만큼 가벼워야 한다.
    """

    @property
    @abstractmethod
    def n_joints(self) -> int:
        """관절 수."""

    @property
    @abstractmethod
    def joint_limits(self) -> np.ndarray:
        """관절 위치 한계, shape=(n_joints, 2), [:, 0]=min, [:, 1]=max [rad]."""

    @property
    @abstractmethod
    def torque_limits(self) -> np.ndarray:
        """관절별 최대 토크(대칭), shape=(n_joints,) [Nm]."""

    @abstractmethod
    def read_state(self) -> JointState:
        """현재 관절 상태를 읽는다."""

    @abstractmethod
    def send_command(self, cmd: JointCommand) -> None:
        """임피던스 지령을 보낸다."""
