"""기본 제어 노드 — 데모/테스트용이자 새 노드 작성의 참고 예시.

그래프 위상 (데모 기준):
    target ──▶ impedance
    (q_des 생성)   (임피던스 지령 생성)

미션이 정해지면 target 노드만 미션별 궤적 생성기로 갈아끼우면 된다.
"""

from __future__ import annotations

import numpy as np

from robot_core.graph.node import Node
from robot_core.hal.interface import JointCommand, JointState


class TargetNode(Node):
    """조립 접근 목표 생성기.

    관절 0을 params["depth"] - params["retreat"] 위치로 밀어넣고
    나머지 관절은 0을 유지한다.

    params (전부 float — LLM이 조정 가능):
      depth   : 접근 깊이 목표 [rad]
      retreat : 후퇴량. 회복 전략이 "일단 물러나"를 표현하는 손잡이 [rad]
    """

    def __init__(self, name: str = "target", params: dict | None = None, enabled: bool = True):
        defaults = {"depth": 0.5, "retreat": 0.0}
        defaults.update(params or {})
        super().__init__(name, defaults, enabled)

    def update(self, inputs: dict) -> dict:
        state: JointState = inputs["state"]
        q_des = np.zeros(len(state.q))
        q_des[0] = float(self.params["depth"]) - float(self.params["retreat"])
        return {"q_des": q_des}


class ImpedanceNode(Node):
    """q_des + 현재 상태 → 임피던스 지령.

    params (전부 float — LLM이 조정 가능):
      kp : 위치 게인 (전 관절 공통 브로드캐스트)
      kd : 속도 게인
    """

    def __init__(self, name: str = "impedance", params: dict | None = None, enabled: bool = True):
        defaults = {"kp": 40.0, "kd": 2.0}
        defaults.update(params or {})
        super().__init__(name, defaults, enabled)

    def update(self, inputs: dict) -> dict:
        state: JointState = inputs["state"]
        n = len(state.q)
        q_des = inputs.get("q_des")
        if q_des is None:  # 업스트림이 꺼져 있으면 현재 위치 홀드 (안전 기본값)
            q_des = state.q.copy()
        qd_des = inputs.get("qd_des")  # 궤적 추종 노드(ChunkSwitchNode)가 주면 사용
        cmd = JointCommand(
            q_des=np.asarray(q_des, dtype=float),
            qd_des=np.zeros(n) if qd_des is None else np.asarray(qd_des, dtype=float),
            tau_ff=np.zeros(n),
            kp=np.full(n, float(self.params["kp"])),
            kd=np.full(n, float(self.params["kd"])),
        )
        return {"command": cmd}


STANDARD_NODE_TYPES = {
    "TargetNode": TargetNode,
    "ImpedanceNode": ImpedanceNode,
}
