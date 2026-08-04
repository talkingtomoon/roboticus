"""제어 노드 그래프 (범용 DAG 인프라).

phorce 피벗 이후 새 아키텍처(감독 루프)는 그래프를 쓰지 않는다.
legacy 참고용 코드와 SafetyGuard(legacy)가 이 패키지를 임포트한다.
데모용 노드(Target/Impedance)는 robot_core/legacy/impedance_nodes.py로 이동.
"""

from robot_core.graph.node import Node
from robot_core.graph.manager import NodeGraphManager

__all__ = ["Node", "NodeGraphManager"]
