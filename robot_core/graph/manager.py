"""노드 그래프 매니저 (ROS 2 비의존).

- 노드 등록 + 방향 간선으로 DAG 구성, 위상 정렬 순서로 실행
- 런타임에 enable/disable/set_params (스레드 안전 — LLM 워커가 제어 루프와
  다른 스레드에서 파라미터를 바꾼다)
- YAML로 그래프 스펙 저장/로드 (현장에서 코드 안 고치고 위상 변경)

실행 규칙:
- step(external)은 위상 정렬 순서로 각 노드의 update()를 부른다
- 노드의 inputs = {**external, **(업스트림 출력들을 위상 순서대로 merge)}
  키가 겹치면 나중에 실행된 업스트림이 이긴다 (단순함 우선; 겹치지 않게 설계할 것)
- disabled 노드는 건너뛰고 출력 {} 취급. 다운스트림은 그 키를 못 받는다
"""

from __future__ import annotations

import threading
from pathlib import Path

import yaml

from robot_core.graph.node import Node


class NodeGraphManager:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[tuple[str, str]] = []
        self._order: list[str] | None = None  # 위상 정렬 캐시
        self._lock = threading.RLock()

    # ------------------------------------------------------------- 그래프 구성
    def add_node(self, node: Node) -> Node:
        with self._lock:
            if node.name in self._nodes:
                raise ValueError(f"node {node.name!r} already registered")
            self._nodes[node.name] = node
            self._order = None
        return node

    def connect(self, src: str, dst: str) -> None:
        """src의 출력을 dst의 입력으로 흘린다."""
        with self._lock:
            for name in (src, dst):
                if name not in self._nodes:
                    raise KeyError(f"unknown node {name!r}")
            if src == dst:
                raise ValueError("self-loop not allowed")
            edge = (src, dst)
            if edge not in self._edges:
                self._edges.append(edge)
                self._order = None
                self._topo_order()  # 사이클이면 여기서 바로 터뜨린다

    def node(self, name: str) -> Node:
        with self._lock:
            if name not in self._nodes:
                raise KeyError(f"unknown node {name!r}")
            return self._nodes[name]

    def node_names(self) -> list[str]:
        with self._lock:
            return list(self._nodes)

    def _topo_order(self) -> list[str]:
        with self._lock:
            if self._order is not None:
                return self._order
            indeg = {n: 0 for n in self._nodes}
            for _, dst in self._edges:
                indeg[dst] += 1
            ready = sorted(n for n, d in indeg.items() if d == 0)  # 이름순 → 결정적
            order: list[str] = []
            indeg = dict(indeg)
            while ready:
                n = ready.pop(0)
                order.append(n)
                for src, dst in self._edges:
                    if src == n:
                        indeg[dst] -= 1
                        if indeg[dst] == 0:
                            ready.append(dst)
                ready.sort()
            if len(order) != len(self._nodes):
                cyclic = sorted(set(self._nodes) - set(order))
                raise ValueError(f"graph has a cycle involving: {cyclic}")
            self._order = order
            return order

    def _upstreams(self, name: str) -> list[str]:
        order = self._topo_order()
        pos = {n: i for i, n in enumerate(order)}
        ups = [src for src, dst in self._edges if dst == name]
        return sorted(ups, key=lambda s: pos[s])

    # ---------------------------------------------------------------- 실행
    def step(self, external: dict | None = None) -> dict[str, dict]:
        """전체 그래프 1스텝. 반환: {node_name: output_dict}."""
        external = external or {}
        with self._lock:
            order = self._topo_order()
            outputs: dict[str, dict] = {}
            for name in order:
                node = self._nodes[name]
                if not node.enabled:
                    outputs[name] = {}
                    continue
                inputs = dict(external)
                for src in self._upstreams(name):
                    inputs.update(outputs.get(src, {}))
                out = node.update(inputs)
                outputs[name] = out if isinstance(out, dict) else {}
            return outputs

    # ------------------------------------------------------------ 런타임 API
    def enable(self, name: str) -> None:
        self.node(name).enabled = True

    def disable(self, name: str) -> None:
        self.node(name).enabled = False

    def set_params(self, name: str, updates: dict) -> None:
        """존재하는 파라미터만 갱신. 오타로 새 키가 생기는 사고를 막는다."""
        with self._lock:
            node = self.node(name)
            unknown = set(updates) - set(node.params)
            if unknown:
                raise KeyError(f"node {name!r} has no params {sorted(unknown)} "
                               f"(existing: {sorted(node.params)})")
            node.params.update(updates)

    def get_params(self, name: str) -> dict:
        with self._lock:
            return dict(self.node(name).params)

    def get_graph_spec(self) -> dict:
        """직렬화 가능한 그래프 스펙 (YAML 저장 포맷과 동일)."""
        with self._lock:
            return {
                "nodes": [
                    {
                        "name": n.name,
                        "type": type(n).__name__,
                        "enabled": bool(n.enabled),
                        "params": {k: v for k, v in n.params.items()},
                    }
                    for n in (self._nodes[name] for name in self._topo_order())
                ],
                "edges": [[s, d] for s, d in self._edges],
            }

    # ------------------------------------------------------------------ YAML
    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.get_graph_spec(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def from_yaml(cls, path: str | Path, node_types: dict[str, type[Node]]) -> "NodeGraphManager":
        """YAML 스펙에서 그래프를 재구성한다.

        node_types: {"TargetNode": TargetNode, ...} — 스펙의 type 문자열을
        실제 클래스로 매핑. 노드 생성자는 (name, params=None, enabled=True)
        시그니처를 지켜야 한다.
        """
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        mgr = cls()
        for ns in spec.get("nodes", []):
            type_name = ns["type"]
            if type_name not in node_types:
                raise KeyError(
                    f"unknown node type {type_name!r} — node_types에 클래스를 넘길 것"
                )
            mgr.add_node(
                node_types[type_name](
                    name=ns["name"],
                    params=ns.get("params") or {},
                    enabled=ns.get("enabled", True),
                )
            )
        for src, dst in spec.get("edges", []):
            mgr.connect(src, dst)
        return mgr
