"""노드 그래프 매니저 — 위상 정렬, 런타임 API, YAML 로드/저장."""

import numpy as np
import pytest

from robot_core import JointState, MockRobotHAL
from robot_core.graph import (
    ImpedanceNode, Node, NodeGraphManager, STANDARD_NODE_TYPES, TargetNode,
)


class AddOne(Node):
    def update(self, inputs):
        return {"x": inputs.get("x", 0) + 1}


class Doubler(Node):
    def update(self, inputs):
        return {"x": inputs.get("x", 0) * 2}


class Recorder(Node):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = []

    def update(self, inputs):
        self.seen.append(dict(inputs))
        return {}


@pytest.fixture
def chain():
    """a(+1) -> b(x2) -> c(기록)."""
    mgr = NodeGraphManager()
    mgr.add_node(AddOne("a"))
    mgr.add_node(Doubler("b"))
    mgr.add_node(Recorder("c"))
    mgr.connect("a", "b")
    mgr.connect("b", "c")
    return mgr


# ------------------------------------------------------------------ 실행
def test_step_flows_outputs_downstream(chain):
    out = chain.step({"x": 10})
    assert out["a"] == {"x": 11}
    assert out["b"] == {"x": 22}  # a의 출력이 외부 입력을 덮는다
    recorder = chain.node("c")
    assert recorder.seen[-1]["x"] == 22


def test_external_inputs_visible_to_all_nodes(chain):
    chain.step({"x": 1, "extra": "hello"})
    assert chain.node("c").seen[-1]["extra"] == "hello"


def test_disabled_node_is_skipped_and_outputs_nothing(chain):
    chain.disable("b")
    out = chain.step({"x": 10})
    assert out["b"] == {}
    # c는 b의 출력을 못 받으므로 외부 x만 본다
    assert chain.node("c").seen[-1]["x"] == 10

    chain.enable("b")
    out = chain.step({"x": 10})
    assert out["b"] == {"x": 22}


def test_topo_order_respects_edges():
    mgr = NodeGraphManager()
    for name in ("z", "m", "a"):
        mgr.add_node(AddOne(name))
    mgr.connect("z", "a")
    mgr.connect("a", "m")
    order = mgr._topo_order()
    assert order.index("z") < order.index("a") < order.index("m")


def test_cycle_is_rejected_immediately():
    mgr = NodeGraphManager()
    mgr.add_node(AddOne("a"))
    mgr.add_node(AddOne("b"))
    mgr.connect("a", "b")
    with pytest.raises(ValueError, match="cycle"):
        mgr.connect("b", "a")


def test_self_loop_and_unknown_nodes_rejected():
    mgr = NodeGraphManager()
    mgr.add_node(AddOne("a"))
    with pytest.raises(ValueError):
        mgr.connect("a", "a")
    with pytest.raises(KeyError):
        mgr.connect("a", "ghost")
    with pytest.raises(ValueError):
        mgr.add_node(AddOne("a"))  # 중복 이름


# ------------------------------------------------------------ 런타임 API
def test_set_params_updates_existing_key():
    mgr = NodeGraphManager()
    mgr.add_node(ImpedanceNode(params={"kp": 10.0}))
    mgr.set_params("impedance", {"kp": 55.0})
    assert mgr.get_params("impedance")["kp"] == 55.0


def test_set_params_rejects_typo_keys():
    """오타로 새 파라미터가 생기는 사고 방지 — SafetyGuard도 여기에 의존한다."""
    mgr = NodeGraphManager()
    mgr.add_node(ImpedanceNode())
    with pytest.raises(KeyError):
        mgr.set_params("impedance", {"Kp": 55.0})


def test_get_params_returns_copy():
    mgr = NodeGraphManager()
    mgr.add_node(ImpedanceNode(params={"kp": 10.0}))
    mgr.get_params("impedance")["kp"] = 999.0
    assert mgr.get_params("impedance")["kp"] == 10.0


# ---------------------------------------------------------------- YAML
def test_graph_spec_and_yaml_roundtrip(tmp_path):
    mgr = NodeGraphManager()
    mgr.add_node(TargetNode(params={"depth": 0.7, "retreat": 0.05}))
    mgr.add_node(ImpedanceNode(params={"kp": 33.0, "kd": 1.1}, enabled=False))
    mgr.connect("target", "impedance")

    path = tmp_path / "graph.yaml"
    mgr.save_yaml(path)
    loaded = NodeGraphManager.from_yaml(path, STANDARD_NODE_TYPES)

    assert loaded.get_graph_spec() == mgr.get_graph_spec()
    assert loaded.node("impedance").enabled is False
    assert loaded.get_params("target")["depth"] == 0.7


def test_from_yaml_unknown_type_raises(tmp_path):
    path = tmp_path / "graph.yaml"
    path.write_text("nodes:\n- name: x\n  type: NoSuchNode\n  params: {}\nedges: []\n")
    with pytest.raises(KeyError, match="NoSuchNode"):
        NodeGraphManager.from_yaml(path, STANDARD_NODE_TYPES)


# ------------------------------------------------- 표준 노드 + HAL 연동
def test_target_impedance_pipeline_tracks_on_mock():
    mgr = NodeGraphManager()
    mgr.add_node(TargetNode(params={"depth": 0.3}))
    mgr.add_node(ImpedanceNode(params={"kp": 40.0, "kd": 2.0}))
    mgr.connect("target", "impedance")

    hal = MockRobotHAL(n_joints=3, dt=1e-3)
    for _ in range(2000):
        cmd = mgr.step({"state": hal.read_state()})["impedance"]["command"]
        hal.send_command(cmd)
    assert hal.read_state().q[0] == pytest.approx(0.3, abs=5e-3)


def test_impedance_node_holds_position_without_upstream():
    """target이 꺼져도 지령이 폭주하지 않고 현재 위치를 잡아야 한다."""
    mgr = NodeGraphManager()
    mgr.add_node(TargetNode())
    mgr.add_node(ImpedanceNode())
    mgr.connect("target", "impedance")
    mgr.disable("target")

    state = JointState(q=np.array([0.2, -0.1]), qd=np.zeros(2),
                       tau_measured=np.zeros(2), timestamp=0.0)
    cmd = mgr.step({"state": state})["impedance"]["command"]
    assert np.allclose(cmd.q_des, state.q)
