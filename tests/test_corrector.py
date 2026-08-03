"""보정 노드 — 클램프/페이드인/킬스위치 (안전장치 3종, 타협 불가)."""

import numpy as np
import pytest

from robot_core import JointState
from robot_core.delta import DeltaCorrectorNode
from robot_core.hal.interface import JointCommand

N = 2
LIMITS = np.array([20.0, 20.0])


class ConstModel:
    """항상 같은 Δτ를 내는 테스트 모델."""

    model_type = "const"

    def __init__(self, delta):
        self.delta = np.asarray(delta, dtype=float)

    def reset(self):
        pass

    def predict(self, q, qd, q_des, qd_des):
        return self.delta.copy()


def state(t=0.0):
    return JointState(q=np.zeros(N), qd=np.zeros(N),
                      tau_measured=np.zeros(N), timestamp=t)


def cmd():
    return JointCommand(q_des=np.zeros(N), qd_des=np.zeros(N), tau_ff=np.zeros(N),
                        kp=np.full(N, 40.0), kd=np.full(N, 2.0))


def make_node(delta, fade_s=0.0, max_frac=0.3, gain=1.0):
    node = DeltaCorrectorNode(
        model=ConstModel(delta), torque_limits=LIMITS,
        params={"gain": gain, "max_frac": max_frac, "fade_s": fade_s})
    node.enable_correction()
    return node


# ------------------------------------------------------------------ 기본
def test_delta_added_to_tau_ff():
    node = make_node([1.5, -0.7])
    out = node.update({"state": state(), "command": cmd()})
    assert np.allclose(out["command"].tau_ff, [1.5, -0.7])
    assert np.allclose(out["delta_tau"], [1.5, -0.7])


def test_upstream_command_not_mutated():
    node = make_node([1.0, 1.0])
    c = cmd()
    node.update({"state": state(), "command": c})
    assert np.allclose(c.tau_ff, 0.0), "업스트림 명령이 변조됐다"


def test_no_model_passthrough():
    node = DeltaCorrectorNode(torque_limits=LIMITS)
    node.enable_correction()
    out = node.update({"state": state(), "command": cmd()})
    assert np.allclose(out["command"].tau_ff, 0.0)
    assert np.allclose(out["delta_tau"], 0.0)


# ------------------------------------------------------- 안전장치 ①: 클램프
def test_hard_clamp_limits_delta():
    node = make_node([100.0, -100.0], max_frac=0.3)
    out = node.update({"state": state(), "command": cmd()})
    assert np.allclose(out["delta_tau"], [6.0, -6.0])  # 0.3 * 20
    assert node.clamp_hits == 1


def test_clamp_respects_config():
    node = make_node([100.0, -100.0], max_frac=0.1)
    out = node.update({"state": state(), "command": cmd()})
    assert np.allclose(out["delta_tau"], [2.0, -2.0])


def test_gain_scales_delta():
    node = make_node([2.0, 2.0], gain=0.5)
    out = node.update({"state": state(), "command": cmd()})
    assert np.allclose(out["delta_tau"], [1.0, 1.0])


# ------------------------------------------------------ 안전장치 ②: 페이드인
def test_fade_in_ramps_from_zero_to_full():
    node = make_node([4.0, 4.0], fade_s=2.0)
    d0 = node.update({"state": state(0.0), "command": cmd()})["delta_tau"]
    d_half = node.update({"state": state(1.0), "command": cmd()})["delta_tau"]
    d_full = node.update({"state": state(2.5), "command": cmd()})["delta_tau"]
    assert np.allclose(d0, 0.0, atol=1e-9)
    assert np.allclose(d_half, 2.0, atol=0.1)
    assert np.allclose(d_full, 4.0)


def test_fade_restarts_on_reenable():
    node = make_node([4.0, 4.0], fade_s=1.0)
    node.update({"state": state(0.0), "command": cmd()})
    node.update({"state": state(2.0), "command": cmd()})  # 램프 완료
    node.disable_correction()
    node.enable_correction()                               # 다시 켬 → 램프 리셋
    d = node.update({"state": state(3.0), "command": cmd()})["delta_tau"]
    assert np.allclose(d, 0.0, atol=1e-9)
    d2 = node.update({"state": state(4.5), "command": cmd()})["delta_tau"]
    assert np.allclose(d2, 4.0)


# ----------------------------------------------------- 안전장치 ③: 킬스위치
def test_kill_switch_zeroes_immediately():
    node = make_node([4.0, 4.0], fade_s=0.0)
    out1 = node.update({"state": state(0.0), "command": cmd()})
    assert np.allclose(out1["delta_tau"], 4.0)

    node.disable_correction()
    out2 = node.update({"state": state(0.001), "command": cmd()})
    assert np.allclose(out2["delta_tau"], 0.0)
    assert np.allclose(out2["command"].tau_ff, 0.0)
    assert not node.correction_enabled


def test_params_are_floats_for_safety_guard_whitelist():
    """지름길 ①의 SafetyGuard가 다룰 수 있게 params는 전부 float."""
    node = make_node([1.0, 1.0])
    for k, v in node.params.items():
        assert isinstance(v, float), f"params[{k!r}]가 float이 아니다"
    # SafetyGuard 경로로 게인을 낮추는 시나리오
    from robot_core.graph import NodeGraphManager
    from robot_core.recovery import ParamSpec, SafetyGuard
    mgr = NodeGraphManager()
    mgr.add_node(node)
    guard = SafetyGuard(mgr, {"delta_corrector.gain":
                              ParamSpec(0.0, 1.0, max_rel_step=None, max_abs_step=0.5)})
    entries = guard.apply([{"node": "delta_corrector", "param": "gain", "value": 0.5}],
                          source="llm")
    assert entries[0].status == "applied"
    out = node.update({"state": state(), "command": cmd()})
    assert np.allclose(out["delta_tau"], [0.5, 0.5])  # gain 0.5 반영


# ---------------------------------------------------------------- 계측
def test_timing_report():
    node = make_node([1.0, 1.0])
    for k in range(200):
        node.update({"state": state(k * 1e-3), "command": cmd()})
    text = node.timing_report(budget_us_per_joint=50.0)
    assert "per joint" in text and "budget 50" in text
    assert "OK" in text or "OVER" in text
