"""SafetyGuard — 화이트리스트, 클램프, 변화율 제한, 감사 로그. 가장 중요한 방어선."""

import math

import pytest

from robot_core.graph import ImpedanceNode, NodeGraphManager, TargetNode
from robot_core.recovery import ParamSpec, SafetyGuard


@pytest.fixture
def mgr():
    m = NodeGraphManager()
    m.add_node(ImpedanceNode(params={"kp": 40.0, "kd": 2.0}))
    m.add_node(TargetNode(params={"depth": 0.5, "retreat": 0.0}))
    return m


@pytest.fixture
def guard(mgr):
    return SafetyGuard(mgr, {
        "impedance.kp": ParamSpec(1.0, 120.0, max_rel_step=2.0),
        "impedance.kd": ParamSpec(0.1, 8.0, max_rel_step=2.0),
        "target.depth": ParamSpec(0.0, 1.0, max_rel_step=None, max_abs_step=0.2),
        "target.retreat": ParamSpec(0.0, 0.3, max_rel_step=None, max_abs_step=0.1),
    })


def apply_one(guard, node, param, value, source="test"):
    return guard.apply([{"node": node, "param": param, "value": value}], source)[0]


# ------------------------------------------------------------ 정상 적용
def test_valid_action_applies(guard, mgr):
    e = apply_one(guard, "impedance", "kp", 60.0)
    assert e.status == "applied" and e.applied == 60.0
    assert mgr.get_params("impedance")["kp"] == 60.0


# ---------------------------------------------------------- 화이트리스트
def test_unlisted_param_rejected(guard, mgr):
    e = apply_one(guard, "impedance", "torque_limit_override", 9999.0)
    assert e.status == "rejected"
    assert "whitelist" in e.reason
    assert "torque_limit_override" not in mgr.get_params("impedance")


def test_unlisted_node_rejected(guard):
    e = apply_one(guard, "motor_driver", "kp", 5.0)
    assert e.status == "rejected"


def test_whitelist_key_format_validated(mgr):
    with pytest.raises(ValueError):
        SafetyGuard(mgr, {"no_dot_here": ParamSpec(0, 1)})


# ------------------------------------------------------------- 값 검증
@pytest.mark.parametrize("bad", ["60", None, [60.0], {"v": 1}, True,
                                 float("nan"), float("inf")])
def test_non_numeric_or_non_finite_rejected(guard, mgr, bad):
    e = apply_one(guard, "impedance", "kp", bad)
    assert e.status == "rejected"
    assert mgr.get_params("impedance")["kp"] == 40.0


def test_malformed_action_rejected_not_crashed(guard):
    entries = guard.apply(["not-a-dict", {"node": 1, "param": "kp", "value": 5}], "llm")
    assert all(e.status == "rejected" for e in entries)


# -------------------------------------------------------------- 범위 클램프
def test_out_of_range_is_clamped_not_rejected(guard, mgr):
    """거부가 아니라 클램프 — 부분적으로라도 회복하게."""
    e = apply_one(guard, "impedance", "kd", 3.0e2)  # 상한 8, rel 한도 4
    # 변화율 제한이 먼저 걸린다: 2*2=4
    assert e.applied == 4.0
    e2 = apply_one(guard, "target", "depth", -5.0)  # abs_step 0.2 → 0.3, 범위 [0,1]
    assert e2.applied == 0.3
    assert e2.status == "rate-limited"


def test_range_clamp_without_rate_limit(guard, mgr):
    g = SafetyGuard(mgr, {"impedance.kp": ParamSpec(1.0, 120.0, max_rel_step=None)})
    e = apply_one(g, "impedance", "kp", 500.0)
    assert e.status == "clamped" and e.applied == 120.0
    assert mgr.get_params("impedance")["kp"] == 120.0


# ------------------------------------------------------------ 변화율 제한
def test_rel_step_limits_multiplicative_jumps(guard, mgr):
    e = apply_one(guard, "impedance", "kp", 400.0)  # kp 40 → 10배 시도
    assert e.status == "rate-limited"
    assert e.applied == 80.0  # 40 * 2.0
    assert mgr.get_params("impedance")["kp"] == 80.0


def test_rel_step_limits_downward_too(guard, mgr):
    e = apply_one(guard, "impedance", "kp", 0.001)
    assert e.applied == 20.0  # 40 / 2.0


def test_abs_step_limit(guard, mgr):
    e = apply_one(guard, "target", "depth", 1.0)  # 0.5 → 1.0 시도, abs 0.2
    assert e.status == "rate-limited" and e.applied == 0.7


def test_rate_limit_from_zero_uses_abs_step(guard, mgr):
    """현재값 0이면 배율 제한이 무의미 — abs_step이 커버해야 한다."""
    e = apply_one(guard, "target", "retreat", 0.3)  # 0 → 0.3 시도, abs 0.1
    assert e.applied == pytest.approx(0.1)


def test_multi_step_changes_eventually_reach_target(guard, mgr):
    """rate limit은 막는 게 아니라 늦추는 것 — 반복 적용으로 도달 가능해야 한다."""
    for _ in range(4):
        apply_one(guard, "impedance", "kp", 120.0)
    assert mgr.get_params("impedance")["kp"] == 120.0


def test_noop_change_is_recorded(guard):
    e = apply_one(guard, "impedance", "kp", 40.0)  # 현재값 그대로
    assert e.applied == 40.0
    assert "unchanged" in e.reason


# ------------------------------------------------------------ 독립 처리
def test_one_bad_action_does_not_block_others(guard, mgr):
    entries = guard.apply([
        {"node": "impedance", "param": "nope", "value": 1.0},
        {"node": "impedance", "param": "kd", "value": 3.0},
    ], source="llm")
    assert entries[0].status == "rejected"
    assert entries[1].status == "applied"
    assert mgr.get_params("impedance")["kd"] == 3.0


# ------------------------------------------------------------- 감사 로그
def test_audit_log_records_everything_in_order(guard):
    apply_one(guard, "impedance", "kp", 60.0, source="llm")
    apply_one(guard, "impedance", "bogus", 1.0, source="llm")
    apply_one(guard, "impedance", "kp", 999.0, source="rules:timeout")

    audit = guard.audit
    assert [e.status for e in audit] == ["applied", "rejected", "rate-limited"]
    assert audit[0].source == "llm"
    assert audit[2].source == "rules:timeout"
    assert audit[1].applied is None
    assert audit[2].requested == 999.0

    text = guard.dump_audit_text()
    assert "APPLIED" in text and "REJECTED" in text and "RATE-LIMITED" in text


def test_describe_whitelist_lists_ranges_and_current_values(guard):
    text = guard.describe_whitelist()
    assert "impedance.kp = 40" in text
    assert "[1, 120]" in text
    assert "max x2 per change" in text
    assert "target.retreat" in text


def test_param_spec_validation():
    with pytest.raises(ValueError):
        ParamSpec(5.0, 1.0)          # min >= max
    with pytest.raises(ValueError):
        ParamSpec(0, 1, max_rel_step=0.5)   # 배율은 > 1
    with pytest.raises(ValueError):
        ParamSpec(0, 1, max_abs_step=-1.0)
