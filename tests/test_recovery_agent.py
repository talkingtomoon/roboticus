"""LLM 회복 에이전트 — 목 클라이언트로 모든 실패 경로를 검증한다.

실제 API 호출 금지. 모든 테스트는 주입된 fake client만 쓴다.
"""

import json
import time

import numpy as np
import pytest

from robot_core import ControlLoopRunner, MockRobotHAL, RingLogger
from robot_core.graph import ImpedanceNode, NodeGraphManager, TargetNode
from robot_core.recovery import (
    DetectorConfig, FailureDetector, FailureEvent, FailureType, LLMConfig,
    LLMRecoveryAgent, ParamSpec, RecoverySupervisor, RuleBasedRecovery, SafetyGuard,
)


def make_stack(client=None, config=None, logger=None):
    """graph + guard + rules + agent 조립."""
    mgr = NodeGraphManager()
    mgr.add_node(TargetNode(params={"depth": 0.5, "retreat": 0.0}))
    mgr.add_node(ImpedanceNode(params={"kp": 40.0, "kd": 2.0}))
    mgr.connect("target", "impedance")
    guard = SafetyGuard(mgr, {
        "impedance.kp": ParamSpec(1.0, 120.0, max_rel_step=2.0),
        "impedance.kd": ParamSpec(0.1, 8.0, max_rel_step=2.0),
        "target.depth": ParamSpec(0.0, 1.0, max_rel_step=None, max_abs_step=0.2),
        "target.retreat": ParamSpec(0.0, 0.3, max_rel_step=None, max_abs_step=0.1),
    })
    rules = RuleBasedRecovery(mgr)
    agent = LLMRecoveryAgent(
        guard, rules, logger=logger,
        config=config or LLMConfig(timeout_s=0.5, cooldown_s=0.2),
        client=client,
    )
    return mgr, guard, rules, agent


def stall_event(joint=0, t=1.0):
    return FailureEvent(type=FailureType.STALL, joint_idx=joint, severity=0.8, t=t,
                        snapshot={"err": 0.3})


def handle_sync(agent, event):
    """워커 스레드 없이 처리 로직만 직접 실행 (결정적 테스트)."""
    agent._handle(event)


# ------------------------------------------------------------ LLM 정상 경로
def test_valid_llm_response_is_applied():
    def client(system, user):
        return json.dumps({"diagnosis": "raise stiffness",
                           "actions": [{"node": "impedance", "param": "kp", "value": 70.0}],
                           "confidence": 0.9})

    mgr, guard, _, agent = make_stack(client)
    handle_sync(agent, stall_event())
    assert mgr.get_params("impedance")["kp"] == 70.0
    assert agent.stats["llm_ok"] == 1
    assert agent.stats["llm_fallback"] == 0
    assert guard.audit[0].source == "llm"


def test_prompt_contains_failure_log_whitelist_and_schema():
    prompts = {}

    def client(system, user):
        prompts["system"] = system
        prompts["user"] = user
        return json.dumps({"diagnosis": "d", "actions": [], "confidence": 0.9})

    hal = MockRobotHAL(n_joints=2, dt=1e-3)
    logger = RingLogger.for_hal(hal)
    from .conftest import pd_command, run_steps
    run_steps(hal, pd_command(hal, np.zeros(2)), 50, logger=logger)

    _, _, _, agent = make_stack(client, logger=logger)
    handle_sync(agent, stall_event())

    assert "JSON" in prompts["system"] and "No markdown" in prompts["system"]
    assert "STALL" in prompts["user"]
    assert "ROBOT STATE DUMP" in prompts["user"]           # RingLogger.dump_text
    assert "impedance.kp" in prompts["user"]               # 화이트리스트 + 범위
    assert "allowed range" in prompts["user"]


def test_repeat_failure_includes_previous_attempts_in_prompt():
    calls = []

    def client(system, user):
        calls.append(user)
        return json.dumps({"diagnosis": "try kp up",
                           "actions": [{"node": "impedance", "param": "kp", "value": 60.0}],
                           "confidence": 0.9})

    _, _, _, agent = make_stack(client)
    handle_sync(agent, stall_event(t=1.0))
    handle_sync(agent, stall_event(t=2.0))

    assert "PREVIOUS RECOVERY ATTEMPTS" not in calls[0]
    assert "PREVIOUS RECOVERY ATTEMPTS" in calls[1]
    assert "Do not repeat" in calls[1]
    assert "kp" in calls[1]


# ------------------------------------------------------- 악의적/불량 응답
def test_malicious_response_cannot_do_damage():
    """미등록 파라미터 + 말도 안 되는 값 → 등록된 것만 rate-limit 걸려 적용."""
    def client(system, user):
        return json.dumps({
            "diagnosis": "trust me",
            "actions": [
                {"node": "impedance", "param": "torque_limit", "value": 99999.0},  # 미등록
                {"node": "estop", "param": "disable", "value": 1.0},               # 미등록 노드
                {"node": "impedance", "param": "kp", "value": 99999.0},            # 폭주 값
                {"node": "impedance", "param": "kd", "value": float("nan")},       # NaN
            ],
            "confidence": 1.0,
        })

    mgr, guard, _, agent = make_stack(client)
    handle_sync(agent, stall_event())

    params = mgr.get_params("impedance")
    assert params["kp"] == 80.0            # 40 * max_rel_step(2.0) 까지만
    assert params["kd"] == 2.0             # NaN 거부
    assert "torque_limit" not in params
    statuses = [e.status for e in guard.audit]
    assert statuses == ["rejected", "rejected", "rate-limited", "rejected"]


@pytest.mark.parametrize("bad_response", [
    "The robot is stuck. I suggest raising kp to 60.",       # 산문
    '{"diagnosis": "d", "actions": "kp=60", "confidence": 1}',  # 스키마 위반
    '{"diagnosis": "d"}',                                      # 필드 누락
    '{"diagnosis": "d", "actions": [], "confidence": "high"}', # 타입 위반
    '[1, 2, 3]',                                               # 객체 아님
    '',                                                        # 빈 응답
    '{"broken json',                                           # 파싱 불가
])
def test_unparseable_or_invalid_response_falls_back_to_rules(bad_response):
    mgr, guard, _, agent = make_stack(lambda s, u: bad_response)
    kp_before = mgr.get_params("impedance")["kp"]
    handle_sync(agent, stall_event())

    assert agent.stats["llm_fallback"] == 1
    # STALL 규칙: kp*2 (rate limit 2.0과 일치 → 80)
    assert mgr.get_params("impedance")["kp"] == kp_before * 2.0
    assert all(e.source.startswith("rules:") for e in guard.audit)


def test_fenced_json_is_tolerated_once():
    """프롬프트로 금지했지만 백틱이 와도 내용이 유효하면 살린다."""
    fenced = "```json\n" + json.dumps(
        {"diagnosis": "d",
         "actions": [{"node": "impedance", "param": "kd", "value": 3.0}],
         "confidence": 0.8}) + "\n```"
    mgr, _, _, agent = make_stack(lambda s, u: fenced)
    handle_sync(agent, stall_event())
    assert mgr.get_params("impedance")["kd"] == 3.0


def test_low_confidence_falls_back():
    def client(system, user):
        return json.dumps({"diagnosis": "not sure",
                           "actions": [{"node": "impedance", "param": "kp", "value": 5.0}],
                           "confidence": 0.1})

    mgr, _, _, agent = make_stack(client)
    handle_sync(agent, stall_event())
    assert agent.stats["llm_fallback"] == 1
    assert mgr.get_params("impedance")["kp"] == 80.0  # 규칙(kp*2)이 적용됨


def test_client_exception_falls_back():
    def client(system, user):
        raise ConnectionError("API down")

    mgr, _, _, agent = make_stack(client)
    handle_sync(agent, stall_event())
    assert agent.stats["llm_fallback"] == 1
    assert mgr.get_params("impedance")["kp"] == 80.0


# ------------------------------------------------------------- 타임아웃
def test_timeout_discards_response_and_falls_back():
    def slow_client(system, user):
        time.sleep(2.0)  # timeout 0.5s보다 훨씬 김
        return json.dumps({"diagnosis": "too late",
                           "actions": [{"node": "impedance", "param": "kp", "value": 5.0}],
                           "confidence": 1.0})

    mgr, _, _, agent = make_stack(slow_client)
    t0 = time.perf_counter()
    handle_sync(agent, stall_event())
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.5, "타임아웃이 안 걸렸다"
    assert agent.stats["llm_timeout"] == 1
    assert mgr.get_params("impedance")["kp"] == 80.0  # 늦은 응답(kp=5)은 폐기


# ---------------------------------------------------------- LLM 없이 동작
def test_no_client_goes_straight_to_rules():
    mgr, guard, _, agent = make_stack(client=None)
    handle_sync(agent, stall_event())
    assert agent.stats["llm_fallback"] == 1
    assert mgr.get_params("impedance")["kp"] == 80.0
    assert guard.audit[0].source.startswith("rules:")


def test_rule_fallbacks_for_each_failure_type():
    mgr, _, rules, _ = make_stack()
    kp0 = mgr.get_params("impedance")["kp"]
    kd0 = mgr.get_params("impedance")["kd"]

    spike = rules.propose(FailureEvent(FailureType.TORQUE_SPIKE, 0, 1.0, 0.0))
    assert {(a["node"], a["param"]) for a in spike} == {("target", "retreat"), ("impedance", "kp")}
    assert next(a for a in spike if a["param"] == "kp")["value"] == kp0 * 0.5

    osc = rules.propose(FailureEvent(FailureType.OSCILLATION, 0, 1.0, 0.0))
    assert next(a for a in osc if a["param"] == "kd")["value"] == pytest.approx(kd0 * 1.8)
    assert next(a for a in osc if a["param"] == "kp")["value"] == pytest.approx(kp0 * 0.6)


def test_rules_skip_missing_nodes_gracefully():
    mgr = NodeGraphManager()
    mgr.add_node(ImpedanceNode())  # target 노드 없음
    rules = RuleBasedRecovery(mgr)
    actions = rules.propose(FailureEvent(FailureType.TORQUE_SPIKE, 0, 1.0, 0.0))
    assert all(a["node"] == "impedance" for a in actions)  # retreat 규칙은 조용히 스킵


# --------------------------------------------------------------- 쿨다운/큐
def test_cooldown_drops_repeated_same_failure():
    _, _, _, agent = make_stack(config=LLMConfig(cooldown_s=10.0))
    ev = stall_event()
    assert agent.submit(ev) is True
    assert agent.submit(ev) is False
    assert agent.stats["dropped_cooldown"] == 1
    # 다른 관절의 같은 타입은 별개 채널
    assert agent.submit(stall_event(joint=1)) is True


def test_cooldown_expires():
    fake_time = [0.0]
    _, _, _, agent = make_stack(config=LLMConfig(cooldown_s=5.0))
    agent._time = lambda: fake_time[0]
    assert agent.submit(stall_event()) is True
    fake_time[0] = 3.0
    assert agent.submit(stall_event()) is False
    fake_time[0] = 6.0
    assert agent.submit(stall_event()) is True


# ------------------------------------------- 절대 원칙 ①: 루프 비블로킹
def test_control_loop_frequency_maintained_during_llm_call():
    """LLM이 응답 중이어도(느려도) 제어 루프 주파수는 유지돼야 한다."""
    def slow_client(system, user):
        time.sleep(1.0)  # 루프 실행 시간 내내 "응답 대기 중"
        return json.dumps({"diagnosis": "slow but valid",
                           "actions": [{"node": "impedance", "param": "kp", "value": 60.0}],
                           "confidence": 0.9})

    mgr, _, _, agent = make_stack(
        slow_client, config=LLMConfig(timeout_s=5.0, cooldown_s=0.1))
    hal = MockRobotHAL(n_joints=3, dt=1e-3)
    runner = ControlLoopRunner(hal, rate_hz=500.0)
    runner.set_policy(lambda st: mgr.step({"state": st})["impedance"]["command"])

    agent.start()
    try:
        agent.submit(stall_event())  # LLM 처리 시작 (백그라운드)
        stats = runner.run(duration_sec=0.6)  # 그동안 제어 루프 실행
        assert stats.achieved_hz == pytest.approx(500.0, rel=0.15), stats.summary()

        assert agent.wait_idle(timeout=5.0)
        assert mgr.get_params("impedance")["kp"] == 60.0  # 응답은 나중에 반영됨
    finally:
        agent.stop()


def test_submit_never_blocks():
    _, _, _, agent = make_stack(config=LLMConfig(cooldown_s=0.0))
    # 워커를 안 띄워서 큐가 안 빠짐 — 포화돼도 블로킹 없이 False
    t0 = time.perf_counter()
    results = [agent.submit(stall_event(t=float(i))) for i in range(100)]
    assert time.perf_counter() - t0 < 0.5
    assert False in results
    assert agent.stats["dropped_queue_full"] > 0


# ------------------------------------------------------- 엔드투엔드 통합
def test_end_to_end_stall_detect_recover_on_mock():
    """감지→LLM→검증→갱신→회복. 스모크로 확인한 시나리오의 회귀 테스트.

    관절 0에 -8Nm 지속 외란: kp=25면 정상상태 오차 0.32rad(스톨),
    kp=60이면 0.13rad. LLM 제안이 적용되면 오차가 절반 이하로 줄어야 한다.
    """
    mgr = NodeGraphManager()
    mgr.add_node(TargetNode(params={"depth": 0.4}))
    mgr.add_node(ImpedanceNode(params={"kp": 25.0, "kd": 1.5}))
    mgr.connect("target", "impedance")
    guard = SafetyGuard(mgr, {"impedance.kp": ParamSpec(1.0, 120.0, max_rel_step=2.5)})
    rules = RuleBasedRecovery(mgr)

    def llm(system, user):
        return json.dumps({"diagnosis": "opposing torque; raise stiffness",
                           "actions": [{"node": "impedance", "param": "kp", "value": 60.0}],
                           "confidence": 0.9})

    hal = MockRobotHAL(n_joints=3, dt=1e-3, torque_limit=20.0)
    logger = RingLogger.for_hal(hal, window_sec=3.0)
    runner = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger)
    runner.set_policy(lambda st: mgr.step({"state": st})["impedance"]["command"])

    agent = LLMRecoveryAgent(guard, rules, logger=logger,
                             config=LLMConfig(timeout_s=2.0, cooldown_s=0.1), client=llm)
    detector = FailureDetector(3, DetectorConfig(
        torque_threshold=hal.torque_limits * 0.8,
        stall_err_threshold=0.08, stall_min_duration_s=0.2, refractory_s=0.5))
    supervisor = RecoverySupervisor(detector, agent, logger)

    agent.start()
    try:
        # 벽시계(duration_sec)가 아니라 n_steps 기준 — 시뮬레이션 시간이 결정적이어야
        # 전체 스위트 부하와 무관하게 감지 창이 정상상태 구간에 걸린다.
        runner.run(n_steps=300)
        hal.inject_disturbance(0, -8.0, duration=30.0)
        runner.run(n_steps=700)
        err_before = abs(hal.read_state().q[0] - 0.4)

        events = supervisor.poll()
        assert any(e.type == FailureType.STALL and e.joint_idx == 0 for e in events)
        assert agent.wait_idle(timeout=5.0)
        assert mgr.get_params("impedance")["kp"] == 60.0

        runner.run(n_steps=500)
        err_after = abs(hal.read_state().q[0] - 0.4)
        assert err_after < err_before / 2

        # 타임라인이 로그에 남았는지 (지름길 ① 데모의 근거 자료)
        dump = logger.dump_text()
        assert "DETECTED STALL" in dump
        assert "recovery[llm]" in dump
    finally:
        agent.stop()
