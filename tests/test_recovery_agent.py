"""LLM 회복 에이전트 — 태그 결정, 불량 응답 7종 폴백, 가드, 쿨다운."""

import json
import time

import numpy as np
import pytest

from robot_core.recovery import (
    DEFAULT_TAG_MAP, FailureEvent, FailureType, LLMConfig, LLMRecoveryAgent,
    TagRuleFallback, TagSafetyGuard,
)

TAGS = {"approach", "insert", "retreat", "retry", "rest", "slow"}


def impact_event(t=1.0):
    return FailureEvent(FailureType.IMPACT, 3, 0.8, t=t,
                        snapshot={"dob_peak": 5.0})


def make_agent(client=None, cooldown=0.0, decisions=None, **cfg):
    guard = TagSafetyGuard(TAGS)
    agent = LLMRecoveryAgent(
        guard, TagRuleFallback(), config=LLMConfig(cooldown_s=cooldown,
                                                   timeout_s=0.5, **cfg),
        client=client,
        on_decision=(lambda d, ev: decisions.append(d)) if decisions is not None
        else None)
    return agent, guard


def run(agent, event=None):
    assert agent.submit(event or impact_event())
    assert agent.process_pending() == 1


# ------------------------------------------------------------- 정상 경로
def test_good_llm_response_yields_validated_tag():
    decisions = []
    client = lambda s, u: json.dumps({
        "intent_tag": "retreat", "urgency": "normal",
        "reasoning": "impact detected", "confidence": 0.9})
    agent, guard = make_agent(client, decisions=decisions)
    run(agent)
    d = decisions[0]
    assert (d.intent_tag, d.urgency, d.source) == ("retreat", "normal", "llm")
    assert agent.stats["llm_ok"] == 1
    assert guard.audit[-1].status == "accepted"


def test_llm_only_picks_tags_never_motion_ids():
    """실행 경로 분리: 응답에 motion id 비슷한 게 있어도 태그만 소비된다."""
    decisions = []
    client = lambda s, u: json.dumps({
        "intent_tag": "retry", "urgency": "slow",
        "reasoning": "play motion 7 slowly", "confidence": 0.9,
        "motion_id": 3})            # 무시돼야 함
    agent, _ = make_agent(client, decisions=decisions)
    run(agent)
    assert decisions[0].intent_tag == "retry"
    assert not hasattr(decisions[0], "motion_id")


# -------------------------------------------------- 불량 응답 7종 → 폴백
BAD_RESPONSES = [
    ("not-json", "이건 JSON이 아님"),
    ("json-array", json.dumps([1, 2, 3])),
    ("missing-tag", json.dumps({"urgency": "normal", "confidence": 0.9})),
    ("bad-urgency", json.dumps({"intent_tag": "retreat", "urgency": "PANIC",
                                "reasoning": "", "confidence": 0.9})),
    ("bad-confidence", json.dumps({"intent_tag": "retreat", "urgency": "normal",
                                   "reasoning": "", "confidence": "high"})),
    ("unknown-tag", json.dumps({"intent_tag": "self_destruct", "urgency": "normal",
                                "reasoning": "", "confidence": 0.95})),
    ("low-confidence", json.dumps({"intent_tag": "retreat", "urgency": "normal",
                                   "reasoning": "", "confidence": 0.1})),
]


@pytest.mark.parametrize("label,payload", BAD_RESPONSES,
                         ids=[b[0] for b in BAD_RESPONSES])
def test_bad_llm_responses_fall_back_to_rule_tag(label, payload):
    """불량 응답 7종 전부: 예외 없이 규칙 폴백 태그(IMPACT→retreat)로."""
    decisions = []
    agent, guard = make_agent(lambda s, u: payload, decisions=decisions)
    run(agent)
    d = decisions[0]
    assert d.intent_tag == DEFAULT_TAG_MAP[FailureType.IMPACT][0]   # "retreat"
    assert d.source.startswith("rules:")
    assert agent.stats["llm_fallback"] == 1


def test_llm_timeout_falls_back():
    def slow_client(s, u):
        time.sleep(2.0)
        return "{}"
    decisions = []
    agent, _ = make_agent(slow_client, decisions=decisions)
    run(agent)
    assert decisions[0].source.startswith("rules:")
    assert agent.stats["llm_timeout"] == 1


def test_llm_exception_falls_back():
    def broken(s, u):
        raise RuntimeError("connection reset")
    decisions = []
    agent, _ = make_agent(broken, decisions=decisions)
    run(agent)
    assert decisions[0].source.startswith("rules:")


def test_no_client_always_rules():
    decisions = []
    agent, _ = make_agent(None, decisions=decisions)
    run(agent)
    assert decisions[0].source.startswith("rules:")
    assert decisions[0].intent_tag == "retreat"


def test_unknown_fallback_tag_degrades_to_rest():
    """규칙 매핑이 카탈로그와 어긋나면(미지 태그) 안전 기본값 rest로."""
    decisions = []
    guard = TagSafetyGuard({"rest", "approach"})     # retreat이 없다!
    agent = LLMRecoveryAgent(guard, TagRuleFallback(),
                             config=LLMConfig(cooldown_s=0.0), client=None,
                             on_decision=lambda d, ev: decisions.append(d))
    assert agent.submit(impact_event())
    agent.process_pending()
    assert decisions[0].intent_tag == "rest"
    assert decisions[0].urgency == "stop"
    assert agent.stats["fallback_tag_rejected"] == 1


# ------------------------------------------------------------ 쿨다운/이력
def test_cooldown_drops_repeat_events():
    agent, _ = make_agent(None, cooldown=10.0)
    assert agent.submit(impact_event(t=1.0))
    assert not agent.submit(impact_event(t=1.1))     # 같은 (타입,축)
    assert agent.stats["dropped_cooldown"] == 1
    other = FailureEvent(FailureType.IMPACT, 7, 0.5, t=1.1)
    assert agent.submit(other)                        # 다른 축은 통과


def test_history_included_in_prompt_on_recurrence():
    prompts = []

    def client(s, u):
        prompts.append(u)
        return json.dumps({"intent_tag": "retreat", "urgency": "normal",
                           "reasoning": "again", "confidence": 0.9})
    agent, _ = make_agent(client, decisions=[])
    run(agent)
    run(agent)
    assert "PREVIOUS RECOVERY ATTEMPTS" not in prompts[0]
    assert "PREVIOUS RECOVERY ATTEMPTS" in prompts[1]
    assert "'retreat'" in prompts[1]


def test_prompt_contains_available_tags_and_feedback():
    from robot_core.logging import FeedbackCache
    from robot_core.hal.phorce import MockMotion, MockPhorceHAL, N_AXES
    hal = MockPhorceHAL({1: MockMotion(0.2, lambda t: np.zeros(N_AXES))})
    cache = FeedbackCache()
    hal.watch(cache.push)
    hal.step(300)

    prompts = []

    def client(s, u):
        prompts.append(u)
        return json.dumps({"intent_tag": "rest", "urgency": "stop",
                           "reasoning": "", "confidence": 0.9})
    guard = TagSafetyGuard(TAGS)
    agent = LLMRecoveryAgent(guard, TagRuleFallback(), cache=cache,
                             config=LLMConfig(cooldown_s=0.0), client=client,
                             on_decision=lambda d, ev: None)
    agent.submit(impact_event())
    agent.process_pending()
    assert "AVAILABLE INTENT TAGS" in prompts[0]
    assert "retreat" in prompts[0]
    assert "PHORCE FEEDBACK DUMP" in prompts[0]
