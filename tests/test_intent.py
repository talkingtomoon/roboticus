"""의도 파이프라인 — 안전 키워드 즉시성, 가드 경유, LLM 폴백, 컨텍스트."""

import json
import time

import pytest

from robot_core.integration.scenarios import build_world, run_ticks
from robot_core.intent import IntentInterpreter, TypedSource
from robot_core.supervisor import SupervisorState


def make(client=None):
    hal, cache, sup, plan, guard = build_world()
    source = TypedSource()
    interp = IntentInterpreter(sup, source=source, client=client)
    return hal, cache, sup, plan, guard, source, interp


# ------------------------------------------------------- 안전 키워드 즉시성
def test_halt_keyword_bypasses_llm_immediately():
    """"멈춰"는 LLM을 기다리지 않는다 — 안전 명령에 3초 지연 금지."""
    calls = []

    def slow_client(s, u):
        calls.append(1)
        time.sleep(2.0)
        return "{}"

    hal, cache, sup, plan, _, _, interp = make(client=slow_client)
    run_ticks(hal, sup, 2)

    t0 = time.perf_counter()
    r = interp.handle_text("로봇 멈춰!")
    elapsed = time.perf_counter() - t0

    assert r.action == "halt" and r.path == "safety"
    assert elapsed < 0.5, f"halt가 {elapsed:.2f}s 걸림 — LLM을 기다렸다"
    assert calls == [], "안전 명령이 LLM을 호출했다"
    assert sup.state == SupervisorState.HALTED


def test_resume_keyword():
    hal, cache, sup, plan, _, _, interp = make()
    run_ticks(hal, sup, 2)
    sup.halt("test")
    r = interp.handle_text("계속해")
    assert r.action == "resume"
    assert sup.state == SupervisorState.DECIDING


# --------------------------------------------------------- 가드 경유 보장
def test_stt_mishear_rejected_by_guard():
    """"저지 동작 해줘" (정지 오인식 유사어): 미지 태그는 어느 경로로 오든
    가드에서 거부돼야 한다."""
    def mishear_client(s, u):
        return json.dumps({"intent_tag": "저지", "urgency": "normal",
                           "reasoning": "mishear", "confidence": 0.9})

    hal, cache, sup, plan, guard, _, interp = make(client=mishear_client)
    run_ticks(hal, sup, 2)
    r = interp.handle_text("저지 동작 해줘")
    assert r.action == "rejected"
    assert not r.accepted
    rejected = [a for a in guard.audit if a.status == "rejected"]
    assert rejected and rejected[-1].requested == "저지"
    assert sup.state != SupervisorState.HALTED     # 절대 정지로 오인 금지


def test_keyword_fallback_also_goes_through_guard():
    hal, cache, sup, plan, guard, _, interp = make()   # LLM 없음
    run_ticks(hal, sup, 2)
    r = interp.handle_text("잠깐 쉬어")
    assert (r.action, r.tag, r.path) == ("intent", "rest", "rules")
    assert guard.audit[-1].status == "accepted"
    assert guard.audit[-1].source == "voice-rules"


# ------------------------------------------------------- LLM 죽어도 동작
@pytest.mark.parametrize("text,tag,urgency", [
    ("잠깐 쉬어", "rest", "stop"),
    ("옆으로 비켜", "retreat", "normal"),
    ("천천히 다시 해봐", "retry", "slow"),
    ("다시 해봐", "retry", "normal"),
    ("천천히 가", "retry", "slow"),
])
def test_keyword_table_without_llm(text, tag, urgency):
    hal, cache, sup, plan, _, _, interp = make()
    run_ticks(hal, sup, 2)
    r = interp.handle_text(text)
    assert (r.tag, r.urgency, r.path) == (tag, urgency, "rules"), r.to_dict()
    assert r.accepted


def test_unknown_utterance_rejected_not_crashed():
    hal, cache, sup, plan, _, _, interp = make()
    r = interp.handle_text("오늘 날씨 어때")
    assert r.action == "rejected" and "해석 불가" in r.detail


def test_llm_timeout_falls_to_keywords():
    def dead_client(s, u):
        time.sleep(5.0)
        return "{}"
    hal, cache, sup, plan, _, _, interp = make(client=dead_client)
    interp.config.timeout_s = 0.2
    run_ticks(hal, sup, 2)
    r = interp.handle_text("천천히 다시")
    assert (r.tag, r.path) == ("retry", "rules")


# ------------------------------------------------------------- 컨텍스트
def test_context_failure_makes_retry_interpretable():
    """실패 이벤트가 있을 때 "다시 해봐" — 프롬프트에 실패가 실리고
    LLM(목)이 retry로 해석한다."""
    prompts = []

    def ctx_client(s, u):
        prompts.append(u)
        assert "RECENT FAILURES" in u and "IMPACT" in u
        return json.dumps({"intent_tag": "retry", "urgency": "slow",
                           "reasoning": "impact then retry slowly",
                           "confidence": 0.9})

    hal, cache, sup, plan, guard, _, interp = make(client=ctx_client)
    run_ticks(hal, sup, 3)
    hal.inject_disturbance(0, 5.0, duration=0.3)
    run_ticks(hal, sup, 3)                          # IMPACT 감지됨

    r = interp.handle_text("다시 해봐")
    assert (r.tag, r.urgency, r.path) == ("retry", "slow", "llm")
    assert r.accepted
    assert "AVAILABLE INTENT TAGS" in prompts[0]
    assert f"next tag = " in prompts[0]              # 계획 단계 포함


# ------------------------------------------------------------- 소스/워커
def test_typed_source_min_length_filter():
    src = TypedSource()
    assert not src.put("아")                         # min_length=2 미달
    assert src.put("멈춰")
    assert src.get() == "멈춰"
    assert src.get() is None
    assert src.dropped_short == 1


def test_process_pending_drains_queue():
    hal, cache, sup, plan, _, source, interp = make()
    run_ticks(hal, sup, 2)
    source.put("잠깐 쉬어")
    source.put("멈춰")
    results = interp.process_pending()
    assert [r.action for r in results] == ["intent", "halt"]
    assert sup.state == SupervisorState.HALTED


def test_whisper_source_unavailable_without_dependency():
    from robot_core.intent import IntentSourceUnavailable, WhisperSource
    try:
        import faster_whisper  # noqa: F401
        pytest.skip("faster-whisper가 설치돼 있음 (Jetson 환경)")
    except ImportError:
        pass
    with pytest.raises(IntentSourceUnavailable, match="faster-whisper"):
        WhisperSource()
