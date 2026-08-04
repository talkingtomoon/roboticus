"""운용 웹 UI — 엔드포인트 스모크 + Supervisor 직결 + 완료 기준 통합."""

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="ui extra 미설치")
from fastapi.testclient import TestClient  # noqa: E402

from robot_core.integration.scenarios import TICK_STEPS, build_world, run_ticks
from robot_core.intent import IntentInterpreter, TypedSource
from robot_core.supervisor import SupervisorState
from robot_core.ui.server import create_app


def make_ui(client_llm=None):
    hal, cache, sup, plan, guard = build_world()
    source = TypedSource()
    interp = IntentInterpreter(sup, source=source, client=client_llm)
    app = create_app(sup, interp, source)
    return hal, cache, sup, plan, TestClient(app)


# ------------------------------------------------------------ 스모크
def test_index_serves_single_page():
    *_, web = make_ui()
    r = web.get("/")
    assert r.status_code == 200
    assert "모식도" in r.text or "타임라인" in r.text
    assert "svg" in r.text.lower()


def test_state_endpoint_shape():
    hal, cache, sup, plan, web = make_ui()
    run_ticks(hal, sup, 3)
    st = web.get("/api/state").json()
    assert st["state"] in [s.value for s in SupervisorState]
    assert {"cursor", "total", "next_tag"} <= st["plan"].keys()
    assert st["axes"]["total"] == 12
    assert st["watchdog"]["limit_ms"] == 1500


def test_events_cursor_pagination():
    hal, cache, sup, plan, web = make_ui()
    run_ticks(hal, sup, 3)
    first = web.get("/api/events?since=0").json()
    assert first["events"] and first["cursor"] > 0
    again = web.get(f"/api/events?since={first['cursor']}").json()
    assert again["events"] == []            # 커서 이후 새 이벤트 없음
    run_ticks(hal, sup, 2)
    more = web.get(f"/api/events?since={first['cursor']}").json()
    assert more["events"]                   # 새로 생긴 것만


def test_events_include_selection_score_table():
    hal, cache, sup, plan, web = make_ui()
    run_ticks(hal, sup, 3)
    texts = [e["text"] for e in web.get("/api/events?since=0").json()["events"]]
    assert any("[SELECT]" in t for t in texts)
    assert any("<-- chosen" in t for t in texts)


# ------------------------------------------------------- Supervisor 직결
def test_halt_endpoint_calls_supervisor():
    hal, cache, sup, plan, web = make_ui()
    run_ticks(hal, sup, 2)
    r = web.post("/api/halt", json={"reason": "test button"})
    assert r.json()["state"] == "HALTED"
    assert sup.state == SupervisorState.HALTED
    assert sup.halt_reason == "test button"

    r = web.post("/api/resume")
    assert r.json()["state"] == "DECIDING"
    assert sup.state == SupervisorState.DECIDING


def test_intent_endpoint_guard_rejection_in_response():
    """가드 거부가 API 응답에 그대로 반영돼야 한다 (B3 요구)."""
    def mishear(s, u):
        import json
        return json.dumps({"intent_tag": "저지", "urgency": "normal",
                           "reasoning": "", "confidence": 0.9})
    hal, cache, sup, plan, web = make_ui(client_llm=mishear)
    run_ticks(hal, sup, 2)
    r = web.post("/api/intent", json={"text": "저지 동작 해줘"}).json()
    assert r["accepted"] is False
    assert r["action"] == "rejected"


def test_intent_endpoint_halt_keyword():
    hal, cache, sup, plan, web = make_ui()
    run_ticks(hal, sup, 2)
    r = web.post("/api/intent", json={"text": "멈춰"}).json()
    assert r["action"] == "halt" and r["accepted"]
    assert sup.state == SupervisorState.HALTED


def test_intent_endpoint_too_short():
    hal, cache, sup, plan, web = make_ui()
    r = web.post("/api/intent", json={"text": "아"}).json()
    assert r["accepted"] is False and "짧은" in r["detail"]


# --------------------------------------------------- 완료 기준 통합 시나리오
def test_full_loop_slow_retry_visible_in_ui_timeline():
    """"천천히 다시 해봐" → 가드 통과 → 다음 경계에서 slow 변주 선곡 —
    전 과정이 UI 타임라인(/api/events)에 나타난다."""
    hal, cache, sup, plan, web = make_ui()          # LLM 없음 → 키워드 경로
    run_ticks(hal, sup, 4)                          # approach 재생/완료 부근

    r = web.post("/api/intent", json={"text": "천천히 다시 해봐"}).json()
    assert r["accepted"] and (r["tag"], r["urgency"]) == ("retry", "slow")

    run_ticks(hal, sup, 8)                          # 다음 경계 → 선곡 → 재생

    texts = [e["text"] for e in web.get("/api/events?since=0").json()["events"]]
    # 1) 해석 이벤트
    assert any(t.startswith("intent[rules]") and "retry" in t for t in texts)
    # 2) 가드 통과 (request_intent 경유)
    assert any("intent accepted: 'retry'" in t for t in texts)
    # 3) slow modifier 선곡
    assert any("mod=['slow']" in t for t in texts)
    # 4) slow 변주 재생 (insert_slow=7 또는 approach_slow=8)
    assert any("play: motion 7" in t or "play: motion 8" in t for t in texts), \
        "\n".join(t for t in texts if t.startswith(("play", "select:")))
    # 상태 API에도 진행이 반영
    st = web.get("/api/state").json()
    assert st["plan"]["cursor"] >= 1
