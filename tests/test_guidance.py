"""B5 안내 번역표 + /op API — 상태·거절코드 → 사람 행동 지시."""

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="ui extra 미설치")
from fastapi.testclient import TestClient  # noqa: E402

from robot_core.integration.scenarios import build_world, run_ticks
from robot_core.intent import IntentInterpreter, TypedSource
from robot_core.ui.guidance import REJECTION_STEPS, compute_guidance
from robot_core.ui.server import create_app


def snap(state="DECIDING", **kw):
    base = {"state": state, "halt_reason": None, "plan": {}, "breaker": {},
            "hold_rest": False, "last_arrival": None}
    base.update(kw)
    return base


def ev(*texts):
    return [{"t": float(i), "text": t} for i, t in enumerate(texts)]


# --------------------------------------------------------- 번역표 단위 검증
def test_state_lines_no_system_jargon():
    for state, expect in [("PLAYING", "물러서"), ("DECIDING", "고르는 중"),
                          ("DONE", "마쳤어요")]:
        g = compute_guidance(snap(state), [])
        assert expect in g["headline"]
        for word in ("PLAYING", "DECIDING", "Supervisor", "tick"):
            assert word not in g["headline"]      # 시스템 용어 금지


def test_rejection_12_gives_button_procedure():
    g = compute_guidance(snap("WAITING_OPERATOR"),
                         ev("operator required (code 12) — WAITING_OPERATOR"))
    assert g["show_resolve"] is True
    assert g["severity"] == "action"
    assert g["steps"] == REJECTION_STEPS[12]["steps"]
    assert any("1번 버튼" in s and "0.6초" in s for s in g["steps"])


def test_rejection_13_gives_parking_procedure():
    g = compute_guidance(snap("WAITING_OPERATOR"),
                         ev("operator required (code 13) — WAITING_OPERATOR"))
    assert any("2번 버튼" in s and "파킹" in s for s in g["steps"])


def test_unknown_rejection_code_falls_back_to_operator_call():
    g = compute_guidance(snap("WAITING_OPERATOR"),
                         ev("operator required (code 99) — WAITING_OPERATOR"))
    assert any("운영 담당자" in s for s in g["steps"])
    assert g["code"] == 99


def test_watchdog_halt_translation():
    g = compute_guidance(snap("HALTED",
                              halt_reason="feedback stale 2100 ms "
                                          "(> kStateFreshLimitMs=1500)"), [])
    assert "연결이 끊겼어요" in g["headline"]
    assert "케이블" in g["detail"]
    assert g["severity"] == "danger"


def test_breaker_halt_translation():
    g = compute_guidance(snap("HALTED",
                              halt_reason="circuit breaker: IMPACT ax0 x5/60s"), [])
    assert "반복" in g["headline"]
    assert any("[재개]" in s for s in g["steps"])


def test_breaker_incomplete_halt_has_specific_wording():
    g = compute_guidance(snap("HALTED", halt_reason=(
        "circuit breaker: MOTION_INCOMPLETE ax3 x5/60s")), [])
    assert "목표에 계속 못 닿아서" in g["headline"]


def test_estop_translation_power_cycle():
    g = compute_guidance(snap(), ev(
        "preflight FAILED: E-stop이 눌린 상태다 — 해제 후 시작할 것"))
    assert "되돌려도 안" in g["detail"]           # E-stop은 되돌려도 안 풀림
    assert any("전원" in s for s in g["steps"])


def test_facade_missing_translation():
    g = compute_guidance(snap(), ev(
        "preflight FAILED: status() 조회 실패: PhorceError: phorce 파사드를 찾을 수 없다"))
    assert "phorce" in g["detail"]
    assert any("phorce doctor" in s for s in g["steps"])


def test_overheat_hold_is_info_only():
    g = compute_guidance(snap(hold_rest=True), [])
    assert "뜨거워서" in g["headline"]
    assert g["steps"] == [] and g["severity"] == "info"   # 행동 불필요


def test_single_incomplete_not_shown_repeated_is():
    """단발 미도달은 /op에 안 뜬다 — 반복(브레이커 근접)일 때만 안내."""
    single = compute_guidance(snap(breaker={"MOTION_INCOMPLETE:ax3": 1}), [])
    assert "못 닿고" not in single["headline"]    # 평상시 문구 유지
    repeated = compute_guidance(snap(breaker={"MOTION_INCOMPLETE:ax3": 2}), [])
    assert "목표에 잘 못 닿고" in repeated["headline"]
    assert repeated["severity"] == "action"


def test_facade_detail_shown_verbatim_when_present():
    """파사드 MotionRejected.detail(한국어)은 **그대로 노출** — 번역표
    detail은 폴백. 절차 steps는 우리 것이 더 구체적이므로 유지."""
    g = compute_guidance(
        snap("WAITING_OPERATOR",
             last_rejection={"code": 12, "category": "operator",
                             "detail": "영점 조정이 필요합니다. 1번 버튼을 누르세요."}),
        ev("operator required (code 12) — WAITING_OPERATOR"))
    assert g["detail"] == "영점 조정이 필요합니다. 1번 버튼을 누르세요."
    assert g["steps"] == REJECTION_STEPS[12]["steps"]     # 절차는 번역표 유지

    # detail 없으면 번역표 폴백
    g2 = compute_guidance(snap("WAITING_OPERATOR"),
                          ev("operator required (code 12)"))
    assert g2["detail"] == REJECTION_STEPS[12]["detail"]


def test_hardware_rejection_halt_translation():
    g = compute_guidance(
        snap("HALTED",
             halt_reason="play rejected code 6 (hardware): EtherCAT 끊김",
             last_rejection={"code": 6, "category": "hardware",
                             "detail": "EtherCAT 통신이 끊겼습니다."}),
        [])
    assert "거절" in g["headline"]
    assert g["detail"] == "EtherCAT 통신이 끊겼습니다."   # detail 그대로
    assert any("터미널1" in s or "phorce_monitor" in s for s in g["steps"])
    assert g["severity"] == "danger"


def test_error_17_after_button_recovery_is_info():
    """알려진 현상: 버튼 복구 직후 첫 재생 error=17 중단 — 자동 재시도되니
    info로만 안내한다 (사람이 할 일 없음)."""
    g = compute_guidance(snap("DECIDING"), ev(
        "play ABORTED: motion 3 aborted: CANCELED: error=17 — robot at arbitrary pose"))
    assert "멈췄어요" in g["headline"]
    assert "17" in g["detail"] and "기다려" in g["detail"]
    assert g["severity"] == "info" and g["steps"] == []


def test_unknown_state_fallback_shows_code():
    g = compute_guidance(snap("WOBBLING"), [])
    assert "운영 담당자" in g["headline"]
    assert g["code"] == "WOBBLING"


# ----------------------------------------------------------- /op + API 통합
def make_ui():
    hal, cache, sup, plan, guard = build_world()
    source = TypedSource()
    interp = IntentInterpreter(sup, source=source, client=None)
    return hal, cache, sup, plan, TestClient(create_app(sup, interp, source))


def test_op_page_served_independently():
    *_, web = make_ui()
    r = web.get("/op")
    assert r.status_code == 200
    assert "해결했어요" in r.text and "멈춰" in r.text
    assert "모식도" not in r.text                 # 대시보드 요소 없음 (독립)


def test_guidance_endpoint_reflects_live_state():
    hal, cache, sup, plan, web = make_ui()
    run_ticks(hal, sup, 2)
    g = web.get("/api/guidance").json()
    assert g["headline"]
    sup.halt("demo")
    g = web.get("/api/guidance").json()
    assert g["severity"] == "danger"


def test_operator_cleared_403_outside_waiting():
    hal, cache, sup, plan, web = make_ui()
    run_ticks(hal, sup, 2)
    r = web.post("/api/operator_cleared")
    assert r.status_code == 403                   # WAITING_OPERATOR 아님

    hal.set_rejection(12)
    run_ticks(hal, sup, 4)
    assert sup.state.value == "WAITING_OPERATOR"
    g = web.get("/api/guidance").json()
    assert g["show_resolve"] and any("1번 버튼" in s for s in g["steps"])

    hal.clear_rejection()
    r = web.post("/api/operator_cleared")
    assert r.status_code == 200
    assert r.json()["state"] == "DECIDING"


def test_incomplete_event_visible_in_events_and_badge():
    hal, cache, sup, plan, web = make_ui()

    hal.inject_jam(0)
    run_ticks(hal, sup, 5)
    texts = [e["text"] for e in web.get("/api/events?since=0").json()["events"]]
    assert any("DETECTED MOTION_INCOMPLETE" in t for t in texts)
    st = web.get("/api/state").json()
    assert st["last_arrival"]["status"] == "incomplete"
    assert st["breaker"]                          # 미도달이 브레이커에 집계됨
