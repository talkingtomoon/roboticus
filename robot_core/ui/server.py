"""운용 웹 UI 서버 — FastAPI + uvicorn, html 한 장 + 1초 폴링.

Jetson에서 돌고 폰/노트북 브라우저로 접속한다 (0.0.0.0).
Supervisor와의 결합은 공개 메서드뿐이다:
snapshot() / halt() / resume() / request_intent()(인터프리터 경유).

    python -m robot_core.ui.server --demo     # 목 월드로 라이브 데모
    # 실물: create_app(supervisor, interpreter, typed_source)을 uvicorn에
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"


class IntentBody(BaseModel):
    text: str


class HaltBody(BaseModel):
    reason: str = "web UI"


def build_inputs_block(snap: dict, interpreter=None) -> dict:
    """'무슨 입력을 받았나' 한 블록. **없는 입력은 없다고 정직하게 말한다.**

    - sensors : 실제로 들어온다 (1kHz 12축 피드백 요약)
    - voice   : 텍스트 입력은 실동작. 마이크는 미연결(WhisperSource 스켈레톤)
    - camera  : 아예 없음 — phorce 인터페이스가 카메라를 주지 않고, 우리도
                붙이지 않았다. UI에 슬롯만 두어 '없음'을 드러낸다
    """
    ax = snap.get("axes") or {}
    wd = snap.get("watchdog") or {}
    fresh = wd.get("fresh_ms")
    sensors = {
        "status": ("live" if fresh is not None
                   and fresh < wd.get("limit_ms", 1500) else "stale"),
        "hz_note": "1kHz · 12축",
        "valid": ax.get("valid"), "total": ax.get("total"),
        "fault": ax.get("fault"),
        "temp_max": ax.get("temp_max"), "temp_max_axis": ax.get("temp_max_axis"),
        "dob_max": ax.get("dob_max"), "dob_max_axis": ax.get("dob_max_axis"),
        "bus_v_min": ax.get("bus_v_min"),
        "fresh_ms": fresh, "seq": ax.get("seq"),
    }
    voice = {"status": "text_only", "mic": "미연결 (WhisperSource 스켈레톤)",
             "last": None}
    results = getattr(interpreter, "results", None) if interpreter else None
    if results:
        d = results[-1].to_dict()
        voice["last"] = {k: d.get(k) for k in
                         ("text", "accepted", "tag", "urgency", "path",
                          "detail", "action")}
    return {
        "sensors": sensors,
        "voice": voice,
        "camera": {"status": "absent",
                   "note": "카메라 입력 없음 — 인터페이스 미제공, 미구현"},
    }


def create_app(supervisor, interpreter=None, typed_source=None) -> FastAPI:
    from robot_core.ui.guidance import compute_guidance

    app = FastAPI(title="robot_core 운용 UI", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/op")
    def op_page():
        """운용자용 간이 페이지 — 시스템을 모르는 팀원용."""
        return FileResponse(STATIC_DIR / "op.html")

    @app.get("/api/guidance")
    def guidance():
        """현재 상태 + 최근 이벤트 → 사람 행동 지시 (번역표는 guidance.py 한 곳)."""
        events = [{"t": e.t, "text": e.text}
                  for e in supervisor.cache.events()]
        return compute_guidance(supervisor.snapshot(), events)

    @app.post("/api/operator_cleared")
    def operator_cleared():
        """[해결했어요] — WAITING_OPERATOR에서만 유효."""
        if supervisor.state.value != "WAITING_OPERATOR":
            return JSONResponse(
                {"error": "지금은 해결할 것이 없어요 (WAITING_OPERATOR 아님)",
                 "state": supervisor.state.value}, status_code=403)
        supervisor.operator_cleared()
        return {"state": supervisor.snapshot()["state"]}

    @app.get("/api/state")
    def state():
        snap = supervisor.snapshot()
        snap["inputs"] = build_inputs_block(snap, interpreter)
        return snap

    @app.get("/api/events")
    def events(since: int = 0):
        evs = supervisor.cache.events_since(since)
        return {"events": [{"seq": e.seq, "t": round(e.t, 3), "text": e.text}
                           for e in evs],
                "cursor": evs[-1].seq if evs else since}

    @app.post("/api/halt")
    def halt(body: HaltBody | None = None):
        supervisor.halt((body.reason if body else "web UI"))
        return {"state": supervisor.snapshot()["state"]}

    @app.post("/api/resume")
    def resume():
        supervisor.resume()
        return {"state": supervisor.snapshot()["state"]}

    @app.post("/api/intent")
    def intent(body: IntentBody):
        if typed_source is None or interpreter is None:
            return JSONResponse({"error": "intent pipeline not attached"},
                                status_code=503)
        if not typed_source.put(body.text):
            return {"action": "rejected", "accepted": False,
                    "detail": "너무 짧은 발화 (min_length)"}
        results = interpreter.process_pending()
        return results[-1].to_dict() if results else \
            {"action": "rejected", "accepted": False, "detail": "no result"}

    return app


# ---------------------------------------------------------------- 데모 모드
def run_demo(host: str = "0.0.0.0", port: int = 8710) -> None:
    """목 월드 + 판단 스레드 + UI. 브라우저에서 상태/타임라인/텍스트 명령."""
    import uvicorn

    from robot_core.integration.scenarios import build_world
    from robot_core.intent import IntentInterpreter, TypedSource
    from robot_core.supervisor import SupervisorConfig

    hal, cache, sup, plan, guard = build_world(
        plan_tags=("approach", "insert", "finish") * 5)   # 길게 — 구경할 시간
    sup.cfg = SupervisorConfig(decision_hz=4.0, sync_recovery=True)
    sup._time = time.monotonic

    stop = threading.Event()

    def sim():                       # 시뮬 시간을 벽시계 근사로 흘린다
        while not stop.is_set():
            hal.step(25)
            time.sleep(0.025)

    threading.Thread(target=sim, daemon=True, name="demo-sim").start()
    hal.step(5)
    sup.start()

    source = TypedSource()
    interp = IntentInterpreter(sup, source=source, client=None)
    app = create_app(sup, interp, source)

    print(f"\n운용 UI: http://localhost:{port}  (폰에서는 http://<Jetson IP>:{port})")
    print("데모 명령 예: '멈춰' '계속해' '천천히 다시 해봐' '잠깐 쉬어' '옆으로 비켜'")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        stop.set()
        sup.stop()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", help="목 월드로 라이브 데모")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8710)
    args = p.parse_args()
    if not args.demo:
        p.error("실물 연결은 create_app()을 직접 쓸 것 — 지금은 --demo만")
    run_demo(args.host, args.port)
