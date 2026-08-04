"""안내 번역표 — 상태·이벤트·거절코드 → 사람 행동 지시.

phorce 매뉴얼 지식(버튼 절차, E-stop 특성)을 코드화한 **유일한** 장소다.
프런트(/op)는 문자열을 하드코딩하지 않는다 — 전부 /api/guidance에서 받는다.
번역표에 없는 상태/코드는 "운영 담당자를 불러주세요" + 원본 코드 표시.

severity: "info"(구경만) | "action"(사람이 할 일 있음) | "danger"(정지 상태)
"""

from __future__ import annotations

import re

# ------------------------------------------------------------ 거절코드 절차
REJECTION_STEPS = {
    12: {   # NOT_READY — 영점
        "headline": "로봇이 사람의 도움을 기다려요.",
        "detail": "영점(원점) 잡기가 필요해요 (코드 12).",
        "steps": [
            "로봇 주변에 사람과 물건이 없는지 확인하세요.",
            "기체의 1번 버튼을 0.6초 이상 꾹 누르세요. 로봇이 움직입니다.",
            "3초 기다린 뒤, 아래 [해결했어요] 버튼을 누르세요.",
        ],
    },
    13: {   # RECOVERY_REQUIRED — 복구
        "headline": "로봇이 사람의 도움을 기다려요.",
        "detail": "복구 절차가 필요해요 (코드 13).",
        "steps": [
            "로봇 주변을 비우세요.",
            "2번 버튼으로 로봇을 파킹시키세요.",
            "그다음 1번 버튼을 0.6초 이상 꾹 누르세요.",
            "로봇이 자리를 잡으면 [해결했어요]를 누르세요.",
        ],
    },
}

UNKNOWN_OPERATOR = {
    "headline": "로봇이 사람의 도움을 기다려요.",
    "detail": "알 수 없는 사유예요.",
    "steps": ["운영 담당자를 불러주세요."],
}

STATE_LINES = {
    "PLAYING": ("로봇이 움직이고 있어요. 물러서 계세요.", "info"),
    "DECIDING": ("다음 동작을 고르는 중이에요.", "info"),
    "IDLE": ("다음 동작을 고르는 중이에요.", "info"),
    "DONE": ("계획한 작업을 마쳤어요.", "info"),
}


def _recent(events: list, needle: str, n: int = 40) -> str | None:
    """최근 이벤트에서 needle을 포함하는 마지막 텍스트."""
    for e in reversed(events[-n:]):
        text = e["text"] if isinstance(e, dict) else e.text
        if needle in text:
            return text
    return None


def compute_guidance(snapshot: dict, events: list) -> dict:
    """현재 상태 + 최근 이벤트 → 사람용 안내. 시스템 용어 금지.

    반환: {headline, detail, steps[], show_resolve, severity, code?}
    """
    state = snapshot.get("state", "?")
    out = {"headline": "", "detail": "", "steps": [], "show_resolve": False,
           "severity": "info", "state": state}

    # ---------------- HALTED: 사유별 번역 ----------------
    if state == "HALTED":
        reason = snapshot.get("halt_reason") or ""
        out["severity"] = "danger"
        out["headline"] = "정지 상태예요."
        if "stale" in reason or "kStateFreshLimitMs" in reason:
            out["headline"] = "로봇 연결이 끊겼어요."
            out["detail"] = "케이블과 전원을 확인하세요."
            out["steps"] = ["젯슨과 로봇 사이 케이블을 확인하세요.",
                            "로봇 전원 표시등을 확인하세요.",
                            "해결되면 [재개]를 누르세요."]
        elif "circuit breaker" in reason:
            if "MOTION_INCOMPLETE" in reason:
                out["headline"] = "로봇이 목표에 계속 못 닿아서 멈췄어요."
                out["detail"] = "물건 위치가 바뀌었거나 뭔가 걸리고 있을 수 있어요."
                out["steps"] = ["로봇 주변과 작업 대상의 위치를 확인하세요.",
                                "걸리는 물건이 있으면 치우세요.",
                                "확인 후 [재개]를 누르세요."]
            else:
                out["headline"] = "같은 문제가 반복돼서 멈췄어요."
                out["detail"] = "로봇 주변에 걸리는 물건이 있는지 확인해 주세요."
                out["steps"] = ["로봇 주변에 걸리는 물건이 있는지 확인하세요.",
                                "확인 후 [재개]를 누르세요."]
        elif any(w in reason for w in ("voice", "web UI", "정지 버튼",
                                       "멈춰 버튼", "operator halt")):
            out["detail"] = "사람이 정지시켰어요. 준비되면 [재개]를 누르세요."
        else:
            out["detail"] = f"정지 사유: {reason}" if reason else \
                "운영 담당자를 불러주세요."
            out["code"] = reason
        return out

    # ---------------- WAITING_OPERATOR: 거절코드별 절차 ----------------
    if state == "WAITING_OPERATOR":
        out["severity"] = "action"
        out["show_resolve"] = True
        text = _recent(events, "operator required")
        code = None
        if text:
            m = re.search(r"code (\d+)", text)
            code = int(m.group(1)) if m else None
        entry = REJECTION_STEPS.get(code, UNKNOWN_OPERATOR)
        out.update({k: entry[k] for k in ("headline", "detail", "steps")})
        if code is not None and code not in REJECTION_STEPS:
            out["detail"] = f"알 수 없는 거절 코드예요 (코드 {code})."
            out["code"] = code
        return out

    # ---------------- preflight/E-stop/파사드 (이벤트 기반) ----------------
    pf = _recent(events, "preflight FAILED")
    if pf:
        out["severity"] = "action"
        out["headline"] = "시작 전 점검에서 문제가 나왔어요."
        if "E-stop" in pf:
            out["detail"] = ("비상정지가 눌려 있어요. E-stop은 되돌려도 안 "
                             "풀립니다 — 로봇 전원을 껐다 켜야 해요.")
            out["steps"] = ["E-stop 버튼을 되돌리세요.",
                            "로봇 전원을 껐다 켜세요.",
                            "운영 담당자에게 재시작을 요청하세요."]
        elif "phorce" in pf or "파사드" in pf or "status() 조회 실패" in pf:
            out["detail"] = ("로봇 소프트웨어(phorce)를 찾을 수 없어요.")
            out["steps"] = ["젯슨에서 `phorce doctor`를 실행해 보세요.",
                            "운영 담당자를 불러주세요."]
        else:
            out["detail"] = pf.replace("preflight FAILED: ", "")
            out["steps"] = ["운영 담당자를 불러주세요."]
        return out

    # ---------------- 정상 상태들 ----------------
    if snapshot.get("hold_rest"):
        out["headline"] = "로봇이 뜨거워서 쉬고 있어요."
        out["detail"] = "기다리면 알아서 계속해요."
        return out

    # MOTION_INCOMPLETE 반복 (브레이커 근접) — 단발은 표시하지 않는다:
    # 시스템이 알아서 재선곡할 일이지 사람이 개입할 일이 아니다
    breaker = snapshot.get("breaker") or {}
    inc = {k: v for k, v in breaker.items() if k.startswith("MOTION_INCOMPLETE")}
    if inc and max(inc.values()) >= 2:
        out["severity"] = "action"
        out["headline"] = "로봇이 동작은 하는데 목표에 잘 못 닿고 있어요."
        out["detail"] = "물건 위치가 바뀌었거나 뭔가 걸리는지 확인해 주세요."
        out["steps"] = ["작업 대상의 위치가 원래 자리인지 확인하세요.",
                        "로봇 경로에 걸리는 것이 있으면 치우세요."]
        return out

    if state in STATE_LINES:
        out["headline"], out["severity"] = STATE_LINES[state]
        if state == "PLAYING" and snapshot.get("current_motion"):
            out["detail"] = f"동작: {snapshot['current_motion']['name']}"
        return out

    # ---------------- 미지 상태 폴백 ----------------
    out["severity"] = "action"
    out["headline"] = "운영 담당자를 불러주세요."
    out["detail"] = f"알 수 없는 상태예요 (코드: {state})."
    out["code"] = state
    return out
