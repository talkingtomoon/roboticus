"""의도 해석기 — 발화 텍스트 → (안전 명령 | 의도 태그).

경로 3단:
1) 안전 키워드 (LLM 안 기다림 — 안전 명령에 3초 지연 금지):
   "멈춰/정지" → Supervisor.halt() 즉시. "재개/계속" → resume().
   halt/resume은 태그가 아니라 Supervisor 메서드 직결이다.
2) LLM: 발화 + 컨텍스트(최근 실패, 계획 단계, 가용 태그) → 기존 회복
   하네스와 동일 스키마 {"intent_tag","urgency","reasoning","confidence"}.
   호출/파싱은 llm_agent의 것을 재사용한다 (중복 구현 금지).
3) 규칙 폴백 (LLM 죽어도 기본 명령 동작): 키워드 표 → 태그.

태그 주입은 **전부** Supervisor.request_intent()로만 — 진입점이 어디든
TagSafetyGuard를 지난다 (STT 오인식 "저지" 같은 미지 태그 방어).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from robot_core.intent.sources import IntentSource
from robot_core.recovery.llm_agent import (
    LLMConfig, LLMRecoveryAgent, LLMUnavailable, call_client_with_timeout,
)

INTENT_SYSTEM_PROMPT = """You are the voice-intent interpreter of a motion-playback robot.
A human said something. Map it to an INTENT TAG describing what kind of motion
should play next. You cannot command trajectories; a deterministic selector maps
your tag to a motion id.

Respond with a SINGLE JSON object and NOTHING else.
- No markdown, no code fences, no backticks.
- Schema (exactly these keys):
{"intent_tag": "<one tag from AVAILABLE INTENT TAGS>",
 "urgency": "normal" | "slow" | "stop",
 "reasoning": "<one short sentence>",
 "confidence": <number 0..1>}

Rules:
- intent_tag MUST be one of the AVAILABLE INTENT TAGS. Anything else is rejected.
- The utterance is Korean, possibly from speech recognition (may contain errors).
- "천천히" means slow, "다시" means retry, "쉬어" means rest, "물러나/비켜" retreat.
- If RECENT FAILURES are listed and the human says "다시 해봐", they mean retry.
- If you cannot map it to an available tag, pick "rest" with low confidence."""

# 안전 명령 — LLM을 기다리지 않는다 (부분 문자열 매칭)
HALT_WORDS = ("멈춰", "멈춤", "정지", "그만", "스톱", "stop")
RESUME_WORDS = ("재개", "다시 시작", "계속해", "계속 해", "resume")

# 규칙 폴백: 키워드 → (태그, 긴급도). 위에서부터 첫 매칭.
DEFAULT_KEYWORD_MAP: list[tuple[tuple[str, ...], tuple[str, str]]] = [
    (("쉬어", "휴식", "쉬자"), ("rest", "stop")),
    (("물러나", "물러서", "비켜", "피해"), ("retreat", "normal")),
    (("천천히 다시", "살살 다시"), ("retry", "slow")),
    (("다시", "재시도", "한번 더", "한 번 더"), ("retry", "normal")),
    (("천천히", "살살", "느리게"), ("retry", "slow")),
]


@dataclass
class IntentResult:
    action: str            # "halt" | "resume" | "intent" | "rejected"
    text: str
    tag: str | None = None
    urgency: str | None = None
    path: str = ""         # "safety" | "llm" | "rules"
    detail: str = ""
    accepted: bool = False
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"action": self.action, "text": self.text, "tag": self.tag,
                "urgency": self.urgency, "path": self.path,
                "detail": self.detail, "accepted": self.accepted,
                "elapsed_ms": round(self.elapsed_ms, 1)}


class IntentInterpreter:
    """발화 → 해석 → Supervisor로. 소스에서 당겨오는 워커도 내장."""

    def __init__(self, supervisor, source: IntentSource | None = None,
                 client=None, config: LLMConfig | None = None,
                 keyword_map=None, on_log=None) -> None:
        self.sup = supervisor
        self.source = source
        self.client = client
        self.config = config or LLMConfig()
        self.keyword_map = keyword_map or DEFAULT_KEYWORD_MAP
        self._on_log = on_log
        self.results: list[IntentResult] = []     # 최근 해석 이력 (UI가 본다)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- 해석
    def handle_text(self, text: str) -> IntentResult:
        t0 = time.perf_counter()
        text = str(text).strip()
        result = self._interpret(text)
        result.elapsed_ms = (time.perf_counter() - t0) * 1e3
        self.results.append(result)
        if len(self.results) > 100:
            del self.results[:50]
        self._mark(result)
        return result

    def _interpret(self, text: str) -> IntentResult:
        low = text.lower()

        # ---- 1) 안전 키워드: LLM 우회, 즉시 실행 ----
        if any(w in low for w in HALT_WORDS):
            self.sup.halt(f"voice: {text!r}")
            return IntentResult("halt", text, path="safety",
                                detail="halt keyword — LLM 우회 즉시 정지",
                                accepted=True)
        if any(w in low for w in RESUME_WORDS):
            self.sup.resume()
            return IntentResult("resume", text, path="safety",
                                detail="resume keyword", accepted=True)

        # ---- 2) LLM 경로 ----
        if self.client is not None:
            try:
                raw = call_client_with_timeout(
                    self.client, INTENT_SYSTEM_PROMPT,
                    self._build_prompt(text), self.config.timeout_s)
                tag, urgency, reasoning, conf = \
                    LLMRecoveryAgent._parse_response(raw)
                if conf < self.config.min_confidence:
                    raise LLMUnavailable(f"low confidence {conf:.2f}")
                accepted = self.sup.request_intent(tag, urgency,
                                                   source="voice-llm")
                return IntentResult(
                    "intent" if accepted else "rejected", text, tag=tag,
                    urgency=urgency, path="llm", detail=reasoning,
                    accepted=accepted)
            except LLMUnavailable as e:
                self._log(f"intent LLM fallback: {e}")

        # ---- 3) 규칙 폴백: 키워드 표 ----
        for words, (tag, urgency) in self.keyword_map:
            if any(w in low for w in words):
                accepted = self.sup.request_intent(tag, urgency,
                                                   source="voice-rules")
                return IntentResult(
                    "intent" if accepted else "rejected", text, tag=tag,
                    urgency=urgency, path="rules",
                    detail=f"keyword {words[0]!r}", accepted=accepted)

        return IntentResult("rejected", text, path="rules",
                            detail="해석 불가 — 아는 키워드/태그 없음")

    # ------------------------------------------------------------ 컨텍스트
    def _build_prompt(self, text: str) -> str:
        parts = [f'UTTERANCE (Korean, may contain STT errors): "{text}"', ""]

        guard = getattr(self.sup, "guard", None)
        if guard is not None:
            parts += ["AVAILABLE INTENT TAGS (the ONLY valid values)",
                      guard.describe_whitelist(), ""]

        plan = getattr(self.sup, "plan", None)
        if plan is not None:
            parts.append(f"PLAN: step {plan.cursor}/{len(plan.tag_sequence)}, "
                         f"next tag = {plan.next_tag!r}")

        cache = getattr(self.sup, "cache", None)
        if cache is not None:
            fails = [e.text for e in cache.events()
                     if e.text.startswith("DETECTED")][-3:]
            if fails:
                parts.append("RECENT FAILURES:")
                parts += [f"  {f}" for f in fails]
        parts.append("")
        parts.append("Respond with the JSON object now.")
        return "\n".join(parts)

    # ------------------------------------------------------------- 워커
    def start(self) -> None:
        """소스에서 발화를 당겨 처리하는 워커 (라이브 UI/STT용)."""
        if self.source is None:
            raise RuntimeError("source가 없다 — handle_text()를 직접 쓰거나 "
                               "TypedSource를 연결할 것")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                text = self.source.get(timeout=0.2)
                if text:
                    try:
                        self.handle_text(text)
                    except Exception as e:   # 해석 실패가 워커를 죽이면 안 됨
                        self._log(f"intent worker error (ignored): {e}")

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="intent-worker")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def process_pending(self) -> list[IntentResult]:
        """소스 큐를 지금 이 스레드에서 비운다 (동기 UI/테스트용)."""
        out = []
        while self.source is not None:
            text = self.source.get()
            if text is None:
                break
            out.append(self.handle_text(text))
        return out

    # ------------------------------------------------------------- 로그
    def _mark(self, r: IntentResult) -> None:
        cache = getattr(self.sup, "cache", None)
        if cache is not None:
            label = {"halt": "HALT", "resume": "RESUME",
                     "intent": f"{r.tag!r}/{r.urgency}",
                     "rejected": "거부"}[r.action]
            cache.mark_event(
                f"intent[{r.path}] {r.text!r} -> {label}"
                + (f" ({r.detail})" if r.detail else ""))

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(msg)
            except Exception:
                pass
