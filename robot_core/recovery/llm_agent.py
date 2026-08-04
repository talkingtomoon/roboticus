"""LLM 회복 에이전트 — phorce 판: 파라미터가 아니라 **의도 태그**를 고른다.

실행 경로 분리 원칙:
- LLM은 태그만 낸다 ({"intent_tag", "urgency", "reasoning", "confidence"})
- 태그는 TagSafetyGuard(카탈로그 태그 화이트리스트)를 통과해야 한다
- motion_id 최종 결정은 선택기(MotionSelector)가 한다
- LLM이 죽거나 헛소리하면 TagRuleFallback(실패 타입 → 태그 표)로 간다

유지된 구조 (임피던스 시절과 동일):
비동기 워커 / JSON 강제 / 하드 타임아웃 / (실패타입,축) 쿨다운 /
이전 시도 이력 프롬프트 주입 / 감사 로그 / 결정적 테스트용 process_pending()
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from robot_core.recovery.events import FailureEvent
from robot_core.recovery.rules import TagDecision, TagRuleFallback
from robot_core.recovery.safety import TagSafetyGuard


@dataclass
class LLMConfig:
    """비용/거동 튜닝은 전부 여기서. 코드에 하드코딩 금지.

    model 문자열은 **API로 검증되지 않았다** (개발 환경에 ANTHROPIC_API_KEY 없음).
    캠프 현장에서 키를 넣은 뒤 아래로 먼저 확인할 것 — 실패하면 대안 문자열로
    교체한다. LLM 경로가 죽어도 규칙 폴백으로 시스템은 계속 돈다:

        python scripts/check_llm_model.py
    """

    model: str = "claude-opus-5"
    max_tokens: int = 512
    effort: str = "low"            # 3초 예산 안에 응답해야 하므로 낮게
    timeout_s: float = 3.0         # 초과 시 응답 폐기 + 폴백
    cooldown_s: float = 5.0        # 동일 (실패타입, 축) 재질의 최소 간격
    min_confidence: float = 0.4    # 이보다 낮으면 LLM 제안 폐기 + 폴백
    history_max: int = 4           # 프롬프트에 넣을 이전 시도 수
    dump_window_s: float = 2.0     # 프롬프트에 넣을 피드백 창
    api_key_env: str = "ANTHROPIC_API_KEY"


class LLMUnavailable(Exception):
    """LLM 경로를 못 쓰는 모든 사유. 메시지가 폴백 사유 라벨이 된다."""


def call_client_with_timeout(client, system: str, user: str,
                             timeout_s: float) -> str:
    """클라이언트 호출을 데몬 스레드로 감싸 하드 타임아웃을 건다.

    회복 에이전트와 의도 해석기가 공유하는 유일한 LLM 호출 경로 —
    타임아웃/에러/빈 응답 처리를 한 곳에서 한다.
    """
    result: dict = {}
    done = threading.Event()

    def target():
        try:
            result["ok"] = client(system, user)
        except Exception as e:
            result["err"] = e
        done.set()

    threading.Thread(target=target, daemon=True, name="llm-call").start()
    if not done.wait(timeout_s):
        raise LLMUnavailable(f"timeout after {timeout_s}s")
    if "err" in result:
        raise LLMUnavailable(
            f"client error: {type(result['err']).__name__}: {result['err']}")
    text = result.get("ok")
    if not isinstance(text, str) or not text.strip():
        raise LLMUnavailable("empty response")
    return text


class AnthropicChatClient:
    """실제 Anthropic API 클라이언트. (system, user) -> 응답 텍스트.

    API 키는 환경변수로만 받는다. 테스트에서는 절대 사용하지 말 것 —
    테스트는 fake client를 주입한다.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None

    @staticmethod
    def available(config: LLMConfig | None = None) -> bool:
        if not os.environ.get((config or LLMConfig()).api_key_env):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def __call__(self, system: str, user: str) -> str:
        if not os.environ.get(self.config.api_key_env):
            raise LLMUnavailable(f"no API key in ${self.config.api_key_env}")
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("anthropic package not installed") from e

        if self._client is None:
            self._client = anthropic.Anthropic()

        response = self._client.with_options(
            timeout=self.config.timeout_s, max_retries=0
        ).messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            output_config={"effort": self.config.effort},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise LLMUnavailable("model refused")
        return "".join(b.text for b in response.content if b.type == "text")


SYSTEM_PROMPT = """You are the recovery brain of a motion-playback robot.
The robot can ONLY replay pre-taught motions by id; you cannot command torques,
gains, or trajectories. Your job: given a failure event and recent feedback,
choose an INTENT TAG describing what kind of motion should play next.
A separate deterministic selector maps your tag to a concrete motion id.

Respond with a SINGLE JSON object and NOTHING else.
- No markdown, no code fences, no backticks.
- No preamble, no explanation outside the JSON.
- Schema (exactly these keys):
{"intent_tag": "<one tag from AVAILABLE INTENT TAGS>",
 "urgency": "normal" | "slow" | "stop",
 "reasoning": "<one short sentence>",
 "confidence": <number 0..1>}

Rules:
- intent_tag MUST be one of the AVAILABLE INTENT TAGS. Anything else is rejected.
- urgency: "slow" = prefer slow variants, "stop" = insert a rest/hold and wait.
- If previous attempts are listed, do NOT repeat a tag that already failed.
- If you are unsure, pick a conservative tag (rest/retreat) with low confidence."""


@dataclass
class _Attempt:
    wall_t: float
    source: str
    intent_tag: str
    urgency: str
    reasoning: str = ""


class LLMRecoveryAgent:
    """백그라운드 워커. FailureEvent → (LLM | 규칙) → TagDecision.

    결정은 on_decision 콜백(감독 루프가 등록)으로 전달된다 — 이 에이전트는
    아무것도 실행하지 않는다.
    """

    def __init__(
        self,
        guard: TagSafetyGuard,
        rules: TagRuleFallback,
        cache=None,                 # FeedbackCache — 프롬프트 텔레메트리용
        config: LLMConfig | None = None,
        client=None,
        on_decision=None,           # callable(TagDecision, FailureEvent)
        on_log=None,
        time_fn=time.monotonic,
    ) -> None:
        """client: callable(system, user) -> str. None이면 항상 규칙 폴백."""
        self.guard = guard
        self.rules = rules
        self.cache = cache
        self.config = config or LLMConfig()
        self.client = client
        self.on_decision = on_decision
        self._on_log = on_log
        self._time = time_fn

        self._queue: queue.Queue[FailureEvent] = queue.Queue(maxsize=32)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_handled: dict[tuple[str, int], float] = {}
        self._history: dict[tuple[str, int], deque[_Attempt]] = defaultdict(
            lambda: deque(maxlen=self.config.history_max)
        )
        self.stats = defaultdict(int)
        self._idle = threading.Event()
        self._idle.set()

    # ------------------------------------------------------------- 수명주기
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="llm-recovery")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def process_pending(self) -> int:
        """큐의 이벤트를 호출 스레드에서 즉시 처리 (결정적 리허설/테스트용).

        워커 스레드(start())와 병행하지 말 것. 실물은 start() 워커를 쓴다 —
        판단 루프가 LLM 왕복(수 초)에 블로킹되면 안 되므로.
        """
        n = 0
        while True:
            try:
                ev = self._queue.get_nowait()
            except queue.Empty:
                return n
            self._handle(ev)
            n += 1

    def wait_idle(self, timeout: float = 10.0) -> bool:
        deadline = self._time() + timeout
        while self._time() < deadline:
            if self._queue.empty() and self._idle.is_set():
                return True
            time.sleep(0.01)
        return False

    # ---------------------------------------------------------------- 제출
    def submit(self, event: FailureEvent) -> bool:
        """이벤트를 큐에 넣는다. 논블로킹. 쿨다운/큐 포화면 False."""
        now = self._time()
        last = self._last_handled.get(event.key)
        if last is not None and now - last < self.config.cooldown_s:
            self.stats["dropped_cooldown"] += 1
            return False
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.stats["dropped_queue_full"] += 1
            return False
        self._last_handled[event.key] = now
        self.stats["submitted"] += 1
        return True

    # ---------------------------------------------------------------- 워커
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._idle.clear()
            try:
                self._handle(event)
            except Exception as e:  # 워커는 무슨 일이 있어도 죽으면 안 된다
                self._log(f"recovery worker error (ignored): {type(e).__name__}: {e}")
            finally:
                self._idle.set()

    def _handle(self, event: FailureEvent) -> None:
        self._log(f"handling {event.describe()}")
        source = "llm"
        reasoning = ""
        confidence = 1.0
        tag = urgency = None

        try:
            if self.client is None:
                raise LLMUnavailable("no LLM client configured")
            text = self._call_with_timeout(event)
            raw_tag, urgency, reasoning, confidence = self._parse_response(text)
            if confidence < self.config.min_confidence:
                raise LLMUnavailable(
                    f"low confidence {confidence:.2f} < {self.config.min_confidence}")
            tag = self.guard.validate(raw_tag, source="llm")
            if tag is None:
                raise LLMUnavailable(f"tag {raw_tag!r} rejected by guard")
            self.stats["llm_ok"] += 1
            self._log(f'LLM intent: {tag!r}/{urgency} "{reasoning}" '
                      f"(confidence {confidence:.2f})")
        except LLMUnavailable as e:
            source = f"rules:{e}"
            self.stats["llm_fallback"] += 1
            fb_tag, urgency = self.rules.propose(event)
            reasoning = f"rule-based fallback ({e})"
            confidence = 1.0
            # 폴백 태그도 가드를 통과시킨다 — 매핑 테이블이 카탈로그와 어긋나면
            # 여기서 잡혀야 한다 (조용한 미지 태그 금지)
            tag = self.guard.validate(fb_tag, source=source)
            if tag is None:
                self.stats["fallback_tag_rejected"] += 1
                self._log(f"fallback tag {fb_tag!r}도 카탈로그에 없다 — "
                          f"안전 기본값 'rest' 시도")
                tag = self.guard.validate("rest", source=source + "+default")
                urgency = "stop"
            self._log(f"falling back to rules: {e}")

        decision = TagDecision(intent_tag=tag, urgency=urgency, source=source,
                               reasoning=reasoning, confidence=confidence)
        self._history[event.key].append(_Attempt(
            wall_t=self._time(), source=source, intent_tag=str(tag),
            urgency=str(urgency), reasoning=reasoning))
        if self.cache is not None:
            self.cache.mark_event(
                f"recovery[{source.split(':')[0]}] for {event.type.value} "
                f"ax{event.joint_idx}: intent={tag!r} urgency={urgency}")
        if self.on_decision is not None and tag is not None:
            self.on_decision(decision, event)

    # ------------------------------------------------------------ LLM 호출
    def _call_with_timeout(self, event: FailureEvent) -> str:
        try:
            return call_client_with_timeout(
                self.client, SYSTEM_PROMPT, self._build_prompt(event),
                self.config.timeout_s)
        except LLMUnavailable as e:
            if "timeout" in str(e):
                self.stats["llm_timeout"] += 1
            raise

    def _build_prompt(self, event: FailureEvent) -> str:
        parts = ["FAILURE EVENT", event.describe(), ""]

        if self.cache is not None:
            parts += ["RECENT FEEDBACK",
                      self.cache.dump_text(window_sec=self.config.dump_window_s), ""]

        parts += ["AVAILABLE INTENT TAGS (the ONLY valid values for intent_tag)",
                  self.guard.describe_whitelist(), ""]

        history = self._history.get(event.key)
        if history:
            parts.append("PREVIOUS RECOVERY ATTEMPTS FOR THIS SAME FAILURE "
                         "(it recurred, so these did not permanently fix it):")
            for i, at in enumerate(history, 1):
                parts.append(f"{i}. source={at.source} tag={at.intent_tag!r} "
                             f"urgency={at.urgency} ({at.reasoning})")
            parts.append("Do not repeat a tag that already failed.")
            parts.append("")

        parts.append("Respond with the JSON object now.")
        return "\n".join(parts)

    # ---------------------------------------------------------------- 파싱
    @staticmethod
    def _parse_response(text: str) -> tuple[str, str, str, float]:
        """엄격 파싱. 실패 사유는 전부 LLMUnavailable로 → 폴백."""
        raw = text.strip()
        # 프롬프트로 금지했지만, 그래도 펜스를 붙여 오면 한 번은 벗겨본다
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMUnavailable(f"unparseable JSON: {e}") from None

        if not isinstance(data, dict):
            raise LLMUnavailable("response is not a JSON object")
        tag = data.get("intent_tag")
        urgency = data.get("urgency")
        reasoning = data.get("reasoning", "")
        confidence = data.get("confidence")
        if not isinstance(tag, str) or not tag:
            raise LLMUnavailable("missing/invalid 'intent_tag'")
        if urgency not in ("normal", "slow", "stop"):
            raise LLMUnavailable(f"invalid 'urgency': {urgency!r}")
        if not isinstance(reasoning, str):
            raise LLMUnavailable("invalid 'reasoning'")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LLMUnavailable("missing/invalid 'confidence'")
        return tag, urgency, reasoning, float(confidence)

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(msg)
            except Exception:
                pass


def check_model(config: LLMConfig | None = None) -> int:
    """모델 문자열이 실제로 존재하는지 최소 요청(max_tokens=1)으로 확인.

    현장 첫날 LLM 경로 점검용. 키가 없으면 '미확인'으로 보고하고 넘어간다 —
    LLM이 없어도 회복 루프는 규칙 폴백으로 동작하므로 치명적 실패가 아니다.
    반환값: 0=확인됨/건너뜀, 1=모델 문자열 문제 (교체 필요).
    """
    cfg = config or LLMConfig()
    print(f"[check-model] LLMConfig.model = {cfg.model!r}")

    if not os.environ.get(cfg.api_key_env):
        print(f"[check-model] SKIPPED — ${cfg.api_key_env} 없음. 모델 문자열 '미확인'.")
        print("[check-model] 키를 설정한 뒤 다시 실행할 것. "
              "(LLM 없이도 규칙 폴백으로 시스템은 동작)")
        return 0
    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("[check-model] SKIPPED — anthropic 패키지 없음: pip install anthropic")
        return 0

    probe = LLMConfig(**{**cfg.__dict__, "max_tokens": 1})
    try:
        AnthropicChatClient(probe)("ping", "ping")
        print(f"[check-model] OK — {cfg.model!r} 사용 가능")
        return 0
    except Exception as e:
        print(f"[check-model] FAILED — {type(e).__name__}: {e}")
        print("[check-model] LLMConfig.model 을 유효한 문자열로 교체할 것 "
              "(예: 'claude-sonnet-4-6'). 교체해도 실패하면 규칙 폴백으로 진행.")
        return 1
