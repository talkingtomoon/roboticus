"""LLM 자가 회복 에이전트 (비동기 워커).

절대 원칙 (이거 어기면 로봇 부순다):
1. 제어 루프를 절대 블로킹하지 않는다 — 모든 LLM 호출은 이 파일의 워커 스레드
   안에서만 일어난다. submit()은 큐에 넣고 즉시 리턴한다.
2. LLM 출력을 그대로 믿지 않는다 — 액션은 반드시 SafetyGuard를 통과한다.
   파싱 실패 / 타임아웃 / 낮은 신뢰도 / 스키마 위반 → 전부 규칙 기반 폴백.
3. LLM 없이도 동작한다 — client=None이거나 API 키가 없으면 곧장 폴백으로 간다.

응답 스키마 (JSON only):
    {"diagnosis": str,
     "actions": [{"node": str, "param": str, "value": float}, ...],
     "confidence": float}   # 0..1
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
from robot_core.recovery.rules import RuleBasedRecovery
from robot_core.recovery.safety import SafetyGuard


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
    cooldown_s: float = 5.0        # 동일 (실패타입, 관절) 재질의 최소 간격 (벽시계)
    min_confidence: float = 0.4    # 이보다 낮으면 LLM 제안 폐기 + 폴백
    history_max: int = 4           # 프롬프트에 넣을 이전 시도 수
    dump_window_s: float = 1.0     # 프롬프트에 넣을 로그 창
    api_key_env: str = "ANTHROPIC_API_KEY"


class LLMUnavailable(Exception):
    """LLM 경로를 못 쓰는 모든 사유. 메시지가 폴백 사유 라벨이 된다."""


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


SYSTEM_PROMPT = """You are a real-time robot recovery assistant embedded in a control system.
A failure was detected on a torque-controlled robot arm. You will receive the failure
event, recent telemetry, current tunable parameters and their allowed ranges, and any
previous recovery attempts.

Respond with a SINGLE JSON object and NOTHING else.
- No markdown, no code fences, no backticks.
- No preamble, no explanation outside the JSON.
- Schema (exactly these keys):
{"diagnosis": "<one short sentence>",
 "actions": [{"node": "<node name>", "param": "<param name>", "value": <number>}],
 "confidence": <number 0..1>}

Rules:
- Only use node/param pairs listed under TUNABLE PARAMETERS. Anything else is rejected.
- Stay within the allowed ranges. Out-of-range values get clamped.
- Values can change at most by the listed per-change limits; plan multi-step if needed.
- If previous attempts are listed, do NOT repeat an adjustment that already failed.
- Prefer small, reversible changes. 1-3 actions maximum.
- If you are unsure, return an empty actions list with low confidence."""


@dataclass
class _Attempt:
    wall_t: float
    source: str
    diagnosis: str
    applied: list[str] = field(default_factory=list)  # "node.param req -> applied (status)"


class LLMRecoveryAgent:
    """백그라운드 워커. FailureEvent를 받아 LLM 또는 규칙으로 회복 액션을 만든다."""

    def __init__(
        self,
        guard: SafetyGuard,
        rules: RuleBasedRecovery,
        logger=None,
        config: LLMConfig | None = None,
        client=None,
        on_log=None,
        time_fn=time.monotonic,
    ) -> None:
        """client: callable(system, user) -> str. None이면 항상 규칙 폴백."""
        self.guard = guard
        self.rules = rules
        self.logger = logger
        self.config = config or LLMConfig()
        self.client = client
        self._on_log = on_log
        self._time = time_fn

        self._queue: queue.Queue[FailureEvent] = queue.Queue(maxsize=32)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_handled: dict[tuple[str, int], float] = {}
        self._history: dict[tuple[str, int], deque[_Attempt]] = defaultdict(
            lambda: deque(maxlen=self.config.history_max)
        )
        self.stats = defaultdict(int)  # submitted/dropped_cooldown/llm_ok/llm_fallback/...
        self._idle = threading.Event()
        self._idle.set()

    # ------------------------------------------------------------- 수명주기
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="llm-recovery")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def process_pending(self) -> int:
        """큐에 쌓인 이벤트를 호출 스레드에서 즉시 처리하고 개수를 돌려준다.

        결정적 리허설/테스트용 동기 경로 — 워커 스레드(start())와 병행하지 말 것.
        실물 운용에서는 start()로 백그라운드 워커를 쓴다 (제어 루프 비블로킹).
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
        """큐가 비고 처리 중인 이벤트가 없어질 때까지 대기 (테스트/데모용)."""
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
        actions: list[dict] = []
        diagnosis = ""
        source = "llm"

        try:
            if self.client is None:
                raise LLMUnavailable("no LLM client configured")
            text = self._call_with_timeout(event)
            diagnosis, actions, confidence = self._parse_response(text)
            if confidence < self.config.min_confidence:
                raise LLMUnavailable(
                    f"low confidence {confidence:.2f} < {self.config.min_confidence}"
                )
            self.stats["llm_ok"] += 1
            self._log(f'LLM diagnosis: "{diagnosis}" (confidence {confidence:.2f})')
        except LLMUnavailable as e:
            source = f"rules:{e}"
            self.stats["llm_fallback"] += 1
            actions = self.rules.propose(event)
            diagnosis = f"rule-based fallback ({e})"
            self._log(f"falling back to rules: {e}")

        audit = self.guard.apply(actions, source=source)
        applied_lines = [a.line() for a in audit]
        for line in applied_lines:
            self._log(f"guard: {line}")
        for a in audit:
            self.stats[f"guard_{a.status}"] += 1

        self._history[event.key].append(
            _Attempt(wall_t=self._time(), source=source, diagnosis=diagnosis,
                     applied=applied_lines)
        )
        if self.logger is not None:
            self.logger.mark_event(
                f"recovery[{source.split(':')[0]}] for {event.type.value} j{event.joint_idx}: "
                + (", ".join(
                    f"{a.node}.{a.param}={a.applied:.3g}" for a in audit if a.applied is not None
                ) or "no action")
            )

    # ------------------------------------------------------------ LLM 호출
    def _call_with_timeout(self, event: FailureEvent) -> str:
        """클라이언트 호출을 데몬 스레드로 감싸 하드 타임아웃을 건다.

        (타임아웃 나도 호출 스레드는 데몬이라 프로세스 종료를 막지 않는다)
        """
        system = SYSTEM_PROMPT
        user = self._build_prompt(event)
        result: dict = {}
        done = threading.Event()

        def target():
            try:
                result["ok"] = self.client(system, user)
            except Exception as e:
                result["err"] = e
            done.set()

        threading.Thread(target=target, daemon=True, name="llm-call").start()
        if not done.wait(self.config.timeout_s):
            self.stats["llm_timeout"] += 1
            raise LLMUnavailable(f"timeout after {self.config.timeout_s}s")
        if "err" in result:
            raise LLMUnavailable(f"client error: {type(result['err']).__name__}: {result['err']}")
        text = result.get("ok")
        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailable("empty response")
        return text

    def _build_prompt(self, event: FailureEvent) -> str:
        parts = ["FAILURE EVENT", event.describe(), ""]

        if self.logger is not None:
            parts += ["RECENT TELEMETRY",
                      self.logger.dump_text(window_sec=self.config.dump_window_s), ""]

        parts += ["TUNABLE PARAMETERS (whitelist — the ONLY things you may change)",
                  self.guard.describe_whitelist(), ""]

        history = self._history.get(event.key)
        if history:
            parts.append("PREVIOUS RECOVERY ATTEMPTS FOR THIS SAME FAILURE "
                         "(it recurred, so these did not permanently fix it):")
            for i, at in enumerate(history, 1):
                parts.append(f"{i}. source={at.source} diagnosis={at.diagnosis!r}")
                for line in at.applied:
                    parts.append(f"   {line}")
            parts.append("Do not repeat an identical adjustment.")
            parts.append("")

        parts.append("Respond with the JSON object now.")
        return "\n".join(parts)

    # ---------------------------------------------------------------- 파싱
    @staticmethod
    def _parse_response(text: str) -> tuple[str, list[dict], float]:
        """엄격 파싱. 실패 사유는 전부 LLMUnavailable로 → 폴백."""
        raw = text.strip()
        # 프롬프트로 금지했지만, 그래도 펜스를 붙여 오면 한 번은 벗겨본다 (경고성 관용)
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
        diagnosis = data.get("diagnosis")
        actions = data.get("actions")
        confidence = data.get("confidence")
        if not isinstance(diagnosis, str):
            raise LLMUnavailable("missing/invalid 'diagnosis'")
        if not isinstance(actions, list):
            raise LLMUnavailable("missing/invalid 'actions'")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LLMUnavailable("missing/invalid 'confidence'")
        # 액션 항목의 세부 검증은 SafetyGuard가 한다 (여기서 거르면 감사 로그에 안 남는다)
        return diagnosis, actions, float(confidence)

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
        import anthropic
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
              "(예: 'claude-sonnet-4-6'). 교체해도 실패하면 --no-llm 경로로 진행.")
        return 1


