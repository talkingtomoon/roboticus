"""감독 루프 — "관찰 → 판단 → 선곡"의 오케스트레이터.

두 속도의 분리를 구조로 강제한다 (매뉴얼 핵심 규칙):
- 1kHz 경로: hal.watch(cache.push) — 최신 프레임 저장뿐, 판단 금지
- 2Hz 경로: tick() — 상태 관찰 → 실패 판정 → (경계에서) 선곡 → play_async

상태기계:
    IDLE → DECIDING → PLAYING → (완료|중단) → DECIDING → ...
                         │
                         └ 거절 12/13 → WAITING_OPERATOR (자동 재시도 금지,
                           operator_cleared() 호출까지 대기 + 콘솔 경고)
    계획 소진 + 보류 결정 없음 → DONE

동작 경계 감지의 1차 수단은 play_async 핸들의 완료 이벤트다.
is_busy() 폴링은 핸들 유실 시 폴백으로만 쓴다.

BUSY(5) 원칙: 재생 중에는 play를 아예 부르지 않는다 (경계 확인 후에만 선곡).
BUSY가 났다는 것 자체가 버그다 — busy_rejections 카운터로 계측하고
테스트가 0을 강제한다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from robot_core.catalog.motion_catalog import MotionCatalog
from robot_core.hal.phorce import (
    MotionAborted, MotionBusy, MotionRejected, PhorceHAL,
)
from robot_core.logging.feedback_cache import FeedbackCache
from robot_core.recovery.detector import FailureDetector
from robot_core.recovery.events import FailureEvent, FailureType
from robot_core.recovery.rules import TagDecision
from robot_core.switching.selector import MissionPlan


class SupervisorState(str, Enum):
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    DECIDING = "DECIDING"
    WAITING_OPERATOR = "WAITING_OPERATOR"
    DONE = "DONE"


@dataclass
class SupervisorConfig:
    decision_hz: float = 2.0
    detector_window_s: float = 2.0
    # 결정적 리허설: 회복 이벤트를 tick 안에서 동기 처리.
    # 실물은 False + agent.start() 워커 (판단 루프가 LLM 왕복에 막히면 안 됨)
    sync_recovery: bool = True


class Supervisor:
    def __init__(
        self,
        hal: PhorceHAL,
        cache: FeedbackCache,
        detector: FailureDetector,
        selector,                      # MotionSelector 또는 베이스라인
        agent,                         # LLMRecoveryAgent (None이면 규칙조차 없음)
        catalog: MotionCatalog,
        plan: MissionPlan,
        config: SupervisorConfig | None = None,
        time_fn=None,
    ) -> None:
        self.hal = hal
        self.cache = cache
        self.detector = detector
        self.selector = selector
        self.agent = agent
        self.catalog = catalog
        self.plan = plan
        self.cfg = config or SupervisorConfig()
        self._time = time_fn or (lambda: getattr(hal, "t", time.monotonic()))

        # 1kHz 경로 연결: 콜백은 저장만 한다
        hal.watch(cache.push)
        if agent is not None:
            agent.on_decision = self._on_decision

        self.state = SupervisorState.IDLE
        self._handle = None
        self._current_meta = None
        self._pending: TagDecision | None = None
        self._hold_rest = False          # urgency "stop" → 과열/이상 해제까지 휴지
        self.busy_rejections = 0         # 재생 중 play 시도 (0이어야 정상)
        self.no_candidate_ticks = 0
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    # ------------------------------------------------------------- 외부 API
    def operator_cleared(self) -> None:
        """사람이 영점/복구를 마친 뒤 호출 — 이걸 부르기 전엔 재시도하지 않는다."""
        if self.state == SupervisorState.WAITING_OPERATOR:
            self.cache.mark_event("operator: cleared — resuming decisions")
            self.state = SupervisorState.DECIDING

    def start(self) -> None:
        """실물용: 2Hz 스레드 루프. 리허설/테스트는 tick()을 직접 부른다."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()

        def loop():
            period = 1.0 / self.cfg.decision_hz
            while not self._stop_evt.wait(period):
                try:
                    self.tick()
                except Exception as e:   # 판단 루프는 죽으면 안 된다
                    self.cache.mark_event(f"tick error (ignored): "
                                          f"{type(e).__name__}: {e}")

        self._thread = threading.Thread(target=loop, daemon=True, name="decision-loop")
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ----------------------------------------------------------------- 틱
    def tick(self) -> None:
        """판단 틱 (~2Hz). 무겁지 않아야 한다 — LLM은 비동기, play도 async."""
        self._observe()
        # 상태 전이는 틱당 최대 2단계 (예: 완료 감지 → 즉시 선곡)
        for _ in range(2):
            state = self.state
            if state == SupervisorState.IDLE:
                self.state = SupervisorState.DECIDING
            elif state == SupervisorState.PLAYING:
                if not self._check_boundary():
                    return
            elif state == SupervisorState.DECIDING:
                self._decide_and_play()
                return
            elif state in (SupervisorState.WAITING_OPERATOR, SupervisorState.DONE):
                self._maybe_wake_from_done()
                return

    # ------------------------------------------------------------- 내부: 관찰
    def _observe(self) -> None:
        arrays = self.cache.to_arrays(self.cfg.detector_window_s)
        if len(arrays["t"]) < 3:
            return
        for ev in self.detector.check(arrays):
            self.cache.mark_event(f"DETECTED {ev.describe()}", t=ev.t)
            if ev.type == FailureType.OVERHEAT_CLEARED:
                # 실패가 아니라 복귀 트리거 — LLM/쿨다운 우회
                if self._hold_rest and not self.detector.overheat_active():
                    self._hold_rest = False
                    self.cache.mark_event("hold released — overheat cleared, "
                                          "resuming plan")
                continue
            if self.agent is not None:
                accepted = self.agent.submit(ev)
                if not accepted:
                    self.cache.mark_event(
                        f"recovery submit dropped (cooldown/queue) "
                        f"{ev.type.value} ax{ev.joint_idx}")
                elif self.cfg.sync_recovery:
                    self.agent.process_pending()

    def _on_decision(self, decision: TagDecision, event: FailureEvent) -> None:
        """에이전트 콜백 (워커 스레드에서 올 수 있음). 최신 결정만 유지."""
        self._pending = decision
        if decision.urgency == "stop":
            self._hold_rest = True

    # ------------------------------------------------------- 내부: 경계 감지
    def _check_boundary(self) -> bool:
        """재생 종료 여부. True면 상태가 바뀌어 같은 틱에서 계속 진행."""
        if self._handle is not None:
            if not self._handle.done():
                return False
            try:
                mid = self._handle.result()
                self.cache.mark_event(f"play done: motion {mid}")
                if (self._current_meta is not None
                        and self.plan.advance_if_matches(self._current_meta)):
                    self.cache.mark_event(
                        f"plan advanced -> next={self.plan.next_tag!r} "
                        f"({self.plan.cursor}/{len(self.plan.tag_sequence)})")
            except MotionAborted as e:
                # 로봇은 임의 자세 — 선택기의 진입 하드 필터가 받는다
                self.cache.mark_event(f"play ABORTED: {e} — robot at arbitrary pose")
            self._handle = None
            self._current_meta = None
            self.state = SupervisorState.DECIDING
            return True

        # 핸들 유실 폴백: is_busy 폴링
        if not self.hal.is_busy():
            self.cache.mark_event("boundary via is_busy() fallback (handle lost)")
            self._current_meta = None
            self.state = SupervisorState.DECIDING
            return True
        return False

    # ----------------------------------------------------------- 내부: 선곡
    def _intent(self) -> tuple[str | None, tuple]:
        if self._hold_rest:
            return "rest", ()
        if self._pending is not None:
            mods = ("slow",) if self._pending.urgency == "slow" else ()
            return self._pending.intent_tag, mods
        return self.plan.next_tag, ()

    def _decide_and_play(self) -> None:
        intent, mods = self._intent()
        if intent is None:                      # 계획 소진 + 보류 결정 없음
            self.state = SupervisorState.DONE
            self.cache.mark_event("mission plan complete — DONE (holding)")
            return

        fb = self.cache.latest()
        if fb is None:
            self.cache.mark_event("no feedback yet — waiting")
            return

        report = self.selector.select(
            position=fb.position_rad, valid=fb.usable, dob=fb.dob_a,
            intent_tag=intent, plan=self.plan, modifiers=mods)
        for line in report.table(top=3).splitlines():
            self.cache.mark_event("select: " + line.strip())

        if report.best_id is None:
            self.no_candidate_ticks += 1
            self.cache.mark_event(
                "select: no playable candidate (전원 하드필터) — retry next tick")
            return

        # BUSY 원칙: 여기 도달했다는 건 경계 확인이 끝났다는 뜻.
        # 그래도 방어적으로 is_busy를 마지막에 본다 (폭주 금지의 이중 방어).
        if self.hal.is_busy():
            self.cache.mark_event("still busy at decide time — skip (no play call)")
            return

        try:
            handle = self.hal.play_async(report.best_id)
        except MotionBusy:
            self.busy_rejections += 1        # 일어나면 안 되는 일 — 계측만
            self.cache.mark_event("BUSY rejection — should not happen (counted)")
            return
        except MotionRejected as e:
            if e.needs_operator:
                self.state = SupervisorState.WAITING_OPERATOR
                self.cache.mark_event(
                    f"operator required (code {e.code}) — WAITING_OPERATOR, "
                    f"자동 재시도 안 함")
                print(f"** 사람 개입 필요: 거절 코드 {e.code} "
                      f"({'영점 버튼' if e.code == 12 else '복구 절차'}) — "
                      f"완료 후 operator_cleared() 호출")
            else:
                # 3/4: 이 id는 앞으로도 안 된다 — 후보에서 영구 제외
                self.selector.loaded.discard(report.best_id)
                self.cache.mark_event(
                    f"motion {report.best_id} rejected (code {e.code}) — "
                    f"excluded from candidates")
            return

        self._handle = handle
        self._current_meta = (self.catalog.get(report.best_id)
                              if report.best_id in self.catalog else None)
        if self._pending is not None and not self._hold_rest:
            self._pending = None             # 결정 소비
        self.state = SupervisorState.PLAYING
        self.cache.mark_event(
            f"play: motion {report.best_id} (intent={intent!r}"
            + (f", mod={list(mods)}" if mods else "") + ")")

    def _maybe_wake_from_done(self) -> None:
        if self.state == SupervisorState.DONE and self._pending is not None:
            self.state = SupervisorState.DECIDING   # 회복 결정은 DONE에서도 소비
