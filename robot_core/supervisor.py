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
핸들 유실 시 폴백은 hal.busy_state() — 실물에서는 robot.status().state_name
이고, "IDLE"만 유휴의 증거다. "unknown"(STALE 등)이면 판단을 **보류**한다
(다른 신호로 idle을 추측하지 말 것 — 문서 명시. motion_slot_state 폴링 금지).

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


# 피드백 신선도 한계 — 매뉴얼 상수 kStateFreshLimitMs. 임의값이 아니다.
K_STATE_FRESH_LIMIT_MS = 1500


class SupervisorState(str, Enum):
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    DECIDING = "DECIDING"
    WAITING_OPERATOR = "WAITING_OPERATOR"
    HALTED = "HALTED"          # 소프트 정지: 관찰은 계속, 새 play 금지
    DONE = "DONE"


@dataclass
class SupervisorConfig:
    decision_hz: float = 2.0
    detector_window_s: float = 2.0
    # 결정적 리허설: 회복 이벤트를 tick 안에서 동기 처리.
    # 실물은 False + agent.start() 워커 (판단 루프가 LLM 왕복에 막히면 안 됨)
    sync_recovery: bool = True
    # 피드백 워치독: 수신 벽시계 기준 이 시간 이상 새 프레임이 없으면 halt.
    # (몇 초 전 상태를 보고 play를 계속 던지는 것이 최악의 실패 모드다)
    fresh_limit_s: float = K_STATE_FRESH_LIMIT_MS / 1000.0
    # 버스 전압 경고 (한 줄짜리 감시 — 전압 강하는 실물의 흔한 문제)
    bus_v_warn: float = 42.0
    # 도달 판정: 재생 완료 후 |q - end_pose| 최대 오차가 이보다 크면
    # MOTION_INCOMPLETE (완료 ≠ 성공 — 재생 완료는 시간 기준일 뿐이다).
    # end_pose 없는 구버전 주석은 판정불가로 통과.
    incomplete_err_rad: float = 0.15

    # 서킷브레이커: 같은 (실패, 축)이 시간창 안에 N회면 rest로 강등, M회면 halt.
    # (stall→retry→stall→retry 무한 루프 차단.)
    # 리셋 신호로 '계획 전진'을 쓰지 않는 이유: 재생 완료는 시간 기준이라
    # 물리적으로 실패해도 계획이 전진한다 — 진행의 증거가 못 된다.
    # 대신 롤링 창: 창 밖으로 밀려난 과거 실패는 잊는다.
    breaker_demote: int = 3
    breaker_halt: int = 5
    breaker_window_s: float = 60.0


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
        guard=None,                    # TagSafetyGuard — 모든 태그 진입점의 관문
    ) -> None:
        self.hal = hal
        self.cache = cache
        self.detector = detector
        self.selector = selector
        self.agent = agent
        self.catalog = catalog
        self.plan = plan
        self.guard = guard if guard is not None else getattr(agent, "guard", None)
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
        self._pending_lock = threading.Lock()   # 워커→판단 스레드 전달 원자성
        self._hold_rest = False          # urgency "stop" → 과열/이상 해제까지 휴지
        self.halt_reason: str | None = None
        self._fail_times: dict = {}      # (실패타입, 축) → [event.t, ...] (브레이커)
        self._last_busv_warn = -1e9
        self.busy_rejections = 0         # 재생 중 play 시도 (0이어야 정상)
        self.no_candidate_ticks = 0
        self.last_arrival: dict | None = None   # 직전 재생의 도달 판정 (뷰용)
        self.last_rejection: dict | None = None  # 직전 거절 (guidance가 detail 노출)
        self._busy_unknown_marked = False
        self._dob_peak: tuple | None = None     # (peak, axis) — _observe가 갱신
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    # ------------------------------------------------------------- 외부 API
    def operator_cleared(self) -> None:
        """사람이 영점/복구를 마친 뒤 호출 — 이걸 부르기 전엔 재시도하지 않는다."""
        if self.state == SupervisorState.WAITING_OPERATOR:
            self.cache.mark_event("operator: cleared — resuming decisions")
            self.state = SupervisorState.DECIDING

    def halt(self, reason: str = "operator halt") -> None:
        """소프트 정지 (어느 스레드에서든). 재생 중 모션은 중단 불가(인터페이스
        제약)이므로 끝까지 가지만, **새 play는 절대 나가지 않는다.**
        관찰(감지·타임라인)은 계속 돈다. resume()으로만 해제."""
        if self.state != SupervisorState.HALTED:
            self.halt_reason = reason
            self.state = SupervisorState.HALTED
            self.cache.mark_event(f"HALT: {reason} — no further plays until resume()")
            print(f"** HALT: {reason}")

    def resume(self) -> None:
        if self.state == SupervisorState.HALTED:
            self.cache.mark_event(f"resume (was halted: {self.halt_reason})")
            self.halt_reason = None
            self._fail_times.clear()   # 사람이 개입했다 — 브레이커 새 출발
            self.state = SupervisorState.DECIDING

    def request_intent(self, tag: str, urgency: str = "normal",
                       source: str = "operator") -> bool:
        """외부(사람/음성/미래의 의도 파이프라인)에서 의도 태그 주입.

        진입점이 어디든 태그는 전부 같은 관문(TagSafetyGuard)을 지난다 —
        STT 오인식("정지"→"저지")이 미지 태그로 들어와도 여기서 거부된다.
        """
        if urgency not in ("normal", "slow", "stop"):
            self.cache.mark_event(
                f"intent rejected: invalid urgency {urgency!r} (src={source})")
            return False
        if self.guard is None:
            self.cache.mark_event(
                f"intent rejected: no TagSafetyGuard configured (src={source})")
            return False
        accepted = self.guard.validate(tag, source=source)
        if accepted is None:
            self.cache.mark_event(
                f"intent rejected by guard: {tag!r} (src={source})")
            return False
        self._on_decision(TagDecision(intent_tag=accepted, urgency=urgency,
                                      source=source, reasoning="external intent"),
                          None)
        self.cache.mark_event(f"intent accepted: {accepted!r}/{urgency} (src={source})")
        return True

    # ------------------------------------------------------------- 뷰 전용
    def snapshot(self) -> dict:
        """상태 스냅샷 — 웹 UI 등 외부 관찰자용. 락 잡고 dict 반환, 조작 없음.

        (UI는 이 메서드 + halt/resume/request_intent만 쓴다 — 내부 접근 금지)
        """
        with self._pending_lock:
            pending = self._pending
        fb = self.cache.latest()
        wall = self.cache.latest_wall()
        axes = None
        if fb is not None:
            usable = fb.usable
            temps = fb.temp_c[usable] if usable.any() else fb.temp_c
            hot_ax = int(np.argmax(fb.temp_c)) if len(fb.temp_c) else -1
            # 외란은 **최근 창의 피크**로 본다 — 충격은 수백 ms 과도 신호라
            # 순간값만 보면 2Hz 관측 사이로 빠져나간다.
            # 값은 _observe()가 감지기용 배열에서 한 번에 계산해 둔 것을 읽는다
            # (여기서 to_arrays를 또 부르면 틱마다 버퍼를 두 번 훑게 된다)
            if self._dob_peak is not None:
                dob_max, dob_ax = self._dob_peak
            else:
                dob = np.where(usable, np.abs(fb.dob_a), 0.0)
                dob_ax = int(np.argmax(dob)) if dob.size else -1
                dob_max = float(dob[dob_ax]) if dob.size else 0.0
            busv = fb.bus_v[usable] if usable.any() else fb.bus_v
            axes = {
                "valid": int(usable.sum()),
                "total": int(len(usable)),
                "fault": int(fb.fault.sum()),
                "temp_max": round(float(temps.max()), 1) if temps.size else None,
                "temp_max_axis": hot_ax,
                # 외란 관측기 — 접촉/충격의 1차 신호 (UI 입력 패널이 쓴다).
                # 최근 lookback 창의 피크 (순간값 아님)
                "dob_max": round(dob_max, 2),
                "dob_max_axis": dob_ax,
                "bus_v_min": round(float(busv.min()), 1) if busv.size else None,
                "seq": int(fb.seq),
            }
        cur = None
        if self._current_meta is not None:
            m = self._current_meta
            cur = {"id": m.id, "name": m.name, "tags": list(m.tags),
                   "duration_s": round(float(m.duration_s), 2)}
        elif self._handle is not None:
            cur = {"id": self._handle.motion_id, "name": "?", "tags": [],
                   "duration_s": None}
        if cur is not None and self._handle is not None \
                and self._handle.t_start is not None and fb is not None:
            # 재생 진행률 — 로봇 클록 기준 (없으면 None)
            elapsed = max(0.0, float(fb.t) - float(self._handle.t_start))
            cur["elapsed_s"] = round(elapsed, 2)
            if cur.get("duration_s"):
                cur["progress"] = round(min(1.0, elapsed / cur["duration_s"]), 3)
        return {
            "state": self.state.value,
            "halt_reason": self.halt_reason,
            "plan": {"cursor": self.plan.cursor,
                     "total": len(self.plan.tag_sequence),
                     "next_tag": self.plan.next_tag},
            "current_motion": cur,
            "pending_intent": (None if pending is None else
                               {"tag": pending.intent_tag,
                                "urgency": pending.urgency,
                                "source": pending.source}),
            "hold_rest": self._hold_rest,
            "last_arrival": self.last_arrival,
            "last_rejection": self.last_rejection,
            "busy_rejections": self.busy_rejections,
            "breaker": {f"{k[0]}:ax{k[1]}": len(v)
                        for k, v in self._fail_times.items() if v},
            "watchdog": {
                "fresh_ms": (None if wall is None else
                             round((time.monotonic() - wall) * 1e3, 1)),
                "limit_ms": K_STATE_FRESH_LIMIT_MS,
            },
            "axes": axes,
        }

    # ---------------------------------------------------------- 사전 점검
    def preflight(self, wait_s: float = 2.0) -> list[str]:
        """시작 전 게이트. 문제 리스트 반환 (빈 리스트 = 출발 가능).

        로봇이 준비 안 된 상태로 조용히 돌기 시작해 거절만 쌓이는 것을 막는다.
        """
        problems: list[str] = []

        loaded = getattr(self.selector, "loaded", set())
        if len(self.catalog) == 0:
            problems.append("카탈로그가 비었다 — 교시가 안 됐거나 JSON 경로 오류")
        if not loaded:
            problems.append("적재된 모션이 없다 — SD 인식 실패 또는 reconcile 전멸")
        for tag in getattr(self.plan, "tag_sequence", []):
            if not any(tag in m.tags and m.id in loaded for m in self.catalog.all()):
                problems.append(f"계획 태그 {tag!r}에 대응하는 적재 모션이 없다")

        status = {}
        if hasattr(self.hal, "status"):
            try:
                status = self.hal.status() or {}
            except Exception as e:
                problems.append(f"status() 조회 실패: {type(e).__name__}: {e}")
        if status.get("estop_active"):
            problems.append("E-stop이 눌린 상태다 — 해제 후 시작할 것")
        if status.get("ethercat_operational") is False:
            problems.append("EtherCAT이 동작 중이 아니다 — 버스/전원 확인")

        # robot.doctor() 자가진단 — 실물 HAL만 제공한다 (목은 건너뜀)
        if hasattr(self.hal, "doctor"):
            try:
                d = self.hal.doctor() or {}
                if d.get("ok") is False:
                    for issue in d.get("issues") or ["doctor: 원인 미상 실패"]:
                        problems.append(f"doctor: {issue}")
            except Exception as e:
                problems.append(f"doctor() 조회 실패: {type(e).__name__}: {e}")

        # 피드백 침묵 검사 (QoS 실수의 전형적 증상 — 에러 없이 0개 수신)
        first = self.cache.latest()
        if first is None and wait_s > 0:
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline and self.cache.latest() is None:
                time.sleep(0.05)
            first = self.cache.latest()
        if first is None:
            problems.append(
                f"피드백이 한 프레임도 안 왔다 ({wait_s:.0f}s 대기) — "
                "watch 연결/QoS(qos_profile_sensor_data) 확인")
        return problems

    def start(self, skip_preflight: bool = False) -> None:
        """실물용: 2Hz 스레드 루프. 리허설/테스트는 tick()을 직접 부른다.

        preflight가 문제를 찾으면 명확한 메시지와 함께 **시작을 거부**한다.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        if not skip_preflight:
            problems = self.preflight()
            if problems:
                msg = "preflight 실패 — 시작 거부:\n" + "\n".join(
                    f"  - {p}" for p in problems)
                self.cache.mark_event("preflight FAILED: " + "; ".join(problems))
                raise RuntimeError(msg)
            self.cache.mark_event("preflight OK — starting decision loop")
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
        self._watchdog()
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
            elif state == SupervisorState.HALTED:
                # 관찰은 계속, 재생 종료도 기록 — 하지만 새 play는 없다
                if self._handle is not None and self._handle.done():
                    try:
                        mid = self._handle.result()
                        self.cache.mark_event(f"play done: motion {mid} (halted)")
                    except MotionAborted as e:
                        self.cache.mark_event(f"play ABORTED while halted: {e}")
                    self._handle = None
                    self._current_meta = None
                return
            elif state in (SupervisorState.WAITING_OPERATOR, SupervisorState.DONE):
                self._maybe_wake_from_done()
                return

    def _watchdog(self) -> None:
        """피드백 신선도(수신 벽시계) + 버스 전압. 스트림이 죽었는데
        마지막 프레임으로 계속 판단하는 것이 최악의 실패 모드다."""
        wall = self.cache.latest_wall()
        if wall is not None and self.state != SupervisorState.HALTED:
            age = time.monotonic() - wall
            if age > self.cfg.fresh_limit_s:
                self.halt(f"feedback stale {age * 1e3:.0f} ms "
                          f"(> kStateFreshLimitMs={K_STATE_FRESH_LIMIT_MS})")
                return
        fb = self.cache.latest()
        if fb is not None:
            busv = fb.bus_v[fb.usable] if fb.usable.any() else fb.bus_v
            if busv.size and float(busv.min()) < self.cfg.bus_v_warn \
                    and time.monotonic() - self._last_busv_warn > 5.0:
                self._last_busv_warn = time.monotonic()
                self.cache.mark_event(
                    f"WARN bus voltage {float(busv.min()):.1f} V "
                    f"< {self.cfg.bus_v_warn:.0f} V")

    # ------------------------------------------------------------- 내부: 관찰
    def _observe(self) -> None:
        arrays = self.cache.to_arrays(self.cfg.detector_window_s)
        if len(arrays["t"]) < 3:
            return
        self._update_dob_peak(arrays)
        for ev in self.detector.check(arrays):
            self.cache.mark_event(f"DETECTED {ev.describe()}", t=ev.t)
            if ev.type == FailureType.OVERHEAT_CLEARED:
                # 실패가 아니라 복귀 트리거 — LLM/쿨다운 우회
                if self._hold_rest and not self.detector.overheat_active():
                    self._hold_rest = False
                    self.cache.mark_event("hold released — overheat cleared, "
                                          "resuming plan")
                continue

            if self._handle_failure_event(ev):
                return                       # 브레이커가 halt시켰다

    def _handle_failure_event(self, ev: FailureEvent) -> bool:
        """브레이커 집계 → (강등 | 정지 | 회복 제출). True = halt 발생.

        감지기 이벤트와 도달 판정(MOTION_INCOMPLETE)이 같은 경로를 탄다 —
        미도달도 실패로 집계되므로 브레이커 카운트에 포함된다.
        """
        times = self._fail_times.setdefault(ev.key, [])
        times.append(ev.t)
        cutoff = ev.t - self.cfg.breaker_window_s
        times[:] = [t for t in times if t >= cutoff]
        streak = len(times)
        if streak >= self.cfg.breaker_halt:
            self.halt(f"circuit breaker: {ev.type.value} ax{ev.joint_idx} "
                      f"x{streak}/{self.cfg.breaker_window_s:.0f}s — "
                      f"같은 실패 반복, 사람이 볼 차례")
            return True
        if streak >= self.cfg.breaker_demote:
            self.cache.mark_event(
                f"circuit breaker: {ev.type.value} ax{ev.joint_idx} "
                f"x{streak} — demoting to rest (LLM 우회)")
            self._on_decision(TagDecision(
                intent_tag="rest", urgency="stop",
                source=f"breaker:{ev.type.value}x{streak}",
                reasoning="repeated failure"), ev)
            return False

        if self.agent is not None:
            accepted = self.agent.submit(ev)
            if not accepted:
                self.cache.mark_event(
                    f"recovery submit dropped (cooldown/queue) "
                    f"{ev.type.value} ax{ev.joint_idx}")
            elif self.cfg.sync_recovery:
                self.agent.process_pending()
        return False

    def _update_dob_peak(self, arrays: dict) -> None:
        """감지기용 배열에서 외란 피크를 한 번에 뽑아 둔다 (UI 입력 패널용).

        감지기의 lookback 창과 같은 구간을 보므로, 패널의 수치와 감지 발화가
        같은 이야기를 한다.
        """
        t = arrays["t"]
        lb = self.detector.cfg.impact_lookback_s
        m = t >= (t[-1] - lb)
        dw = np.where(arrays["valid"][m], np.abs(arrays["dob"][m]), 0.0)
        if dw.size:
            flat = int(dw.argmax())
            self._dob_peak = (round(float(dw.flat[flat]), 2),
                              int(flat % dw.shape[1]))

    def _on_decision(self, decision: TagDecision, event) -> None:
        """결정 수신 (에이전트 워커/외부 의도 — 어느 스레드든). 최신 결정만 유지."""
        with self._pending_lock:
            self._pending = decision
        if decision.urgency == "stop":
            self._hold_rest = True

    def _take_pending(self) -> TagDecision | None:
        """보류 결정의 원자적 소비 — 읽기→지우기 사이에 낀 새 결정을
        지워버리는 경합을 막는다 (스레드 모드 테스트가 강제)."""
        with self._pending_lock:
            pending, self._pending = self._pending, None
            return pending

    # ------------------------------------------------------- 내부: 경계 감지
    def _check_boundary(self) -> bool:
        """재생 종료 여부. True면 상태가 바뀌어 같은 틱에서 계속 진행."""
        if self._handle is not None:
            if not self._handle.done():
                return False
            try:
                mid = self._handle.result()
                self.cache.mark_event(f"play done: motion {mid}")
                arrival = self._assess_arrival(mid)
                self.last_arrival = arrival
                if arrival["status"] == "incomplete":
                    # 완료 ≠ 성공: 계획을 전진시키지 않는다 — 다음 선곡이
                    # 같은 단계를 다시 시도하고, 반복되면 브레이커가 잡는다
                    ev = FailureEvent(
                        FailureType.MOTION_INCOMPLETE, arrival["axis"],
                        severity=min(1.0, arrival["err_rad"]
                                     / (2 * self.cfg.incomplete_err_rad)),
                        t=self._time(),
                        snapshot={"motion_id": mid,
                                  "err_rad": round(arrival["err_rad"], 3),
                                  "threshold": self.cfg.incomplete_err_rad})
                    self.cache.mark_event(f"DETECTED {ev.describe()}", t=ev.t)
                    self._handle_failure_event(ev)
                elif (self._current_meta is not None
                        and self.plan.advance_if_matches(self._current_meta)):
                    self.cache.mark_event(
                        f"plan advanced -> next={self.plan.next_tag!r} "
                        f"({self.plan.cursor}/{len(self.plan.tag_sequence)})")
            except MotionAborted as e:
                # 로봇은 임의 자세 — 선택기의 진입 하드 필터가 받는다
                self.cache.mark_event(f"play ABORTED: {e} — robot at arbitrary pose")
            self._handle = None
            self._current_meta = None
            if self.state != SupervisorState.HALTED:   # 브레이커 halt 존중
                self.state = SupervisorState.DECIDING
            return True

        # 핸들 유실 폴백: busy_state() — "idle"만 경계의 증거다.
        # "unknown"(STALE/CONTRACT_INACTIVE/UNKNOWN)은 모른다는 뜻이므로
        # 보류하고 다음 틱에 다시 본다 (추측으로 play를 내보내는 게 최악).
        bs = self.hal.busy_state()
        if bs == "idle":
            self._busy_unknown_marked = False
            self.cache.mark_event("boundary via busy_state fallback (handle lost)")
            self._current_meta = None
            self.state = SupervisorState.DECIDING
            return True
        if bs == "unknown" and not self._busy_unknown_marked:
            self._busy_unknown_marked = True    # 틱마다 도배하지 않게 1회만
            self.cache.mark_event(
                "busy_state unknown (handle lost) — 경계 판단 보류")
        return False

    def _assess_arrival(self, motion_id: int) -> dict:
        """재생 완료 후 실제 자세 ↔ 교시 종료 자세(end_pose) 대조.

        end_pose 없는 구버전 주석은 'unknown' (판정불가) — 막지 않는다.
        """
        meta = self._current_meta
        fb = self.cache.latest()
        if meta is None or meta.end_pose is None or fb is None:
            return {"motion_id": motion_id, "status": "unknown",
                    "err_rad": None, "axis": None}
        err_vec = np.where(fb.usable,
                           np.abs(fb.position_rad - meta.end_pose), 0.0)
        axis = int(err_vec.argmax())
        err = float(err_vec[axis])
        status = "incomplete" if err > self.cfg.incomplete_err_rad else "ok"
        return {"motion_id": motion_id, "status": status,
                "err_rad": round(err, 3), "axis": axis}

    # ----------------------------------------------------------- 내부: 선곡
    def _restore_pending(self, taken: TagDecision | None) -> None:
        """선곡이 무산됐을 때 소비했던 결정을 되돌린다 — 단, 그 사이 더 새
        결정이 도착했으면 그쪽이 우선 (최신 결정 유지 원칙)."""
        if taken is None:
            return
        with self._pending_lock:
            if self._pending is None:
                self._pending = taken

    def _decide_and_play(self) -> None:
        taken: TagDecision | None = None
        if self._hold_rest:
            intent, mods = "rest", ()
        else:
            taken = self._take_pending()          # 원자적 소비
            if taken is not None:
                intent = taken.intent_tag
                mods = ("slow",) if taken.urgency == "slow" else ()
            else:
                intent, mods = self.plan.next_tag, ()

        if intent is None:                      # 계획 소진 + 보류 결정 없음
            self.state = SupervisorState.DONE
            self.cache.mark_event("mission plan complete — DONE (holding)")
            return

        fb = self.cache.latest()
        if fb is None:
            self.cache.mark_event("no feedback yet — waiting")
            self._restore_pending(taken)
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
            self._restore_pending(taken)
            return

        # BUSY 원칙: 여기 도달했다는 건 경계 확인이 끝났다는 뜻.
        # 그래도 방어적으로 상태를 마지막에 본다 (폭주 금지의 이중 방어).
        # "unknown"도 skip — 모르는 채로 play를 내보내지 않는다.
        bs = self.hal.busy_state()
        if bs != "idle":
            self.cache.mark_event(
                f"busy_state={bs} at decide time — skip (no play call)")
            self._restore_pending(taken)
            return

        try:
            handle = self.hal.play_async(report.best_id)
        except MotionBusy:
            self.busy_rejections += 1        # 일어나면 안 되는 일 — 계측만
            self.cache.mark_event("BUSY rejection — should not happen (counted)")
            self._restore_pending(taken)
            return
        except MotionRejected as e:
            detail = getattr(e, "detail", "")
            self.last_rejection = {"code": e.code,
                                   "category": getattr(e, "category", "fatal"),
                                   "detail": detail}
            if e.needs_operator:
                self.state = SupervisorState.WAITING_OPERATOR
                self.cache.mark_event(
                    f"operator required (code {e.code}) — WAITING_OPERATOR, "
                    f"자동 재시도 안 함"
                    + (f" | detail: {detail}" if detail else ""))
                print(f"** 사람 개입 필요: 거절 코드 {e.code} "
                      f"({'영점 버튼' if e.code == 12 else '복구 절차'}) — "
                      f"완료 후 operator_cleared() 호출")
            elif getattr(e, "category", None) == "permanent":
                # 3/4: 이 id는 앞으로도 안 된다 — 후보에서 영구 제외
                self.selector.loaded.discard(report.best_id)
                self.cache.mark_event(
                    f"motion {report.best_id} rejected (code {e.code}) — "
                    f"excluded from candidates")
            else:
                # hardware(6/11)/fatal(그 외): id 문제가 아니다 — 재선곡으로
                # 풀리지 않으니 멈추고 파사드 detail을 사람에게 보인다
                self.halt(f"play rejected code {e.code} "
                          f"({getattr(e, 'category', 'fatal')})"
                          + (f": {detail}" if detail else ""))
            self._restore_pending(taken)
            return

        self._handle = handle
        self._current_meta = (self.catalog.get(report.best_id)
                              if report.best_id in self.catalog else None)
        self.state = SupervisorState.PLAYING
        self.cache.mark_event(
            f"play: motion {report.best_id} (intent={intent!r}"
            + (f", mod={list(mods)}" if mods else "") + ")")

    def _maybe_wake_from_done(self) -> None:
        with self._pending_lock:
            has_pending = self._pending is not None
        if self.state == SupervisorState.DONE and has_pending:
            self.state = SupervisorState.DECIDING   # 회복 결정은 DONE에서도 소비
