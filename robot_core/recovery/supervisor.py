"""RecoverySupervisor — 감지→회복 파이프라인을 하나로 묶는 저주파 모니터.

제어 루프(1kHz)와 완전히 분리된 스레드에서 돈다 (기본 20Hz):
    RingLogger.to_arrays() → FailureDetector.check() → LLMRecoveryAgent.submit()

제어 루프는 이 클래스의 존재를 모른다. 파라미터 갱신은 SafetyGuard →
NodeGraphManager.set_params() 경로로만 넘어간다 (스레드 안전).
"""

from __future__ import annotations

import threading
import time

from robot_core.logging.ring_logger import RingLogger
from robot_core.recovery.detector import FailureDetector
from robot_core.recovery.events import FailureEvent
from robot_core.recovery.llm_agent import LLMRecoveryAgent


class RecoverySupervisor:
    def __init__(
        self,
        detector: FailureDetector,
        agent: LLMRecoveryAgent,
        logger: RingLogger,
        *,
        check_window_s: float = 1.0,
        period_s: float = 0.05,
        on_log=None,
    ) -> None:
        self.detector = detector
        self.agent = agent
        self.logger = logger
        self.check_window_s = float(check_window_s)
        self.period_s = float(period_s)
        self._on_log = on_log

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.events_seen: list[FailureEvent] = []

    # ---------------------------------------------------------------- 폴링
    def poll(self) -> list[FailureEvent]:
        """감지 1회. 백그라운드 스레드 없이 직접 불러도 된다 (테스트용)."""
        arrays = self.logger.to_arrays(window_sec=self.check_window_s)
        events = self.detector.check(arrays)
        for ev in events:
            self.events_seen.append(ev)
            self.logger.mark_event(f"DETECTED {ev.describe()}", t=ev.t)
            self._log(f"DETECTED {ev.describe()}")
            accepted = self.agent.submit(ev)
            if not accepted:
                self._log(f"  (submit dropped — cooldown/queue) {ev.type.value} j{ev.joint_idx}")
        return events

    # ------------------------------------------------------------- 백그라운드
    def start(self) -> None:
        self.agent.start()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="recovery-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.agent.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.poll()
            except Exception as e:  # 모니터가 죽으면 회복 루프 전체가 죽는다
                self._log(f"supervisor poll error (ignored): {type(e).__name__}: {e}")
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.0, self.period_s - elapsed))

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(msg)
            except Exception:
                pass
