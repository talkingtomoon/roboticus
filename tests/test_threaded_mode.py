"""스레드 모드 통합 — 실물이 실제로 쓰는 경로의 첫 검증.

지금까지 129개 테스트는 전부 동기 모드(tick() 직접 호출 + sync_recovery)였다.
실물은 Supervisor.start() 판단 스레드 + agent.start() LLM 워커 + 1kHz 수신
스레드가 동시에 돈다. 이 파일이 그 조합을 검증한다:

- 시뮬 스레드(연속 step) + 판단 스레드 + LLM 워커로 임무 완주
- 캐시 해머링 (push 폭주 중 to_arrays 반복 — deque 순회 경합)
- 워커 스레드 결정 전달의 원자성 (유실 금지)
- 깨끗한 종료 (join 타임아웃 없이)
"""

import json
import threading
import time

import numpy as np
import pytest

from robot_core.hal.phorce import MockPhorceHAL, N_AXES, PhorceFeedback
from robot_core.integration.scenarios import IMPACT_DOB, build_world
from robot_core.logging import FeedbackCache
from robot_core.supervisor import SupervisorConfig, SupervisorState


class SimClock:
    """시뮬 스레드: 목을 벽시계보다 빠르게 계속 돌린다 (실물의 '시간 흐름')."""

    def __init__(self, hal, steps_per_loop=20, sleep_s=0.002):
        self.hal = hal
        self._stop = threading.Event()
        self.errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="sim-clock")
        self.steps_per_loop = steps_per_loop
        self.sleep_s = sleep_s

    def _run(self):
        while not self._stop.is_set():
            try:
                self.hal.step(self.steps_per_loop)
            except BaseException as e:      # 스레드 안 예외를 반드시 표면화
                self.errors.append(e)
                return
            time.sleep(self.sleep_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        assert not self._thread.is_alive(), "시뮬 스레드가 join 안 됨"


def scripted_llm_slow(system, user):
    """워커 스레드 경로를 실제로 태우는 느린 LLM 대역 (50ms)."""
    time.sleep(0.05)
    if "IMPACT" in user:
        return json.dumps({"intent_tag": "retreat", "urgency": "normal",
                           "reasoning": "impact", "confidence": 0.9})
    return json.dumps({"intent_tag": "rest", "urgency": "stop",
                       "reasoning": "unknown", "confidence": 0.6})


def build_threaded(client=None):
    hal, cache, sup, plan, guard = build_world(client=client)
    sup.cfg = SupervisorConfig(decision_hz=20.0, sync_recovery=False)
    # 스레드 모드에서 시간축은 벽시계 (로봇 클록은 시뮬 스레드가 소유)
    sup._time = time.monotonic
    return hal, cache, sup, plan


# ------------------------------------------------------- 전체 스택 스레드
def test_threaded_mission_completes():
    """판단 스레드 + 시뮬 스레드로 S1 임무 완주. 예외/BUSY/미종료 전부 0."""
    hal, cache, sup, plan = build_threaded()
    with SimClock(hal) as sim:
        sup.start()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not (
                plan.done and sup.state == SupervisorState.DONE):
            time.sleep(0.05)
        sup.stop()
        assert sim.errors == [], f"시뮬 스레드 예외: {sim.errors}"

    assert plan.done and sup.state == SupervisorState.DONE
    assert sup.busy_rejections == 0
    tick_errors = [e.text for e in cache.events() if "tick error" in e.text]
    assert tick_errors == [], f"판단 스레드에서 예외: {tick_errors}"


def test_threaded_recovery_via_llm_worker():
    """충격 → LLM **워커 스레드** 결정 → retreat 선곡까지, 전부 비동기로."""
    hal, cache, sup, plan = build_threaded(client=scripted_llm_slow)
    sup.agent.start()
    try:
        # 시뮬이 벽시계보다 빠르므로(≈10x) 임무가 끝나기 전에 주입해야 한다
        with SimClock(hal, steps_per_loop=10) as sim:
            sup.start()
            time.sleep(0.15)                     # 첫 모션 재생 진입
            hal.inject_disturbance(0, IMPACT_DOB, duration=0.5)
            deadline = time.monotonic() + 25.0
            while time.monotonic() < deadline:
                texts = [e.text for e in cache.events()]
                if plan.done and any("recovery[llm]" in t for t in texts):
                    break
                time.sleep(0.05)
            sup.stop()
            assert sim.errors == []
    finally:
        sup.agent.stop()

    texts = [e.text for e in cache.events()]
    assert any("DETECTED IMPACT" in t for t in texts)
    assert any("recovery[llm]" in t for t in texts), "워커 경로가 결정을 전달 못함"
    assert plan.done
    assert sup.busy_rejections == 0


def test_threaded_clean_shutdown_midway():
    """재생 도중 stop() — 데드락/미종료 없이 내려와야 한다."""
    hal, cache, sup, plan = build_threaded()
    with SimClock(hal):
        sup.start()
        time.sleep(0.5)                          # 어중간한 시점
        sup.stop(timeout=3.0)
    assert sup._thread is not None and not sup._thread.is_alive()


# ---------------------------------------------------------- 캐시 경합 해머
def test_cache_hammer_push_vs_to_arrays():
    """1kHz push 폭주와 to_arrays/latest 동시 호출 — 순회 경합이 없어야 한다.

    (deque는 append가 원자적이지만 '순회 중 변형'은 RuntimeError를 던질 수
    있다 — push와 스냅샷이 같은 락을 공유하는지 검증)
    """
    cache = FeedbackCache()
    frame_proto = None

    hal = MockPhorceHAL({1: (0.2, lambda t: np.zeros(N_AXES))})
    frame_proto = hal.latest_feedback()
    stop = threading.Event()
    errors: list[BaseException] = []

    def pusher():
        seq = 0
        while not stop.is_set():
            seq += 1
            f = PhorceFeedback(
                t=seq * 1e-3, seq=seq,
                position_rad=frame_proto.position_rad,
                velocity_rad_s=frame_proto.velocity_rad_s,
                current_a=frame_proto.current_a, dob_a=frame_proto.dob_a,
                bus_v=frame_proto.bus_v, temp_c=frame_proto.temp_c,
                kp_echo=frame_proto.kp_echo, kd_echo=frame_proto.kd_echo,
                valid=frame_proto.valid, oper=frame_proto.oper,
                stale=frame_proto.stale, fault=frame_proto.fault)
            try:
                cache.push(f)
            except BaseException as e:
                errors.append(e)
                return

    threads = [threading.Thread(target=pusher, daemon=True) for _ in range(2)]
    for th in threads:
        th.start()
    try:
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end:
            try:
                a = cache.to_arrays(0.5)
                _ = cache.latest()
                _ = cache.dump_text(0.2)
                assert len(a["t"]) == len(a["position"])
            except BaseException as e:
                errors.append(e)
                break
    finally:
        stop.set()
        for th in threads:
            th.join(timeout=2.0)
    assert errors == [], f"캐시 경합: {[type(e).__name__ for e in errors]}: {errors[:1]}"


# ------------------------------------------------------ 결정 전달 원자성
def test_pending_decision_not_lost_under_race():
    """워커가 결정을 넣는 동시에 판단 루프가 소비 — 마지막 결정이 유실되면
    안 된다 (읽기→지우기 사이에 낀 새 결정을 지워버리는 고전적 경합)."""
    hal, cache, sup, plan = build_threaded()
    from robot_core.recovery.rules import TagDecision

    stop = threading.Event()
    delivered = []

    def worker():
        i = 0
        while not stop.is_set():
            i += 1
            d = TagDecision(intent_tag="retreat", urgency="normal",
                            source=f"test#{i}", reasoning="")
            sup._on_decision(d, None)
            delivered.append(i)
            time.sleep(0.0005)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        consumed_sources = []
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            got = sup._take_pending()
            if got is not None:
                consumed_sources.append(got.source)
    finally:
        stop.set()
        th.join(timeout=2.0)

    # 소비된 것 + 남은 것 = 전달된 것의 부분집합이며, '마지막' 결정은
    # 반드시 소비됐거나 아직 남아 있어야 한다 (조용한 유실 금지)
    leftover = sup._take_pending()
    last = f"test#{delivered[-1]}"
    assert (leftover is not None and leftover.source == last) or \
        (consumed_sources and consumed_sources[-1] == last), \
        f"마지막 결정 유실: last={last}, consumed[-1]=" \
        f"{consumed_sources[-1] if consumed_sources else None}, leftover={leftover}"


# ------------------------------------------------------ 목 HAL play/step 경합
def test_mock_hal_play_from_other_thread_while_stepping():
    """시뮬 스레드가 step 중일 때 다른 스레드의 play_async — 상태가 안 깨져야."""
    hal = MockPhorceHAL({i: (0.1, lambda t: np.zeros(N_AXES)) for i in range(1, 6)})
    errors: list[BaseException] = []
    with SimClock(hal, steps_per_loop=50, sleep_s=0.0005) as sim:
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end:
            try:
                if not hal.is_busy():
                    h = hal.play_async(int(np.random.default_rng().integers(1, 6)))
                    h.wait(timeout=2.0)
                    h.result()
            except BaseException as e:      # MotionBusy 포함 어떤 예외도 안 됨
                errors.append(e)
                break
        assert sim.errors == [], f"시뮬 스레드 예외: {sim.errors}"
    assert errors == [], f"{[type(e).__name__ for e in errors]}"
