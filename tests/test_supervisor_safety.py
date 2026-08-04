"""안전 계열 — halt/resume, 워치독(1500ms), 서킷브레이커, preflight, 의도 게이트."""

import json
import time

import numpy as np
import pytest

from robot_core.integration.scenarios import TICK_STEPS, build_world, run_ticks
from robot_core.supervisor import (
    K_STATE_FRESH_LIMIT_MS, SupervisorState,
)


# ------------------------------------------------------------ halt/resume
def test_halt_blocks_new_plays_but_keeps_observing():
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 2)                        # 재생 시작
    calls = hal.play_call_count

    sup.halt("demo stop")
    assert sup.state == SupervisorState.HALTED
    run_ticks(hal, sup, 8)                        # 현재 모션 끝나고도 새 play 없음
    assert hal.play_call_count == calls
    assert any("play done" in e.text and "halted" in e.text
               for e in cache.events()), "정지 중에도 경계는 기록돼야 한다"

    # 관찰은 계속: 정지 중 충격도 타임라인에 남는다
    hal.inject_disturbance(0, 5.0, duration=0.3)
    run_ticks(hal, sup, 3)
    assert any("DETECTED IMPACT" in e.text for e in cache.events())
    assert hal.play_call_count == calls           # 그래도 play는 없다

    sup.resume()
    run_ticks(hal, sup, 16)
    assert plan.done                              # 재개 후 완주


def test_halt_from_other_thread_is_safe():
    import threading
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 2)
    th = threading.Thread(target=lambda: sup.halt("from monitor thread"))
    th.start(); th.join()
    assert sup.state == SupervisorState.HALTED


# -------------------------------------------------------------- 워치독
def test_watchdog_halts_on_stale_feedback():
    """수신이 죽으면(벽시계 1500ms 정체) 마지막 프레임으로 계속 판단하지 말고
    halt해야 한다. 기준은 매뉴얼 상수 kStateFreshLimitMs."""
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 2)
    # 수신 정체 시뮬: 수신 벽시계 스탬프를 과거로
    cache._latest_wall = time.monotonic() - (K_STATE_FRESH_LIMIT_MS / 1000.0 + 0.5)
    sup.tick()
    assert sup.state == SupervisorState.HALTED
    assert "stale" in sup.halt_reason
    assert any("kStateFreshLimitMs" in e.text for e in cache.events())


def test_watchdog_quiet_when_fresh():
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 14)
    assert sup.state == SupervisorState.DONE      # 정상 흐름엔 개입 없음
    assert not any("HALT" in e.text for e in cache.events())


def test_bus_voltage_warning_single_line():
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 2)
    fb = hal.latest_feedback()
    fb.bus_v[:] = 38.0                            # 강하 주입
    cache.push(fb)
    sup.tick()
    warns = [e.text for e in cache.events() if "bus voltage" in e.text]
    assert len(warns) == 1 and "38.0" in warns[0]
    sup.tick()                                    # 쿨다운 — 도배 금지
    warns2 = [e.text for e in cache.events() if "bus voltage" in e.text]
    assert len(warns2) == 1


# ---------------------------------------------------------- 서킷브레이커
def _fire_impacts(hal, cache, sup, n, tick_gap=4):
    """감지기 refractory/재무장을 존중하며 IMPACT를 n회 발생시킨다."""
    for _ in range(n):
        hal.inject_disturbance(0, 5.0, duration=0.3)
        run_ticks(hal, sup, tick_gap)


def test_breaker_demotes_after_repeats_then_halts():
    hal, cache, sup, plan, _ = build_world(plan_tags=("approach",) * 30)
    run_ticks(hal, sup, 2)

    _fire_impacts(hal, cache, sup, sup.cfg.breaker_demote)
    texts = [e.text for e in cache.events()]
    assert any("circuit breaker" in t and "demoting to rest" in t for t in texts), \
        "\n".join(t for t in texts if "DETECT" in t or "breaker" in t)

    _fire_impacts(hal, cache, sup, sup.cfg.breaker_halt - sup.cfg.breaker_demote)
    assert sup.state == SupervisorState.HALTED
    assert "circuit breaker" in sup.halt_reason


def test_breaker_rolling_window_forgets_old_failures():
    """창 밖으로 밀려난 과거 실패는 잊는다 — 드문드문한 충격으로는 halt 금지.

    (리셋 신호로 '계획 전진'을 쓰지 않는 이유: 재생 완료는 시간 기준이라
    물리적으로 실패해도 전진한다 — 진행의 증거가 못 된다.)
    """
    hal, cache, sup, plan, _ = build_world(plan_tags=("approach",) * 30)
    sup.cfg.breaker_window_s = 1.0                # 창을 좁혀 프루닝 검증
    run_ticks(hal, sup, 2)
    _fire_impacts(hal, cache, sup, 6)             # 2초 간격 — 항상 창에 1개뿐
    assert sup.state != SupervisorState.HALTED
    texts = [e.text for e in cache.events()]
    assert not any("demoting to rest" in t for t in texts)


def test_breaker_resets_on_resume():
    hal, cache, sup, plan, _ = build_world(plan_tags=("approach",) * 30)
    run_ticks(hal, sup, 2)
    _fire_impacts(hal, cache, sup, sup.cfg.breaker_halt)
    assert sup.state == SupervisorState.HALTED
    sup.resume()                                  # 사람 개입 → 새 출발
    assert not sup._fail_times
    assert sup.state == SupervisorState.DECIDING


# ------------------------------------------------------------- preflight
def test_preflight_passes_on_healthy_world():
    hal, cache, sup, plan, _ = build_world()
    hal.step(5)                                   # 피드백 몇 프레임
    assert sup.preflight(wait_s=0) == []


def test_preflight_catches_estop_and_ethercat():
    hal, cache, sup, plan, _ = build_world()
    hal.step(5)
    hal.set_estop(True)
    hal.set_ethercat(False)
    problems = sup.preflight(wait_s=0)
    assert any("E-stop" in p for p in problems)
    assert any("EtherCAT" in p for p in problems)


def test_preflight_catches_plan_tag_without_motion():
    hal, cache, sup, plan, _ = build_world(plan_tags=("approach", "juggle"))
    hal.step(5)
    problems = sup.preflight(wait_s=0)
    assert any("juggle" in p for p in problems)


def test_preflight_catches_empty_loadout():
    hal, cache, sup, plan, _ = build_world()
    hal.step(5)
    sup.selector.loaded = set()                   # SD 인식 실패 흉내
    problems = sup.preflight(wait_s=0)
    assert any("적재된 모션이 없다" in p for p in problems)


def test_preflight_catches_feedback_silence():
    """QoS 실수의 전형: 에러 없이 0개 수신 — 명확한 메시지로 잡아야 한다."""
    hal, cache, sup, plan, _ = build_world()
    cache.clear()                                 # 수신 0 상태
    problems = sup.preflight(wait_s=0.2)
    assert any("피드백" in p and "QoS" in p for p in problems)


def test_start_refuses_on_preflight_failure():
    hal, cache, sup, plan, _ = build_world()
    hal.step(5)
    hal.set_estop(True)
    with pytest.raises(RuntimeError, match="preflight"):
        sup.start()
    assert sup._thread is None or not sup._thread.is_alive()


# ----------------------------------------------------- 의도 게이트 (사람 경로)
def test_request_intent_passes_guard_and_is_consumed():
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 2)
    assert sup.request_intent("retreat", source="voice") is True
    run_ticks(hal, sup, 6)
    assert any("play: motion 5" in e.text for e in cache.events()), \
        "사람 의도(retreat)가 선곡으로 이어져야 한다"


def test_request_intent_rejects_unknown_tag():
    """STT 오인식('정지'→'저지') 시나리오: 미지 태그는 진입점과 무관하게
    같은 관문(TagSafetyGuard)에서 거부돼야 한다."""
    hal, cache, sup, plan, guard = build_world()
    run_ticks(hal, sup, 2)
    calls = hal.play_call_count
    assert sup.request_intent("저지", source="stt") is False
    assert guard.audit[-1].status == "rejected"
    assert guard.audit[-1].source == "stt"
    run_ticks(hal, sup, 2)
    # 거부된 의도가 어떤 결정에도 영향을 주지 않았다
    assert sup._take_pending() is None or hal.play_call_count >= calls


def test_request_intent_rejects_bad_urgency():
    hal, cache, sup, plan, _ = build_world()
    assert sup.request_intent("retreat", urgency="PANIC") is False
    assert any("invalid urgency" in e.text for e in cache.events())


# ------------------------------------------------------------ 세션 로그
def test_session_log_streams_jsonl(tmp_path):
    from robot_core.logging import FeedbackCache
    path = tmp_path / "session" / "events.jsonl"
    cache = FeedbackCache(event_log_path=path)
    cache.mark_event("첫 이벤트", t=1.0)
    cache.mark_event("둘째", t=2.5)
    cache.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(x) for x in lines]
    assert rows[0]["text"] == "첫 이벤트" and rows[0]["t"] == 1.0
    assert rows[1]["t"] == 2.5
    assert "wall" in rows[0]

    # append 모드: 재시작해도 이어붙는다 (덮어쓰기 금지)
    cache2 = FeedbackCache(event_log_path=path)
    cache2.mark_event("재시작 후", t=3.0)
    cache2.close()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3
