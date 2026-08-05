"""감독 루프 — 상태기계, BUSY 폭주 금지, 사람 개입 대기, 경계 감지, 1kHz 부하."""

import threading
import time

import numpy as np
import pytest

from robot_core.hal.phorce import N_AXES
from robot_core.integration.scenarios import (
    POSE_A, TICK_STEPS, build_world, run_ticks,
)
from robot_core.supervisor import SupervisorState


# ------------------------------------------------------------ BUSY 원칙
def test_no_busy_storm_play_only_at_boundaries():
    """매뉴얼의 '느린 루프' 규칙을 테스트로 강제:
    재생 중 play 시도 = 0. play 호출 수 == 실제 재생 시작 수."""
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 14)

    assert plan.done
    plays = sum(1 for e in cache.events() if e.text.startswith("play: motion"))
    assert sup.busy_rejections == 0
    assert hal.play_call_count == plays          # 거절된 호출이 하나도 없다


def test_tick_while_playing_does_not_call_play():
    hal, cache, sup, plan, _ = build_world()
    sup.tick()                                    # 첫 틱 (피드백 대기)
    hal.step(TICK_STEPS)
    sup.tick()                                    # 선곡 + play
    calls_after_start = hal.play_call_count
    for _ in range(3):                            # 재생 중 여러 틱
        hal.step(100)
        sup.tick()
    assert hal.play_call_count == calls_after_start
    assert sup.state == SupervisorState.PLAYING


# ------------------------------------------------- 12/13 → WAITING_OPERATOR
def test_operator_rejection_waits_without_auto_retry():
    hal, cache, sup, plan, _ = build_world()
    hal.set_rejection(12)
    run_ticks(hal, sup, 4)

    assert sup.state == SupervisorState.WAITING_OPERATOR
    calls = hal.play_call_count
    run_ticks(hal, sup, 5)                        # 대기 중 — 자동 재시도 금지
    assert hal.play_call_count == calls
    assert sup.state == SupervisorState.WAITING_OPERATOR

    hal.clear_rejection()
    run_ticks(hal, sup, 3)                        # cleared 없이는 여전히 대기
    assert hal.play_call_count == calls

    sup.operator_cleared()                        # 사람이 확인해줘야 재개
    run_ticks(hal, sup, 14)
    assert plan.done


def test_code_4_rejection_excludes_id_permanently():
    hal, cache, sup, plan, _ = build_world()
    # 카탈로그에는 있지만 적재가 빠진 상황을 재현: HAL에서 슬롯 제거
    hal._loaded.discard(1)
    hal._loaded.discard(8)                        # approach 계열 전부 미적재
    sup.selector.loaded |= {1, 8}                 # 선택기는 (잘못) 적재로 알고 있음
    run_ticks(hal, sup, 6)

    texts = [e.text for e in cache.events()]
    assert any("rejected (code 4)" in t for t in texts)
    assert 1 not in sup.selector.loaded           # 영구 제외됐다
    assert sup.state != SupervisorState.WAITING_OPERATOR   # 4는 대기 아님


# ------------------------------------------------------------- 경계 감지
def test_boundary_via_handle_completion():
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 6)                        # 대기→선곡→재생(1.0s)→경계
    assert any("play done" in e.text for e in cache.events())


def test_boundary_fallback_via_is_busy_when_handle_lost():
    """핸들 유실 시 is_busy 폴백으로 경계를 잡고 계속 간다.

    폴백 경계는 완료를 어느 모션에 귀속시킬 수 없으므로(핸들이 곧 귀속 증거)
    plan은 전진하지 않는다 — 다음 선곡이 같은 단계를 다시 시도한다.
    """
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 2)                        # 재생 시작
    assert sup.state == SupervisorState.PLAYING
    sup._handle = None                            # 핸들 유실 시뮬레이션
    run_ticks(hal, sup, 3)
    assert any("is_busy() fallback" in e.text for e in cache.events())
    plays = sum(1 for e in cache.events() if e.text.startswith("play: motion"))
    assert plays >= 2                             # 멈추지 않고 재선곡했다
    run_ticks(hal, sup, 14)                       # 이후 정상 완주
    assert plan.done


def test_aborted_motion_leads_to_entry_filtered_reselection():
    """MotionAborted 후 임의 자세: 진입 하드 필터가 처리해야 한다 (요구 6-보완)."""
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 2)                        # motion 1 재생 중
    hal.step(200)                                 # 재생 중간까지
    hal.abort_playback("obstacle hit")
    sup.tick()                                    # 경계: ABORTED 처리 → 재선곡

    texts = [e.text for e in cache.events()]
    assert any("ABORTED" in t for t in texts)
    # 임의 자세에서도 진입 가능한 후보만 선곡됐다 (엉뚱한 자세 재생 방지)
    fb = hal.latest_feedback()
    if sup.state == SupervisorState.PLAYING:
        meta = sup._current_meta
        valid = fb.usable
        dist = float(np.linalg.norm(np.where(valid,
                                             meta.start_pose - fb.position_rad, 0.0)))
        assert dist <= sup.selector.entry_max_dist + 0.2
    hal.clear_rejection() if False else None
    run_ticks(hal, sup, 16)                       # 회복 후 완주
    assert plan.done


# ------------------------------------------------------------ DONE 거동
def test_done_holds_but_wakes_for_recovery_decision():
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 14)
    assert sup.state == SupervisorState.DONE
    calls = hal.play_call_count

    run_ticks(hal, sup, 3)                        # DONE에서 아무것도 안 함
    assert hal.play_call_count == calls

    # DONE 중 충격 → 회복 결정 → 다시 움직인다
    hal.inject_disturbance(0, 6.0, duration=0.3)
    run_ticks(hal, sup, 4)
    assert hal.play_call_count > calls


# ------------------------------------------------- 1kHz 부하 중 판단 루프
def test_decision_tick_stays_fast_under_1khz_push_load():
    """1kHz 콜백(저장만)이 도는 동안 판단 틱이 제때 도는지 (요구 9).

    벽시계 측정이라 다른 테스트/프로세스와 CPU를 다투면 튄다 — 2회 중
    좋은 쪽을 본다. 진짜 회귀(예: 틱이 버퍼 전체를 훑게 됨)라면 두 번 다
    초과한다. 임계 100ms는 2Hz 주기(500ms) 대비 5배 여유다.
    """
    hal, cache, sup, plan, _ = build_world()
    stop = threading.Event()

    def feeder():
        # 실물 수신 스레드 흉내: 계속 프레임을 밀어넣는다
        frame = hal.latest_feedback()
        while not stop.is_set():
            cache.push(frame)
            time.sleep(0.0005)

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    try:
        best = None
        for attempt in range(2):
            durations = []
            for _ in range(12):
                hal.step(TICK_STEPS)
                t0 = time.perf_counter()
                sup.tick()
                durations.append(time.perf_counter() - t0)
            p95 = sorted(durations)[int(len(durations) * 0.95)]
            best = p95 if best is None else min(best, p95)
            if best < 0.1:
                break
        assert best < 0.1, f"판단 틱 p95 {best * 1e3:.1f} ms — 2Hz 루프에 과부하"
    finally:
        stop.set()
        th.join(timeout=1.0)


def test_to_arrays_cost_scales_with_window_not_buffer():
    """창을 뜨는 비용이 **버퍼 크기**에 끌려가면 안 된다.

    (0.7초 창을 뜨는데 10초 버퍼 전체를 훑던 회귀가 있었다 — 판단 틱에
    ms 단위가 얹혔다. 같은 창을 버퍼 크기만 4배로 늘려 재본다.)
    """
    hal, cache, sup, plan, _ = build_world()
    hal.step(2500)
    small = _median_ms(lambda: cache.to_arrays(0.5))
    hal.step(7500)                       # 버퍼 4배
    assert len(cache) > 9000
    large = _median_ms(lambda: cache.to_arrays(0.5))
    assert large < small * 2.5 + 1.0, (
        f"버퍼가 4배가 되자 같은 창이 {small:.2f}→{large:.2f} ms — "
        f"창이 아니라 버퍼에 비례하고 있다")


def test_snapshot_is_cheap_enough_for_1hz_ui_polling():
    """UI가 1초마다 부르는 경로 — 여기서 배열을 새로 뜨면 안 된다."""
    hal, cache, sup, plan, _ = build_world()
    for _ in range(3):
        sup.tick()
        hal.step(TICK_STEPS)
    hal.step(9000)                       # 큰 버퍼
    assert _median_ms(sup.snapshot) < 2.0


def _median_ms(fn, n=12):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


def test_push_is_storage_only_no_decision_side_effects():
    """1kHz 경로(push)는 판단 부작용이 없어야 한다 — 저장만."""
    hal, cache, sup, plan, _ = build_world()
    hal.step(500)
    calls_before = hal.play_call_count
    frame = hal.latest_feedback()
    for _ in range(5000):
        cache.push(frame)
    assert hal.play_call_count == calls_before
    assert sup.state == SupervisorState.IDLE      # push만으로 상태 변화 없음
