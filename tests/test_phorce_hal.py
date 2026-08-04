"""MockPhorceHAL — play/피드백/거절/중단 계약 (실물 파사드와 동일해야 하는 것들)."""

import numpy as np
import pytest

from robot_core.hal.phorce import (
    MOTION_ID_MAX, MockMotion, MockPhorceHAL, MotionAborted, MotionBusy,
    MotionRejected, N_AXES,
)

HOME = np.zeros(N_AXES)
POSE = np.zeros(N_AXES); POSE[:3] = [0.5, -0.3, 0.6]


def minjerk(p0, p1, T):
    def fn(t):
        s = min(max(t / T, 0.0), 1.0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        return p0 + s * (p1 - p0)
    return fn


@pytest.fixture
def hal():
    return MockPhorceHAL({
        1: MockMotion(0.5, minjerk(HOME, POSE, 0.5)),
        2: MockMotion(0.3, minjerk(POSE, HOME, 0.3)),
    }, loaded_ids={1, 2})


# ---------------------------------------------------------------- 기본 계약
def test_catalog_is_loaded_slots(hal):
    assert hal.catalog() == {1: 0.5, 2: 0.3}


def test_feedback_available_immediately(hal):
    fb = hal.latest_feedback()
    assert fb is not None
    assert fb.position_rad.shape == (N_AXES,)
    assert fb.usable.all()
    assert not fb.playing


def test_play_blocking_reaches_target(hal):
    assert hal.play(1) == 1
    q = hal.latest_feedback().position_rad
    assert np.abs(q - POSE).max() < 0.05
    assert not hal.is_busy()


def test_play_async_handle_completes(hal):
    h = hal.play_async(1)
    assert not h.done() and hal.is_busy()
    hal.step(600)                      # 0.5s 재생 + 여유
    assert h.done() and not hal.is_busy()
    assert h.result() == 1
    assert h.t_end is not None and h.t_end > h.t_start


def test_feedback_playing_flag_tracks_playback(hal):
    hal.play_async(1)
    hal.step(100)
    assert hal.latest_feedback().playing
    hal.step(600)
    assert not hal.latest_feedback().playing


# ------------------------------------------------------------------ 거절들
def test_busy_rejection_while_playing(hal):
    hal.play_async(1)
    with pytest.raises(MotionBusy):
        hal.play_async(2)
    with pytest.raises(MotionBusy):
        hal.play(2)


def test_not_loaded_rejection_code_4(hal):
    with pytest.raises(MotionRejected) as ei:
        hal.play_async(9)
    assert ei.value.code == 4 and not ei.value.needs_operator


def test_out_of_range_rejection_code_3(hal):
    with pytest.raises(MotionRejected) as ei:
        hal.play_async(MOTION_ID_MAX + 1)
    assert ei.value.code == 3


@pytest.mark.parametrize("code", [12, 13])
def test_operator_rejections_persist_until_cleared(hal, code):
    hal.set_rejection(code)
    with pytest.raises(MotionRejected) as ei:
        hal.play_async(1)
    assert ei.value.needs_operator
    # "사람이 버튼 누를 때까지" — 다시 시도해도 계속 거절
    with pytest.raises(MotionRejected):
        hal.play_async(1)
    hal.clear_rejection()
    assert hal.play_async(1).motion_id == 1


# ------------------------------------------------------------------ 중단
def test_abort_leaves_arbitrary_pose_and_raises(hal):
    h = hal.play_async(1)
    hal.step(250)                      # 재생 절반
    hal.abort_playback("test")
    assert h.done()
    with pytest.raises(MotionAborted):
        h.result()
    q = hal.latest_feedback().position_rad
    # 시작도 끝도 아닌 임의 자세
    assert np.abs(q - HOME).max() > 0.05 and np.abs(q - POSE).max() > 0.05
    assert not hal.is_busy()


# -------------------------------------------------------------- 주입/피드백
def test_disturbance_appears_in_dob(hal):
    hal.inject_disturbance(2, 4.0, duration=0.1)
    hal.step(20)
    assert hal.latest_feedback().dob_a[2] == pytest.approx(4.0)
    hal.step(200)                      # 만료 후
    assert hal.latest_feedback().dob_a[2] == pytest.approx(0.0)


def test_temperature_injection_and_cooling(hal):
    hal.set_temperature(1, 80.0)
    hal.step(1)
    assert hal.latest_feedback().temp_c[1] > 75.0
    hal.step(3000)
    assert hal.latest_feedback().temp_c[1] < 80.0   # 냉각 진행


def test_valid_and_fault_injection(hal):
    hal.set_axis_valid(4, False)
    hal.inject_fault(7)
    hal.step(1)
    fb = hal.latest_feedback()
    assert not fb.usable[4]
    assert fb.fault[7]
    hal.set_axis_valid(4, True)
    hal.clear_fault(7)
    hal.step(1)
    fb = hal.latest_feedback()
    assert fb.usable[4] and not fb.fault[7]


def test_watch_callback_receives_every_frame(hal):
    frames = []
    hal.watch(frames.append)
    hal.step(100)
    assert len(frames) == 100
    assert frames[-1].seq - frames[0].seq == 99
