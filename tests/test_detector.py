"""실패 감지기 — phorce 피드백 기준 4종 + valid 마스크 전제."""

import numpy as np
import pytest

from robot_core.recovery import DetectorConfig, FailureDetector, FailureType

N = 12
RATE = 1000.0


def frames(n_sec=1.5, **cols):
    """합성 피드백 배열. cols로 특정 열 덮어쓰기."""
    n = int(n_sec * RATE)
    z = np.zeros((n, N))
    a = {
        "t": np.arange(n) / RATE,
        "position": z.copy(), "velocity": z.copy(), "current": z.copy(),
        "dob": z.copy(), "temp": np.full((n, N), 35.0),
        "valid": np.ones((n, N), dtype=bool),
        "fault": np.zeros((n, N), dtype=bool),
        "playing": np.zeros(n, dtype=bool),
    }
    a.update(cols)
    return a


def det(**cfg):
    return FailureDetector(N, DetectorConfig(**cfg))


# ------------------------------------------------------------------ IMPACT
def test_impact_fires_on_dob_run():
    a = frames()
    a["dob"][-100:-50, 3] = 5.0        # 50ms 스파이크 (이미 지나감)
    events = det(impact_dob_threshold=3.0).check(a)
    assert [e.type for e in events] == [FailureType.IMPACT]
    assert events[0].joint_idx == 3
    assert events[0].snapshot["dob_peak"] == pytest.approx(5.0)


def test_impact_survives_slow_polling():
    """2Hz 판단 루프: 폴 사이(0.5s)에 끝난 0.2s 충격도 다음 폴에서 잡혀야 한다."""
    d = det(impact_dob_threshold=3.0)
    a = frames(n_sec=2.0)
    a["dob"][1200:1400, 5] = 6.0       # t=1.2..1.4 충격
    # 폴은 t=1.0과 t=1.999에만 (충격 전 / 충격이 끝나고 0.6s 뒤)
    m1 = a["t"] <= 1.0
    assert d.check({k: v[m1] for k, v in a.items()}) == []
    events = d.check(a)                # lookback 0.7s이 커버해야 한다
    assert [e.type for e in events] == [FailureType.IMPACT]


def test_impact_single_frame_glitch_ignored():
    a = frames()
    a["dob"][-10, 2] = 50.0            # 1프레임 글리치
    assert det(impact_dob_threshold=3.0).check(a) == []


def test_impact_refractory_and_rearm():
    d = det(impact_dob_threshold=3.0, refractory_s=0.5)
    a = frames()
    a["dob"][-100:, 1] = 5.0
    assert len(d.check(a)) == 1
    assert d.check(a) == []            # 히스테리시스 — 재무장 전 재발동 금지
    calm = frames(n_sec=3.0)
    calm["t"] = calm["t"] + 5.0
    assert d.check(calm) == []         # 조용 → 재무장만
    again = frames()
    again["t"] = again["t"] + 10.0     # 시간 전진 (refractory 통과)
    again["dob"][-80:, 1] = 5.0
    assert len(d.check(again)) == 1    # 새 충격은 다시 잡힌다


def test_invalid_axis_excluded_from_impact():
    """valid=False 축의 쓰레기 dob는 판정에서 제외 — 모든 판정의 전제."""
    a = frames()
    a["dob"][-200:, 4] = 99.0
    a["valid"][:, 4] = False
    assert det(impact_dob_threshold=3.0).check(a) == []


def test_recently_invalid_axis_samples_excluded():
    a = frames()
    a["dob"][-200:, 4] = 99.0
    a["valid"][-300:, 4] = False       # 스파이크 구간이 전부 invalid
    assert det(impact_dob_threshold=3.0).check(a) == []


# ----------------------------------------------------------- PLAYBACK_STALL
def _stall_arrays(playing=True, vel=0.0, cur=3.0):
    a = frames()
    a["playing"][:] = playing
    a["velocity"][:] = vel
    a["current"][:, 2] = cur
    return a


def test_stall_fires_when_playing_frozen_and_pushing():
    events = det().check(_stall_arrays())
    assert [e.type for e in events] == [FailureType.PLAYBACK_STALL]
    assert events[0].joint_idx == 2    # 미는 축에 귀속


def test_stall_needs_playing():
    assert det().check(_stall_arrays(playing=False)) == []


def test_stall_needs_pushing_current():
    """전류가 안 흐르면 의도된 정지 동작일 수 있다 — 발화 금지."""
    assert det().check(_stall_arrays(cur=0.2)) == []


def test_stall_not_fired_while_moving():
    a = _stall_arrays()
    a["velocity"][:, 6] = 0.5          # 어느 축이든 움직이면 진행 중
    assert det().check(a) == []


# -------------------------------------------------------------- OVERHEAT
def test_overheat_fires_and_clears_in_pair():
    d = det(overheat_temp_c=70.0, overheat_release_c=62.0)
    hot = frames()
    hot["temp"][:, 8] = 75.0
    events = d.check(hot)
    assert [e.type for e in events] == [FailureType.OVERHEAT]
    assert d.overheat_active()
    assert d.check(hot) == []          # 유지 중 재발동 없음

    cool = frames()
    cool["temp"][:, 8] = 55.0
    cool["t"] = cool["t"] + 5.0
    cleared = d.check(cool)
    assert [e.type for e in cleared] == [FailureType.OVERHEAT_CLEARED]
    assert not d.overheat_active()
    assert d.check(cool) == []         # 해제도 한 번만


def test_overheat_needs_sustained_temp():
    a = frames()
    a["temp"][-3:, 8] = 90.0           # 마지막 3ms만 (창 전체가 아님)
    assert det().check(a) == []


# ------------------------------------------------------------- AXIS_FAULT
def test_axis_fault_fires_immediately():
    a = frames()
    a["fault"][-50:, 10] = True
    events = det().check(a)
    assert [e.type for e in events] == [FailureType.AXIS_FAULT]
    assert events[0].joint_idx == 10
    assert events[0].severity == 1.0


def test_axis_fault_rearm_after_clear():
    d = det(refractory_s=0.1)
    a = frames()
    a["fault"][-50:, 10] = True
    assert len(d.check(a)) == 1
    clean = frames(); clean["t"] = clean["t"] + 3.0
    assert d.check(clean) == []        # 해제 → 재무장
    again = frames(); again["t"] = again["t"] + 6.0
    again["fault"][-50:, 10] = True
    assert len(d.check(again)) == 1
