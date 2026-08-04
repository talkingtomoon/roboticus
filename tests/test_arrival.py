"""⑦ end_pose — 도달 판정, MOTION_INCOMPLETE, 계획 미전진, 하위호환."""

import numpy as np
import pytest

from robot_core.catalog import MotionCatalog, MotionMeta, annotate_from_recording
from robot_core.hal.phorce import MockMotion, MockPhorceHAL, N_AXES
from robot_core.integration.scenarios import (
    POSE_A, build_world, run_ticks,
)
from robot_core.logging import FeedbackCache
from robot_core.supervisor import SupervisorState


# ---------------------------------------------------------- 메타/주석/JSON
def test_meta_end_pose_roundtrip(tmp_path):
    m = MotionMeta(1, "a", ["approach"], np.zeros(N_AXES),
                   np.zeros(N_AXES), 1.0, end_pose=POSE_A)
    cat = MotionCatalog([m])
    loaded = MotionCatalog.from_json(cat.to_json(tmp_path / "c.json"))
    assert np.allclose(loaded.get(1).end_pose, POSE_A, atol=1e-4)


def test_legacy_json_without_end_pose_loads_as_none(tmp_path):
    import json
    payload = {"3": {"name": "old", "tags": ["approach"],
                     "start_pose": [0.0] * N_AXES,
                     "initial_direction": [0.0] * N_AXES,
                     "duration_s": 1.0}}          # end_pose 없음 (구버전)
    p = tmp_path / "old.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    m = MotionCatalog.from_json(p).get(3)
    assert m.end_pose is None                     # 하위호환


def test_annotate_extracts_end_pose():
    home = np.zeros(N_AXES)

    def traj(t):
        s = min(t / 1.0, 1.0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        return home + s * (POSE_A - home)

    hal = MockPhorceHAL({1: MockMotion(1.0, traj)})
    cache = FeedbackCache()
    hal.watch(cache.push)
    hal.play(1)
    hal.step(30)
    m = annotate_from_recording(1, "a", ["approach"],
                                cache.to_arrays(window_sec=None))
    assert m.end_pose is not None
    assert np.abs(m.end_pose - POSE_A).max() < 0.05


# ------------------------------------------------------ 도달 판정 (감독 루프)
def test_arrival_ok_advances_plan_and_sets_badge():
    hal, cache, sup, plan, _ = build_world()
    run_ticks(hal, sup, 6)
    assert plan.cursor >= 1                       # 정상 도달 → 전진
    assert sup.last_arrival["status"] == "ok"
    assert sup.snapshot()["last_arrival"]["status"] == "ok"


def test_incomplete_blocks_plan_advance_and_fires_event():
    """완료 ≠ 성공: jam으로 목표 미도달 → MOTION_INCOMPLETE + 계획 미전진."""
    hal, cache, sup, plan, _ = build_world()

    def on_tick(k):
        if k == 1:                 # approach(1.0s) 재생 초반에 축 0을 잼
            hal.inject_jam(0)
    run_ticks(hal, sup, 5, on_tick)

    texts = [e.text for e in cache.events()]
    assert any("DETECTED MOTION_INCOMPLETE" in t for t in texts), \
        "\n".join(texts)
    assert plan.cursor == 0, "미도달인데 계획이 전진했다"
    assert sup.last_arrival["status"] == "incomplete"
    assert sup.last_arrival["err_rad"] > sup.cfg.incomplete_err_rad
    # 브레이커 집계에 포함 (미도달도 실패)
    assert any(k[0] == "MOTION_INCOMPLETE" for k in sup._fail_times)

    hal.clear_jam(0)
    run_ticks(hal, sup, 18)                       # 재시도 → 완주
    assert plan.done


def test_repeated_incomplete_trips_breaker():
    hal, cache, sup, plan, _ = build_world(plan_tags=("approach",) * 20)
    hal.inject_jam(0)                             # 영구 잼 — 계속 미도달
    run_ticks(hal, sup, 40)
    assert sup.state == SupervisorState.HALTED
    assert "MOTION_INCOMPLETE" in sup.halt_reason


def test_no_end_pose_means_unknown_not_incomplete():
    """구버전 메타(end_pose=None)는 판정불가 — 미도달로 오판하면 안 된다."""
    hal, cache, sup, plan, _ = build_world()
    for m in sup.catalog.all():
        m.end_pose = None
    hal.inject_jam(0)                             # 물리적으로는 실패하지만
    run_ticks(hal, sup, 5)
    assert sup.last_arrival["status"] == "unknown"
    assert not any("MOTION_INCOMPLETE" in e.text for e in cache.events())
