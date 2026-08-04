"""모션 선택기 — 채점 4항목, 하드 필터 2종, 베이스라인 인터페이스."""

import numpy as np
import pytest

from robot_core.catalog import MotionCatalog, MotionMeta
from robot_core.hal.phorce import N_AXES
from robot_core.switching import (
    FirstMotionSelector, MissionPlan, MotionSelector, RandomMotionSelector,
    SelectorWeights,
)

HOME = np.zeros(N_AXES)
POSE = np.zeros(N_AXES); POSE[:3] = [0.5, -0.3, 0.6]
VALID = np.ones(N_AXES, dtype=bool)
NO_DOB = np.zeros(N_AXES)


def meta(mid, name, tags, p0, direction=None, T=1.0):
    d = np.zeros(N_AXES) if direction is None else np.asarray(direction, float)
    return MotionMeta(mid, name, list(tags), p0, d, T)


def dirvec(*pairs):
    d = np.zeros(N_AXES)
    for axis, val in pairs:
        d[axis] = val
    return d / (np.linalg.norm(d) or 1.0)


@pytest.fixture
def catalog():
    return MotionCatalog([
        meta(1, "approach", ["approach"], HOME, dirvec((0, 1.0))),
        meta(2, "approach_slow", ["approach", "slow"], HOME, dirvec((0, 1.0)), T=2.0),
        meta(3, "retreat_left", ["retreat"], POSE, dirvec((1, +1.0))),
        meta(4, "retreat_right", ["retreat"], POSE, dirvec((1, -1.0))),
        meta(5, "far_motion", ["approach"], POSE + 2.0, dirvec((0, 1.0))),
        meta(6, "rest", ["rest"], POSE, None),
    ])


def plan():
    return MissionPlan(["approach", "insert"])


# ------------------------------------------------------------------ 채점
def test_tag_match_wins(catalog):
    sel = MotionSelector(catalog, catalog.ids())
    rep = sel.select(HOME, VALID, NO_DOB, "approach", plan())
    assert rep.best_id in (1, 2)
    by = {r.motion_id: r for r in rep.rows}
    assert by[1].tag_cost == 0.0
    assert by[3].excluded or by[3].tag_cost == 1.0


def test_entry_distance_prefers_nearby_start(catalog):
    sel = MotionSelector(catalog, catalog.ids(), entry_max_dist=5.0)
    rep = sel.select(HOME, VALID, NO_DOB, "approach", plan())
    by = {r.motion_id: r for r in rep.rows}
    assert by[1].entry_cost < by[5].entry_cost
    assert rep.best_id in (1, 2)


def test_compliance_prefers_yielding_direction(catalog):
    """외란이 축1을 +로 민다: +방향으로 비켜서는 retreat_left가 이겨야 한다."""
    sel = MotionSelector(catalog, catalog.ids())
    dob = np.zeros(N_AXES); dob[1] = 5.0
    rep = sel.select(POSE, VALID, dob, "retreat", plan())
    by = {r.motion_id: r for r in rep.rows}
    assert by[3].compliance_cost == pytest.approx(0.0, abs=1e-6)   # 순응
    assert by[4].compliance_cost == pytest.approx(1.0, abs=1e-6)   # 정면 대항
    assert rep.best_id == 3


def test_progress_cost_penalizes_off_plan(catalog):
    sel = MotionSelector(catalog, catalog.ids())
    rep = sel.select(POSE, VALID, NO_DOB, "retreat", plan())
    by = {r.motion_id: r for r in rep.rows}
    # retreat은 계획(approach 다음)을 전진 못 시키므로 progress가 더 비싸다
    assert by[3].progress_cost > 0.0


def test_modifier_slow_bonus(catalog):
    sel = MotionSelector(catalog, catalog.ids())
    normal = sel.select(HOME, VALID, NO_DOB, "approach", plan())
    slow = sel.select(HOME, VALID, NO_DOB, "approach", plan(), modifiers=("slow",))
    assert normal.best_id == 1          # 동률이면 slow 없는 쪽이 progress 동일 → 1
    assert slow.best_id == 2            # slow modifier가 2를 밀어올린다


# --------------------------------------------------------------- 하드 필터
def test_unloaded_id_hard_filtered(catalog):
    """미적재 id는 점수 이전에 제외 — 재생하면 코드 4 거절이므로."""
    sel = MotionSelector(catalog, loaded_ids={2, 3, 4, 5, 6})   # 1 미적재
    rep = sel.select(HOME, VALID, NO_DOB, "approach", plan())
    by = {r.motion_id: r for r in rep.rows}
    assert by[1].excluded and "not loaded" in by[1].excluded
    assert rep.best_id == 2


def test_entry_distance_hard_filter(catalog):
    """시작 자세가 먼 모션은 제외 — 엉뚱한 자세에서 재생 방지."""
    sel = MotionSelector(catalog, catalog.ids(), entry_max_dist=0.8)
    rep = sel.select(HOME, VALID, NO_DOB, "approach", plan())
    by = {r.motion_id: r for r in rep.rows}
    assert by[5].excluded and "entry dist" in by[5].excluded


def test_all_filtered_returns_none(catalog):
    sel = MotionSelector(catalog, loaded_ids=set())
    rep = sel.select(HOME, VALID, NO_DOB, "approach", plan())
    assert rep.best_id is None


def test_invalid_axes_ignored_in_entry_distance(catalog):
    """valid=False 축의 (엉터리) 위치값은 진입 거리 계산에서 제외돼야 한다."""
    sel = MotionSelector(catalog, catalog.ids(), entry_max_dist=0.8)
    pos = HOME.copy()
    pos[7] = 99.0                      # 고장 축의 쓰레기 값
    valid = VALID.copy(); valid[7] = False
    rep = sel.select(pos, valid, NO_DOB, "approach", plan())
    assert rep.best_id in (1, 2)       # 99.0이 들어갔으면 전부 필터됐을 것


# ---------------------------------------------------------------- 계획
def test_mission_plan_advance():
    p = MissionPlan(["approach", "insert"])
    m_app = meta(1, "a", ["approach"], HOME)
    m_ins = meta(2, "i", ["insert"], HOME)
    assert p.next_tag == "approach"
    assert not p.advance_if_matches(m_ins)     # 순서 안 맞음
    assert p.advance_if_matches(m_app)
    assert p.next_tag == "insert"
    assert p.advance_if_matches(m_ins)
    assert p.done and p.next_tag is None


# ------------------------------------------------------------- 베이스라인
def test_baselines_share_interface_and_loaded_filter(catalog):
    for cls in (RandomMotionSelector, FirstMotionSelector):
        sel = (cls(catalog, {3, 4}, seed=1) if cls is RandomMotionSelector
               else cls(catalog, {3, 4}))
        rep = sel.select(HOME, VALID, NO_DOB, "approach", plan())
        assert rep.best_id in (3, 4)           # 적재 필터만 공유
        by = {r.motion_id: r for r in rep.rows}
        assert by[1].excluded                  # 미적재는 베이스라인도 제외


def test_first_selector_is_slot_order(catalog):
    sel = FirstMotionSelector(catalog, catalog.ids())
    rep = sel.select(POSE, VALID, NO_DOB, "retreat", plan())
    assert rep.best_id == 1                    # 자세도 태그도 무시 — 약점 그 자체
