"""카탈로그·계획 정적 검증 — 교시 지침의 자동 검사화."""

import numpy as np
import pytest

from robot_core.catalog import MotionCatalog, MotionMeta
from robot_core.hal.phorce import N_AXES
from scripts.validate_catalog import validate

HOME = np.zeros(N_AXES)
POSE = np.zeros(N_AXES); POSE[:3] = [0.5, -0.3, 0.6]


def meta(mid, name, tags, p0=None, direction=(0, 1.0), T=1.0, end=None):
    p0 = HOME if p0 is None else p0
    d = np.zeros(N_AXES)
    if direction is not None:
        d[direction[0]] = direction[1]
        n = np.linalg.norm(d)
        d = d / n if n else d
    # 건강한 현대 카탈로그는 end_pose를 가진다 (annotate가 자동 추출)
    end = (p0 if end is None else end)
    return MotionMeta(mid, name, list(tags), p0, d, T, end_pose=end)


def good_catalog():
    return MotionCatalog([
        meta(1, "approach", ["approach"], HOME),
        meta(2, "insert", ["insert"], HOME, (3, 1.0)),
        meta(3, "rest", ["rest"], HOME, None),
        meta(4, "ret_left", ["retreat"], HOME, (1, +1.0)),
        meta(5, "ret_right", ["retreat"], HOME, (1, -1.0)),
    ])


PLAN = ["approach", "insert"]


def test_healthy_catalog_passes():
    errors, warnings = validate(good_catalog(), PLAN)
    assert errors == []
    assert warnings == []


def test_missing_plan_tag_is_error():
    errors, _ = validate(good_catalog(), ["approach", "juggle"])
    assert any("juggle" in e for e in errors)


def test_missing_rest_or_retreat_is_error():
    cat = MotionCatalog([meta(1, "a", ["approach"], HOME)])
    errors, _ = validate(cat, ["approach"])
    assert any("'rest'" in e for e in errors)
    assert any("'retreat'" in e for e in errors)


def test_tag_typo_suspicion_warns():
    cat = good_catalog()
    cat.add(meta(6, "typo", ["aproach"], HOME))       # approach 오타
    _, warnings = validate(cat, PLAN)
    assert any("오타 의심" in w and "aproach" in w for w in warnings)


def test_one_sided_retreat_warns():
    cat = MotionCatalog([
        meta(1, "approach", ["approach"], HOME),
        meta(2, "insert", ["insert"], HOME, (3, 1.0)),
        meta(3, "rest", ["rest"], HOME, None),
        meta(4, "ret_a", ["retreat"], HOME, (1, +1.0)),
        meta(5, "ret_b", ["retreat"], HOME, (1, +0.9)),   # 같은 방향뿐
    ])
    _, warnings = validate(cat, PLAN)
    assert any("같은 반구" in w for w in warnings)


def test_long_motion_warns_responsiveness():
    cat = good_catalog()
    cat.add(meta(7, "marathon", ["approach"], HOME, (0, 1.0), T=5.0))
    _, warnings = validate(cat, PLAN, responsiveness_s=2.0)
    assert any("marathon" in w and "5.0s" in w for w in warnings)


def test_zero_direction_non_rest_is_error():
    cat = good_catalog()
    cat.add(meta(8, "no_dir", ["insert"], HOME, None))
    errors, _ = validate(cat, PLAN)
    assert any("no_dir" in e and "initial_direction" in e for e in errors)


def test_step_linkage_uses_end_pose_precisely():
    """정밀 연결 검사: 이전 단계 end_pose ↔ 다음 단계 start_pose.

    approach가 POSE에서 끝나고 insert가 POSE에서 시작하면 —
    시작자세끼리는 멀어도(근사 검사면 경고) 정밀 검사는 통과해야 한다.
    """
    cat = MotionCatalog([
        meta(1, "approach", ["approach"], HOME, end=POSE),   # HOME→POSE
        meta(2, "insert", ["insert"], POSE, (3, 1.0)),        # POSE에서 시작
        meta(3, "rest", ["rest"], HOME, None),
        meta(4, "ret_l", ["retreat"], HOME, (1, +1.0)),
        meta(5, "ret_r", ["retreat"], HOME, (1, -1.0)),
    ])
    _, warnings = validate(cat, ["approach", "insert"], entry_max_dist=0.5)
    assert not any("'approach'→'insert'" in w for w in warnings)


def test_step_linkage_warns_on_real_gap():
    far = HOME.copy(); far[:3] = 5.0
    cat = MotionCatalog([
        meta(1, "approach", ["approach"], HOME, end=POSE),
        meta(2, "insert_far", ["insert"], far, (3, 1.0)),     # 어디서도 못 이음
        meta(3, "rest", ["rest"], HOME, None),
        meta(4, "ret_l", ["retreat"], HOME, (1, +1.0)),
        meta(5, "ret_r", ["retreat"], HOME, (1, -1.0)),
    ])
    _, warnings = validate(cat, ["approach", "insert"], entry_max_dist=0.8)
    assert any("'approach'→'insert'" in w and "종료↔시작" in w for w in warnings)


def test_missing_end_pose_falls_back_to_approx_with_warning():
    """구버전 주석(end_pose 없음): 근사 검사로 폴백 + 재주석 권장 경고."""
    cat = MotionCatalog([
        MotionMeta(1, "approach", ["approach"], HOME,
                   np.eye(N_AXES)[0], 1.0),                   # end_pose=None
        meta(2, "insert", ["insert"], HOME, (3, 1.0)),
        meta(3, "rest", ["rest"], HOME, None),
        meta(4, "ret_l", ["retreat"], HOME, (1, +1.0)),
        meta(5, "ret_r", ["retreat"], HOME, (1, -1.0)),
    ])
    _, warnings = validate(cat, ["approach", "insert"])
    assert any("end_pose 주석 없음" in w for w in warnings)


def test_empty_catalog_is_error():
    errors, _ = validate(MotionCatalog(), PLAN)
    assert any("비었다" in e for e in errors)
