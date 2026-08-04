"""모션 카탈로그 — JSON 왕복, 적재 슬롯 대조, 자동 주석."""

import numpy as np
import pytest

from robot_core.catalog import MotionCatalog, MotionMeta, annotate_from_recording
from robot_core.hal.phorce import MockMotion, MockPhorceHAL, N_AXES
from robot_core.logging import FeedbackCache

POSE = np.zeros(N_AXES); POSE[:3] = [0.4, -0.2, 0.5]


def meta(mid=1, name="m", tags=("approach",), p0=None, p1=None, T=1.0):
    p0 = np.zeros(N_AXES) if p0 is None else p0
    p1 = POSE if p1 is None else p1
    d = p1 - p0
    n = np.linalg.norm(d)
    return MotionMeta(mid, name, list(tags), p0, d / n if n > 1e-6 else d, T)


def test_meta_validation():
    with pytest.raises(ValueError, match="1~50"):
        meta(mid=0)
    with pytest.raises(ValueError, match="shape"):
        MotionMeta(1, "x", ["a"], np.zeros(3), np.zeros(N_AXES), 1.0)


def test_catalog_basics_and_tags():
    cat = MotionCatalog([meta(1, "a", ["approach"]),
                         meta(2, "b", ["insert", "slow"])])
    assert cat.ids() == [1, 2]
    assert 1 in cat and 9 not in cat
    assert [m.name for m in cat.by_tag("slow")] == ["b"]
    assert cat.all_tags() == {"approach", "insert", "slow"}
    with pytest.raises(ValueError, match="duplicate"):
        cat.add(meta(1))


def test_json_roundtrip(tmp_path):
    cat = MotionCatalog([meta(3, "approach", ["approach", "slow"], T=1.25)])
    path = cat.to_json(tmp_path / "cat.json")
    loaded = MotionCatalog.from_json(path)
    m = loaded.get(3)
    assert m.name == "approach" and m.tags == ["approach", "slow"]
    assert m.duration_s == pytest.approx(1.25)
    assert np.allclose(m.start_pose, cat.get(3).start_pose, atol=1e-4)
    assert np.allclose(m.initial_direction, cat.get(3).initial_direction, atol=1e-4)


def test_reconcile_warns_both_directions():
    """JSON에만 있는 id → 경고+제외. 슬롯에만 있는 id → 미등록 경고."""
    cat = MotionCatalog([meta(1), meta(2, "b")])
    usable, warns = cat.reconcile({2: 1.0, 5: 0.8})   # 1은 미적재, 5는 미등록
    assert usable == {2}
    assert any("미적재" in w and "1" in w for w in warns)
    assert any("미등록" in w and "5" in w for w in warns)


def test_annotate_from_recording_extracts_metadata():
    """목에서 1회 재생 녹화 → start_pose/initial_direction/duration 자동 추출."""
    home = np.zeros(N_AXES)

    def traj(t):
        s = min(t / 1.0, 1.0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        return home + s * (POSE - home)

    hal = MockPhorceHAL({1: MockMotion(1.0, traj)})
    cache = FeedbackCache()
    hal.watch(cache.push)
    hal.play(1)
    hal.step(50)

    m = annotate_from_recording(1, "approach", ["approach"],
                                cache.to_arrays(window_sec=None))
    assert np.abs(m.start_pose - home).max() < 0.02
    assert m.duration_s == pytest.approx(1.0, abs=0.05)
    expected_dir = POSE / np.linalg.norm(POSE)
    assert float(m.initial_direction @ expected_dir) > 0.99   # 방향 일치
    assert np.linalg.norm(m.initial_direction) == pytest.approx(1.0)


def test_annotate_requires_playback_in_recording():
    cache = FeedbackCache()
    hal = MockPhorceHAL({1: MockMotion(0.5, lambda t: np.zeros(N_AXES))})
    hal.watch(cache.push)
    hal.step(50)   # 재생 없이 홀드만
    with pytest.raises(ValueError, match="재생 구간"):
        annotate_from_recording(1, "x", ["a"], cache.to_arrays(window_sec=None))
