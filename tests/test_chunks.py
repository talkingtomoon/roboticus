"""MotionChunk / ChunkDictionary / 생성기."""

import numpy as np
import pytest

from robot_core.chunks import ChunkDictionary, MotionChunk
from robot_core.chunks import generator as G


@pytest.fixture
def direct():
    return G.min_jerk("direct", [0.0, 0.0, 0.0], [0.8, 0.0, 0.4], 1.2)


# ------------------------------------------------------------- 스플라인 표현
def test_sample_analytic_derivatives_match_numeric(direct):
    """sample()의 qd, qdd가 수치 미분과 일치해야 한다 (해석적 미분 검증)."""
    ts = np.linspace(0.01, direct.duration - 0.01, 200)
    _, qd, qdd = direct.sample(ts)
    h = 1e-5
    qp, qdp, _ = direct.sample(ts + h)
    qm, qdm, _ = direct.sample(ts - h)
    assert np.abs(qd - (qp - qm) / (2 * h)).max() < 1e-6
    assert np.abs(qdd - (qdp - qdm) / (2 * h)).max() < 1e-5


def test_min_jerk_endpoints_and_rest_boundaries(direct):
    assert np.allclose(direct.q_start, [0, 0, 0], atol=1e-12)
    assert np.allclose(direct.q_end, [0.8, 0.0, 0.4], atol=1e-12)
    assert np.allclose(direct.qd_start, 0.0, atol=1e-9)
    assert np.allclose(direct.qd_end, 0.0, atol=1e-9)


def test_min_jerk_peak_velocity_matches_theory(direct):
    """min-jerk 피크 속도 = 1.875 * 거리 / 시간 (스플라인 근사가 프로파일을 재현하는지)."""
    ts = np.linspace(0, direct.duration, 500)
    _, qd, _ = direct.sample(ts)
    assert np.abs(qd[:, 0]).max() == pytest.approx(1.875 * 0.8 / 1.2, rel=0.01)


def test_sample_clamps_out_of_range(direct):
    q_lo, _, _ = direct.sample(-1.0)
    q_hi, _, _ = direct.sample(direct.duration + 5.0)
    assert np.allclose(q_lo, direct.q_start)
    assert np.allclose(q_hi, direct.q_end)


def test_sample_scalar_and_batch_shapes(direct):
    q, qd, qdd = direct.sample(0.5)
    assert q.shape == qd.shape == qdd.shape == (3,)
    q, qd, qdd = direct.sample(np.linspace(0, 1, 7))
    assert q.shape == (7, 3)


def test_times_normalized_to_zero_start():
    c = MotionChunk.from_waypoints("x", [2.0, 2.5, 3.0], [[0.0], [1.0], [0.0]])
    assert c.times[0] == 0.0
    assert c.duration == pytest.approx(1.0)


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        MotionChunk("x", [0.0, 0.0], np.zeros((1, 2, 4)))   # 시간 비증가
    with pytest.raises(ValueError):
        MotionChunk("x", [0.0, 1.0], np.zeros((3, 2, 4)))   # 세그먼트 수 불일치
    with pytest.raises(ValueError):
        MotionChunk.from_waypoints("x", [0.0, 1.0], [[0.0], [1.0], [2.0]])


def test_from_waypoints_respects_boundary_velocities():
    v0, v1 = np.array([0.5, -0.2]), np.array([-0.1, 0.3])
    c = MotionChunk.from_waypoints(
        "v", np.linspace(0, 1, 6), np.random.default_rng(1).uniform(-1, 1, (6, 2)),
        qd_start=v0, qd_end=v1)
    assert np.allclose(c.qd_start, v0, atol=1e-9)
    assert np.allclose(c.qd_end, v1, atol=1e-9)


def test_interior_c2_continuity():
    """매듭에서 위치·속도·가속도 연속 (클램프드 큐빅은 C2)."""
    rng = np.random.default_rng(2)
    c = MotionChunk.from_waypoints("s", np.linspace(0, 2, 9), rng.uniform(-1, 1, (9, 3)))
    eps = 1e-9
    for tk in c.times[1:-1]:
        a = c.sample(tk - eps)
        b = c.sample(tk + eps)
        for x, y, tol in zip(a, b, (1e-7, 1e-6, 1e-4)):
            assert np.abs(x - y).max() < tol


# ------------------------------------------------------------------- I/O
def test_npz_roundtrip(tmp_path, direct):
    direct.tags = ["approach", "demo"]
    path = direct.save(tmp_path / "direct.npz")
    loaded = MotionChunk.load(path)
    assert loaded.name == "direct"
    assert loaded.tags == ["approach", "demo"]
    ts = np.linspace(0, direct.duration, 50)
    for a, b in zip(direct.sample(ts), loaded.sample(ts)):
        assert np.allclose(a, b, atol=1e-15)


def test_dictionary_directory_roundtrip(tmp_path, direct):
    d = ChunkDictionary([direct, G.retreat("retreat", [0.8, 0, 0.4], [0, 0, 0], 1.0)])
    d.save_dir(tmp_path / "dict")
    loaded = ChunkDictionary.load_dir(tmp_path / "dict")
    assert loaded.names() == ["direct", "retreat"]
    assert "retreat" in loaded.get("retreat").tags
    assert loaded.by_tag("retreat")[0].name == "retreat"
    with pytest.raises(KeyError):
        loaded.get("ghost")
    with pytest.raises(ValueError):
        loaded.add(direct)  # 중복 이름


# ---------------------------------------------------------------- 생성기
def test_via_points_passes_through_vias():
    via = np.array([0.4, 0.35, 0.2])
    c = G.via_points("detour", [0, 0, 0], [via], [0.8, 0, 0.4], 1.5)
    ts = np.linspace(0, c.duration, 400)
    q, _, _ = c.sample(ts)
    assert np.linalg.norm(q - via, axis=1).min() < 0.02


def test_time_scaled_keeps_geometry():
    base = G.min_jerk("b", [0.0], [1.0], 1.0)
    slow = G.time_scaled(base, 2.0)
    assert slow.duration == pytest.approx(2.0)
    ts = np.linspace(0, 1, 50)
    qb, qdb, _ = base.sample(ts)
    qs, qds, _ = slow.sample(2.0 * ts)
    assert np.allclose(qb, qs, atol=1e-9)          # 같은 경로
    assert np.allclose(qdb, 2.0 * qds, atol=1e-9)  # 속도는 절반


def test_amplitude_scaled_about_start():
    base = G.min_jerk("b", [0.2, -0.1], [1.2, 0.9], 1.0)
    half = G.amplitude_scaled(base, 0.5)
    assert np.allclose(half.q_start, base.q_start, atol=1e-9)
    assert np.allclose(half.q_end, base.q_start + 0.5 * (base.q_end - base.q_start),
                       atol=1e-9)


def test_with_detour_bumps_and_returns():
    base = G.min_jerk("b", [0, 0, 0], [0.8, 0, 0], 1.0)
    off = np.array([0.0, 0.3, 0.0])
    dt = G.with_detour(base, t_via=0.5, offset=off)
    assert np.allclose(dt.q_start, base.q_start, atol=1e-6)
    assert np.allclose(dt.q_end, base.q_end, atol=1e-6)
    ts = np.linspace(0, 1, 200)
    q, _, _ = dt.sample(ts)
    assert q[:, 1].max() > 0.25   # 실제로 우회했다
    assert "detour" in dt.tags


def test_retreat_absorbs_initial_velocity():
    v0 = np.array([1.5, -0.5, 0.0])
    c = G.retreat("r", [0.5, 0.3, 0.1], [0, 0, 0], 0.8, qd_from=v0)
    assert np.allclose(c.qd_start, v0, atol=1e-9)
    assert np.allclose(c.q_end, 0.0, atol=1e-9)
    assert np.allclose(c.qd_end, 0.0, atol=1e-9)


# ---------------------------------------------------------------- 한계 검사
def test_validate_chunk_catches_violations():
    c = G.min_jerk("fast", [0.0], [2.0], 0.5)  # 피크 속도 = 1.875*2/0.5 = 7.5
    problems = G.validate_chunk(c, joint_limits=np.array([[-1.0, 1.5]]), qd_max=5.0)
    assert len(problems) == 2
    assert any("position" in p for p in problems)
    assert any("speed" in p for p in problems)
    assert G.validate_chunk(c, joint_limits=np.array([[-3.0, 3.0]]), qd_max=10.0) == []


def test_generators_raise_on_limit_violation():
    with pytest.raises(ValueError, match="violates limits"):
        G.min_jerk("fast", [0.0], [2.0], 0.5, qd_max=5.0)
    with pytest.raises(ValueError, match="violates limits"):
        G.with_detour(G.min_jerk("b", [0.0], [0.5], 1.0), t_via=0.5,
                      offset=np.array([5.0]), joint_limits=np.array([[-1.0, 1.0]]))
