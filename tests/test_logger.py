"""링버퍼 로거 — 창 유지, 통계, 그리고 dump_text 스냅샷.

스냅샷 재생성:
    python -m tests.test_logger --update-snapshot
포맷을 일부러 바꿨을 때만 재생성하고, diff를 눈으로 확인한 뒤 커밋할 것.
"""

from pathlib import Path

import numpy as np
import pytest

from robot_core.hal import JointCommand, MockRobotHAL
from robot_core.logging import RingLogger

from .conftest import pd_command, run_steps

SNAPSHOT = Path(__file__).parent / "snapshots" / "dump_text.txt"


def build_scenario() -> RingLogger:
    """벽시계를 전혀 쓰지 않는 결정적 시나리오.

    0.000~0.099 s : 원점 홀드
    0.100 s       : 관절 1에 15 Nm 외란 30 ms 주입 (임계치 10 Nm 초과)
    0.100~0.199 s : 계속 홀드하며 회복
    """
    hal = MockRobotHAL(n_joints=3, dt=1e-3, torque_limit=20.0)
    logger = RingLogger(n_joints=3, window_sec=1.0, torque_threshold=10.0)
    cmd = pd_command(hal, np.zeros(3), kp=30.0, kd=1.0)

    for _ in range(100):
        hal.send_command(cmd)
        state = hal.read_state()
        logger.log(state, cmd=cmd, loop_dt=8e-5, wall_time=state.timestamp)

    hal.inject_disturbance(joint_idx=1, torque=15.0, duration=0.03)
    logger.mark_event("disturbance 15 Nm / 30 ms injected on joint 1", t=hal.t)

    for i in range(100):
        hal.send_command(cmd)
        state = hal.read_state()
        logger.log(state, cmd=cmd, loop_dt=8e-5, overrun=(i == 42), wall_time=state.timestamp)

    return logger


# ------------------------------------------------------------------ 스냅샷
def test_dump_text_matches_snapshot():
    actual = build_scenario().dump_text()
    assert SNAPSHOT.exists(), f"스냅샷 없음. `python -m tests.test_logger --update-snapshot` 실행"
    expected = SNAPSHOT.read_text(encoding="utf-8")
    assert actual == expected, (
        "dump_text 출력이 스냅샷과 다르다. 포맷을 의도적으로 바꿨다면 "
        "`python -m tests.test_logger --update-snapshot` 로 갱신할 것.\n\n"
        f"--- actual ---\n{actual}"
    )


def test_dump_text_is_deterministic():
    assert build_scenario().dump_text() == build_scenario().dump_text()


def test_dump_text_is_human_readable():
    """LLM에 그대로 먹일 수 있는 수준인지 — 필수 섹션과 단위가 다 있는지."""
    text = build_scenario().dump_text()
    for section in (
        "ROBOT STATE DUMP",
        "[TIMING]",
        "[PER-JOINT TORQUE]",
        "[PER-JOINT TRACKING]",
        "[THRESHOLD EXCURSIONS]",
        "[EVENTS]",
        "[LAST SAMPLE]",
    ):
        assert section in text, f"{section} 섹션이 없다"

    assert "units:" in text and "[Nm]" in text
    assert "time span:" in text
    # 관절별로 한 줄씩은 있어야 한다
    for j in range(3):
        assert f"\n  {j:3d} |" in text
    # 한 줄이 너무 길면 터미널/프롬프트에서 읽기 나쁘다
    assert max(len(line) for line in text.splitlines()) <= 100


def test_dump_reports_the_disturbance_excursion():
    text = build_scenario().dump_text()
    assert "EXCEEDED" in text
    assert "joint 1:" in text
    assert "disturbance 15 Nm / 30 ms injected on joint 1" in text
    # 외란이 없던 관절은 초과 구간이 없어야 한다
    assert "joint 0:" not in text
    assert "joint 2:" not in text


# -------------------------------------------------------------- 링버퍼 동작
def test_ring_buffer_drops_old_samples():
    hal = MockRobotHAL(n_joints=2, dt=1e-3)
    logger = RingLogger(n_joints=2, window_sec=0.05)
    run_steps(hal, pd_command(hal, np.zeros(2)), 500, logger=logger)

    assert 45 <= len(logger) <= 55, f"창 밖 샘플이 안 버려졌다: {len(logger)}"
    t = logger.to_arrays()["t"]
    assert t[-1] - t[0] <= 0.05 + 1e-9


def test_max_samples_is_a_hard_cap():
    hal = MockRobotHAL(n_joints=1, dt=1e-3)
    logger = RingLogger(n_joints=1, window_sec=1e9, max_samples=30)
    run_steps(hal, pd_command(hal, np.zeros(1)), 200, logger=logger)
    assert len(logger) == 30


def test_window_sec_argument_narrows_the_dump():
    logger = build_scenario()
    full = logger.to_arrays()["t"]
    narrow = logger.to_arrays(window_sec=0.02)["t"]
    assert len(narrow) < len(full)
    assert narrow[-1] - narrow[0] <= 0.02 + 1e-9

    header = logger.dump_text(window_sec=0.02).split("\n")[1]
    assert f"{len(narrow)} samples" in header
    assert float(header.split("window")[1].split("s")[0]) <= 0.02 + 1e-9


def test_clear_empties_everything():
    logger = build_scenario()
    logger.clear()
    assert len(logger) == 0
    assert "no samples" in logger.dump_text()


def test_empty_logger_dump_does_not_crash():
    assert "no samples" in RingLogger(n_joints=3).dump_text()


# ------------------------------------------------------------------ 통계
def test_to_arrays_shapes():
    logger = build_scenario()
    d = logger.to_arrays()
    n = len(logger)
    assert d["t"].shape == (n,)
    assert d["q"].shape == d["qd"].shape == d["tau"].shape == d["q_des"].shape == (n, 3)
    assert d["overrun"].sum() == 1


def test_torque_stats_match_raw_data():
    logger = build_scenario()
    d = logger.to_arrays()
    text = logger.dump_text()
    peak = np.abs(d["tau"][:, 1]).max()
    assert peak > 14.0  # 외란 15 Nm이 실제로 기록돼 있다
    assert f"{peak:7.3f}" in text or f"{-peak:7.3f}" in text


def test_threshold_none_is_reported_not_crashed():
    hal = MockRobotHAL(n_joints=2, dt=1e-3)
    logger = RingLogger(n_joints=2, window_sec=1.0, torque_threshold=None)
    run_steps(hal, pd_command(hal, np.zeros(2)), 20, logger=logger)
    text = logger.dump_text()
    assert "no-threshold" in text
    assert "threshold not set" in text


def test_for_hal_uses_torque_limits():
    hal = MockRobotHAL(n_joints=3, torque_limit=[10.0, 20.0, 30.0])
    logger = RingLogger.for_hal(hal, threshold_frac=0.5)
    assert np.allclose(logger.torque_threshold, [5.0, 10.0, 15.0])


def test_threshold_shape_is_validated():
    with pytest.raises(ValueError):
        RingLogger(n_joints=3, torque_threshold=[1.0, 2.0])


def test_logging_without_command_still_works():
    hal = MockRobotHAL(n_joints=2, dt=1e-3)
    logger = RingLogger(n_joints=2, window_sec=1.0)
    for _ in range(10):
        hal.send_command(JointCommand.hold(np.zeros(2), kp=1.0))
        logger.log(hal.read_state())
    text = logger.dump_text()
    assert "no commands logged" in text


def test_events_outside_window_are_filtered():
    logger = build_scenario()
    text = logger.dump_text(window_sec=0.01)  # 외란 주입 시점보다 뒤쪽만
    assert "[EVENTS]\n  (none)" in text


def test_excursion_list_is_truncated():
    """짧은 외란을 여러 번 주면 초과 구간이 여러 개 잡히고, 상위만 표시된다."""
    hal = MockRobotHAL(n_joints=1, dt=1e-3, torque_limit=50.0)
    logger = RingLogger(n_joints=1, window_sec=10.0, torque_threshold=5.0)
    cmd = pd_command(hal, np.zeros(1), kp=1.0, kd=0.1)
    for k in range(6):
        hal.inject_disturbance(0, 10.0 + k, 0.005)
        run_steps(hal, cmd, 5, logger=logger)
        run_steps(hal, cmd, 5, logger=logger)

    text = logger.dump_text(max_excursions=3)
    assert text.count("  joint 0: t=") == 3
    assert "more (peak" in text


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--update-snapshot" in sys.argv:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(build_scenario().dump_text(), encoding="utf-8")
        print(f"snapshot updated: {SNAPSHOT}")
    else:
        print(build_scenario().dump_text())
