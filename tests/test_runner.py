"""제어 루프 러너 — 주파수/지터 측정, 오버런 카운트, 콜백 규약.

타이밍 테스트는 CI/노트북 부하에 따라 흔들리므로 허용치를 넉넉히 잡았다.
"목표 주파수를 대충 지키는지"가 확인 대상이지 하드 실시간 보증이 아니다.
"""

import time

import numpy as np
import pytest

from robot_core import ControlLoopRunner, JointCommand, MockRobotHAL, RingLogger

from .conftest import pd_command


@pytest.fixture
def runner(hal):
    r = ControlLoopRunner(hal, rate_hz=1000.0)
    r.set_policy(lambda state: pd_command(hal, np.zeros(hal.n_joints)))
    return r


# ---------------------------------------------------------------- 기본 규약
def test_run_requires_policy(hal):
    r = ControlLoopRunner(hal, rate_hz=100.0)
    with pytest.raises(RuntimeError, match="policy"):
        r.run(n_steps=10)


def test_set_policy_rejects_non_callable(hal):
    r = ControlLoopRunner(hal, rate_hz=100.0)
    with pytest.raises(TypeError):
        r.set_policy("not a function")


def test_run_requires_a_stop_condition(runner):
    with pytest.raises(ValueError):
        runner.run()


def test_policy_receives_state_and_result_reaches_hal(hal):
    seen = []

    def policy(state):
        seen.append(state)
        return pd_command(hal, np.full(hal.n_joints, 0.2))

    r = ControlLoopRunner(hal, rate_hz=500.0)
    r.set_policy(policy)
    r.run(n_steps=50)

    assert len(seen) == 50
    assert seen[0].timestamp < seen[-1].timestamp
    assert hal.step_count == 50
    assert hal.read_state().q[0] > 0.0  # 지령이 실제로 반영됐다


def test_n_steps_is_exact(runner):
    stats = runner.run(n_steps=137)
    assert stats.n_steps == 137


def test_stop_from_inside_policy(hal):
    r = ControlLoopRunner(hal, rate_hz=1000.0)
    calls = {"n": 0}

    def policy(state):
        calls["n"] += 1
        if calls["n"] == 25:
            r.stop()
        return JointCommand.hold(state.q, kp=0.0, kd=0.0)

    r.set_policy(policy)
    stats = r.run(n_steps=10_000)
    assert stats.n_steps == 25


# ------------------------------------------------------------ 주파수 / 지터
@pytest.mark.parametrize("rate_hz", [100.0, 500.0])
def test_achieved_rate_is_close_to_target(hal, rate_hz):
    r = ControlLoopRunner(hal, rate_hz=rate_hz)
    r.set_policy(lambda s: pd_command(hal, np.zeros(hal.n_joints)))
    stats = r.run(duration_sec=0.3)

    assert stats.achieved_hz == pytest.approx(rate_hz, rel=0.15), stats.summary()
    assert stats.dt_mean_ms == pytest.approx(1000.0 / rate_hz, rel=0.15), stats.summary()
    assert stats.overrun_rate < 0.20, stats.summary()


def test_1khz_is_roughly_achievable(hal):
    """1kHz 목표. 데스크톱 OS라 하드 실시간은 아니고 '대충' 맞으면 통과."""
    r = ControlLoopRunner(hal, rate_hz=1000.0)
    r.set_policy(lambda s: pd_command(hal, np.zeros(hal.n_joints)))
    stats = r.run(duration_sec=0.5)

    assert stats.achieved_hz > 700.0, f"1kHz 목표에 크게 못 미침:\n{stats.summary()}"
    assert stats.dt_p95_ms < 5.0, f"p95 주기가 5ms 초과:\n{stats.summary()}"


def test_stats_fields_are_consistent(runner):
    stats = runner.run(n_steps=200)
    assert stats.target_hz == 1000.0
    assert stats.periods_ms.size == stats.n_steps - 1
    assert stats.dt_min_ms <= stats.dt_p50_ms <= stats.dt_max_ms
    assert stats.dt_p50_ms <= stats.dt_p95_ms <= stats.dt_p99_ms <= stats.dt_max_ms
    assert stats.dt_std_ms >= 0.0
    assert 0.0 <= stats.overrun_rate <= 1.0
    assert stats.compute_p50_ms <= stats.compute_p95_ms <= stats.compute_max_ms
    assert stats.compute_mean_ms <= stats.compute_max_ms
    assert runner.last_stats is stats


def test_summary_is_readable(runner):
    text = runner.run(n_steps=100).summary()
    for token in ("target", "achieved", "jitter", "overruns", "p95"):
        assert token in text


# ------------------------------------------------------------------ 오버런
def test_slow_policy_is_counted_as_overrun(hal):
    """주기보다 오래 걸리는 정책은 전부 오버런으로 잡혀야 한다."""

    def slow_policy(state):
        time.sleep(0.005)  # 5ms > 1ms 주기
        return JointCommand.hold(state.q, kp=0.0, kd=0.0)

    r = ControlLoopRunner(hal, rate_hz=1000.0)
    r.set_policy(slow_policy)
    stats = r.run(n_steps=20)

    assert stats.overruns == 20
    assert stats.overrun_rate == 1.0
    assert stats.achieved_hz < 400.0


def test_fast_policy_does_not_overrun_on_compute(hal):
    """가벼운 정책이면 계산 시간이 주기(5ms)에 한참 못 미쳐야 한다.

    OS 선점 때문에 compute_max는 가끔 주기를 통째로 넘긴다(하드 실시간이 아니다).
    그래서 max가 아니라 p95를 본다.
    """
    r = ControlLoopRunner(hal, rate_hz=200.0)
    r.set_policy(lambda s: JointCommand.hold(s.q, kp=0.0, kd=0.0))
    stats = r.run(n_steps=100)

    assert stats.compute_p95_ms < 5.0, stats.summary()
    assert stats.compute_mean_ms < 1.0, stats.summary()
    assert stats.overrun_rate < 0.25, stats.summary()


# ------------------------------------------------------------- 정책 예외 처리
def test_policy_exception_propagates_and_robot_is_softened(hal):
    def bad_policy(state):
        raise RuntimeError("정책 버그")

    r = ControlLoopRunner(hal, rate_hz=1000.0, on_error="raise")
    r.set_policy(bad_policy)
    with pytest.raises(RuntimeError, match="정책 버그"):
        r.run(n_steps=10)

    # 예외로 빠져나가기 전에 감쇠 지령이 나갔는지 (하드웨어를 그냥 놓지 않는다)
    assert hal.step_count >= 1
    assert np.allclose(hal.read_state().tau_measured, 0.0)


def test_on_error_hold_keeps_loop_alive(hal):
    calls = {"n": 0}

    def flaky_policy(state):
        calls["n"] += 1
        if calls["n"] % 5 == 0:
            raise ValueError("가끔 터짐")
        return pd_command(hal, np.full(hal.n_joints, 0.1))

    r = ControlLoopRunner(hal, rate_hz=1000.0, on_error="hold")
    r.set_policy(flaky_policy)
    stats = r.run(n_steps=50)

    assert stats.n_steps == 50
    assert stats.policy_errors == 10


def test_invalid_constructor_args(hal):
    with pytest.raises(ValueError):
        ControlLoopRunner(hal, rate_hz=0.0)
    with pytest.raises(ValueError):
        ControlLoopRunner(hal, rate_hz=100.0, on_error="explode")


# -------------------------------------------------------------- 로거 연동
def test_runner_feeds_logger(hal):
    logger = RingLogger.for_hal(hal, window_sec=5.0)
    r = ControlLoopRunner(hal, rate_hz=500.0, logger=logger)
    r.set_policy(lambda s: pd_command(hal, np.full(hal.n_joints, 0.2)))
    r.run(n_steps=60)

    assert len(logger) == 60
    arrays = logger.to_arrays()
    assert arrays["q"].shape == (60, hal.n_joints)
    assert np.all(np.isfinite(arrays["q_des"]))
    assert np.all(np.isfinite(arrays["loop_dt"]))


def test_wall_time_is_monotonic_across_multiple_runs(hal):
    """회귀 테스트: run()을 나눠 호출해도 로거의 벽시계가 뒤로 가면 안 된다.

    예전엔 run()마다 0부터 다시 세서, 두 번째 run 구간에서 주기가 음수로 잡히고
    dump_text의 rate/jitter가 통째로 엉망이 됐다.
    """
    logger = RingLogger.for_hal(hal, window_sec=10.0)
    r = ControlLoopRunner(hal, rate_hz=500.0, logger=logger)
    r.set_policy(lambda s: pd_command(hal, np.zeros(hal.n_joints)))

    r.run(n_steps=30)
    r.run(n_steps=30)
    r.run(n_steps=30)

    wall = logger.to_arrays()["wall"]
    assert np.all(np.diff(wall) > 0), "벽시계가 뒤로 갔다"
    assert "not monotonic" not in logger.dump_text()

    achieved_line = [ln for ln in logger.dump_text().splitlines() if "rate" in ln][0]
    assert "Hz achieved" in achieved_line
    hz = float(achieved_line.split("Hz")[0].split(":")[1])
    assert 100.0 < hz < 1000.0, achieved_line


def test_dump_reports_non_monotonic_clock_instead_of_garbage(hal):
    """로그가 섞여 시계가 뒤로 가면, 엉터리 통계 대신 그렇다고 말해야 한다."""
    logger = RingLogger.for_hal(hal, window_sec=10.0)
    state = hal.read_state()
    for wall in (0.0, 0.001, 0.002, 0.0005, 0.0015):  # 일부러 뒤섞음
        hal.send_command(JointCommand.hold(state.q, kp=0.0, kd=0.0))
        logger.log(hal.read_state(), loop_dt=1e-4, wall_time=wall)

    text = logger.dump_text()
    assert "not monotonic" in text
    assert "Hz achieved" not in text


def test_end_to_end_jam_shows_up_in_dump(hal):
    """지름길 ①의 실제 사용 흐름: 러너로 돌다 jam → 로그 덤프에 증거가 남는다."""
    logger = RingLogger.for_hal(hal, window_sec=5.0, threshold_frac=0.7)
    r = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger)
    r.set_policy(lambda s: pd_command(hal, np.array([0.8, 0.0, 0.0]), kp=60.0))

    r.run(n_steps=50)
    hal.inject_jam(0)
    logger.mark_event("jam injected on joint 0", t=hal.t)
    r.run(n_steps=150)

    text = logger.dump_text()
    assert "EXCEEDED" in text
    assert "jam injected on joint 0" in text
    assert "joint 0:" in text
