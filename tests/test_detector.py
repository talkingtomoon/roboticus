"""FailureDetector — 합성 배열 기반 단위 판정 + MockRobotHAL 통합 판정."""

import numpy as np
import pytest

from robot_core import MockRobotHAL, RingLogger
from robot_core.recovery import DetectorConfig, FailureDetector, FailureType

from .conftest import pd_command, run_steps

HZ = 1000.0


def make_arrays(n_joints=2, duration=1.0, tau=None, q=None, qd=None, q_des=None):
    """RingLogger.to_arrays()와 같은 모양의 합성 데이터."""
    n = int(duration * HZ)
    t = np.arange(n) / HZ
    zeros = np.zeros((n, n_joints))
    return {
        "t": t,
        "wall": t.copy(),
        "q": zeros.copy() if q is None else q,
        "qd": zeros.copy() if qd is None else qd,
        "tau": zeros.copy() if tau is None else tau,
        "q_des": zeros.copy() if q_des is None else q_des,
        "loop_dt": np.full(n, 1e-4),
        "overrun": np.zeros(n, dtype=bool),
    }


@pytest.fixture
def det():
    return FailureDetector(2, DetectorConfig(
        torque_threshold=10.0, torque_min_duration_s=0.02,
        stall_err_threshold=0.1, stall_qd_eps=0.05, stall_min_duration_s=0.25,
        osc_qd_amp_eps=0.3, osc_min_flips_hz=8.0, osc_window_s=0.3,
        refractory_s=1.0,
    ))


# --------------------------------------------------------------- TORQUE_SPIKE
def test_sustained_torque_spike_fires(det):
    tau = np.zeros((1000, 2))
    tau[-50:, 1] = 14.0  # 50ms 지속 > 20ms 최소
    events = det.check(make_arrays(tau=tau))
    assert [e.type for e in events] == [FailureType.TORQUE_SPIKE]
    assert events[0].joint_idx == 1
    assert events[0].snapshot["threshold"] == 10.0


def test_single_sample_glitch_does_not_fire(det):
    tau = np.zeros((1000, 2))
    tau[-1, 0] = 50.0  # 마지막 1샘플만 스파이크
    assert det.check(make_arrays(tau=tau)) == []


def test_torque_hysteresis_no_refire_until_release(det):
    tau = np.zeros((1000, 2))
    tau[-100:, 0] = 14.0
    arrays = make_arrays(tau=tau)
    assert len(det.check(arrays)) == 1
    # 계속 초과 상태 (refractory 지나도 release 안 됨) → 재발동 금지
    arrays2 = make_arrays(tau=tau)
    arrays2["t"] = arrays["t"] + 5.0
    arrays2["wall"] = arrays2["t"]
    assert det.check(arrays2) == []
    # 임계치*release_frac 밑으로 떨어진 창 → 재무장
    calm = make_arrays(tau=np.zeros((1000, 2)))
    calm["t"] = arrays["t"] + 10.0
    assert det.check(calm) == []
    # 다시 초과 → 발동
    arrays3 = make_arrays(tau=tau)
    arrays3["t"] = arrays["t"] + 15.0
    assert len(det.check(arrays3)) == 1


def test_negative_torque_also_detected(det):
    tau = np.zeros((1000, 2))
    tau[-50:, 0] = -14.0
    events = det.check(make_arrays(tau=tau))
    assert len(events) == 1


# --------------------------------------------------------------------- STALL
def _stall_arrays(err=0.4, qd_val=0.0):
    n = 1000
    q_des = np.zeros((n, 2))
    q_des[:, 0] = err  # 지령은 err 위치, 실제 q는 0에 고정
    qd = np.zeros((n, 2))
    qd[:, 0] = qd_val
    return make_arrays(q_des=q_des, qd=qd)


def test_stall_fires_when_error_large_and_joint_still(det):
    events = det.check(_stall_arrays())
    assert [e.type for e in events] == [FailureType.STALL]
    assert events[0].joint_idx == 0
    assert events[0].snapshot["err"] == pytest.approx(0.4)


def test_no_stall_when_joint_is_moving(det):
    assert det.check(_stall_arrays(qd_val=0.5)) == []


def test_no_stall_when_error_is_shrinking(det):
    """천천히라도 가고 있으면 스톨이 아니다."""
    n = 1000
    q = np.zeros((n, 2))
    q[:, 0] = np.linspace(0.0, 0.3, n)  # 목표 0.4로 접근 중
    q_des = np.zeros((n, 2))
    q_des[:, 0] = 0.4
    arrays = make_arrays(q=q, q_des=q_des)  # qd=0이지만 오차 감소 중
    assert det.check(arrays) == []


def test_no_stall_without_command_data(det):
    arrays = _stall_arrays()
    arrays["q_des"] = np.full_like(arrays["q_des"], np.nan)
    assert det.check(arrays) == []


# --------------------------------------------------------------- OSCILLATION
def test_oscillation_fires_on_high_freq_sign_flips(det):
    n = 1000
    t = np.arange(n) / HZ
    qd = np.zeros((n, 2))
    qd[:, 1] = 1.5 * np.sin(2 * np.pi * 15.0 * t)  # 15Hz → 부호반전 30/s
    events = det.check(make_arrays(qd=qd))
    assert [e.type for e in events] == [FailureType.OSCILLATION]
    assert events[0].joint_idx == 1
    assert events[0].snapshot["flips_hz"] > 8.0


def test_tiny_amplitude_jitter_ignored(det):
    n = 1000
    t = np.arange(n) / HZ
    qd = np.zeros((n, 2))
    qd[:, 0] = 0.05 * np.sin(2 * np.pi * 30.0 * t)  # 진폭 < amp_eps
    assert det.check(make_arrays(qd=qd)) == []


def test_slow_oscillation_ignored(det):
    n = 2000
    t = np.arange(n) / HZ
    qd = np.zeros((n, 2))
    qd[:, 0] = 1.0 * np.sin(2 * np.pi * 1.0 * t)  # 1Hz — 정상 운동 수준
    assert det.check(make_arrays(duration=2.0, qd=qd)) == []


# ------------------------------------------------------------------- 공통
def test_too_few_samples_is_safe(det):
    arrays = make_arrays(duration=0.002)
    assert det.check(arrays) == []
    assert det.check({"t": np.zeros(0)}) == []


def test_reset_rearms_everything(det):
    tau = np.zeros((1000, 2))
    tau[-50:, 0] = 14.0
    arrays = make_arrays(tau=tau)
    assert len(det.check(arrays)) == 1
    det.reset()
    assert len(det.check(arrays)) == 1


# ------------------------------------------- MockRobotHAL 통합 (이거 꼭)
def test_stall_catches_jam_at_rest_on_real_mock():
    """정지 중 고착: 목표에 도달해 정지한 관절이 걸린 뒤 새 지령이 왔을 때.

    kp를 낮게 잡아 토크가 임계치(80%)에 한참 못 미치게 한다 —
    토크 감지만으로는 절대 못 잡고, STALL 감지만이 잡을 수 있는 시나리오.
    """
    hal = MockRobotHAL(n_joints=3, dt=1e-3, torque_limit=20.0)
    logger = RingLogger.for_hal(hal, window_sec=3.0)
    det = FailureDetector(3, DetectorConfig(
        torque_threshold=hal.torque_limits * 0.8,
        stall_err_threshold=0.1, stall_min_duration_s=0.25,
    ))

    kp = 5.0  # 오차 0.5여도 토크 2.5 Nm << 16 Nm
    run_steps(hal, pd_command(hal, np.zeros(3), kp=kp), 500, logger=logger)
    assert np.allclose(hal.read_state().qd, 0.0, atol=1e-3)  # 정지 상태 확인

    hal.inject_jam(0)
    run_steps(hal, pd_command(hal, np.array([0.5, 0.0, 0.0]), kp=kp), 600, logger=logger)

    events = det.check(logger.to_arrays(window_sec=1.0))
    types = {(e.type, e.joint_idx) for e in events}
    assert (FailureType.STALL, 0) in types
    assert all(e.type != FailureType.TORQUE_SPIKE for e in events), \
        "토크는 임계치 밑이어야 하는 시나리오다"
    # 실제로 토크가 낮았는지 물리적으로도 확인
    assert abs(hal.read_state().tau_measured[0]) < 0.8 * hal.torque_limits[0] / 2


def test_torque_spike_fires_on_jam_while_moving():
    """움직이다 걸림: 높은 kp로 이동 중 jam → 토크 포화 → TORQUE_SPIKE."""
    hal = MockRobotHAL(n_joints=3, dt=1e-3, torque_limit=20.0)
    logger = RingLogger.for_hal(hal, window_sec=3.0)
    det = FailureDetector(3, DetectorConfig(torque_threshold=hal.torque_limits * 0.8))

    hal.inject_jam(0)
    run_steps(hal, pd_command(hal, np.array([1.0, 0.0, 0.0]), kp=60.0), 300, logger=logger)

    events = det.check(logger.to_arrays(window_sec=1.0))
    assert (FailureType.TORQUE_SPIKE, 0) in {(e.type, e.joint_idx) for e in events}


def test_oscillation_fires_on_underdamped_gains_on_real_mock():
    """kd=0 + 높은 kp → 실제 진동 발생 → OSCILLATION."""
    hal = MockRobotHAL(n_joints=2, dt=1e-3, torque_limit=50.0,
                       coulomb_friction=0.0, viscous_friction=0.0)
    logger = RingLogger.for_hal(hal, window_sec=3.0)
    det = FailureDetector(2, DetectorConfig(torque_threshold=45.0))

    run_steps(hal, pd_command(hal, np.array([0.5, 0.0]), kp=100.0, kd=0.0), 800,
              logger=logger)

    events = det.check(logger.to_arrays(window_sec=0.5))
    assert (FailureType.OSCILLATION, 0) in {(e.type, e.joint_idx) for e in events}
