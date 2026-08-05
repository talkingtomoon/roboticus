"""RealPhorceHAL — 실기 파사드 시그니처 대조 + 변환 규칙.

FakeRobot은 **실기 소스(introspect + _impl.py 정독, 2026-08-05)로 확정한**
파사드 계약을 그대로 흉내낸다:
- play_async는 항상 핸들을 반환하고, **거절은 wait()가 예외로 던진다**
  (수락/거절도 비동기 — 우리 HAL은 수락 프로브로 동기 승격)
- wait(timeout) 초과는 builtin TimeoutError (재생은 계속)
- PlayResult(.ok 속성, .status_name, .detail, .completed_count),
  SUCCEEDED/CANCELED만 반환 경로 — REJECTED/ABORTED는 예외 경로
- MotionAborted도 .code/.detail을 가진다 → 타입 이름으로 판별
- status(timeout).state_name / motions() -> Catalog(순회) / doctor()
- 1kHz msg: axis(AxisFeedback[12], 단수) + 프레임 마스크 4종

여기서 지키는 원칙:
- 거절 분류는 **숫자 코드**로만 (문자열 비교 금지)
- "IDLE"만 유휴의 증거 — STALE/CONTRACT_INACTIVE/UNKNOWN은 "unknown"
"""

import importlib.util
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_core.adapters.phorce_ros2 import (
    AM_RX_FRESH_LIMIT_MS, PhorceFeedbackBridge, msg_to_frame,
)
from robot_core.hal.phorce import (
    MotionAborted, MotionBusy, MotionRejected, N_AXES, PhorceError,
    classify_reject_code,
)
from robot_core.hal.real_phorce import RealPhorceHAL, _translate_exception

HAS_PHORCE = importlib.util.find_spec("phorce") is not None
HAS_RCLPY = importlib.util.find_spec("rclpy") is not None


# ------------------------------------------------------------ 파사드 fake
def facade_exc(name: str, **attrs) -> Exception:
    """파사드 예외 흉내 — **타입 이름**과 .code/.reason/.detail이 계약이다."""
    cls = type(name, (Exception,), {})
    e = cls(attrs.get("detail") or attrs.get("reason") or name)
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


class FakeResult:
    """PlayResult 흉내 — .ok는 ★속성이다 (괄호 없음, 실기 확정)."""

    def __init__(self, ok=True, status_name="SUCCEEDED", detail="",
                 completed_count=1):
        self._ok = ok
        self.status_name = status_name
        self.detail = detail
        self.completed_count = completed_count

    @property
    def ok(self):
        return self._ok


class FakeFacadeHandle:
    """파사드 PlayHandle 흉내: wait(timeout)->PlayResult | 예외,
    타임아웃이면 builtin TimeoutError (실기 소스 확정)."""

    def __init__(self):
        self._evt = threading.Event()
        self._result = None
        self._exc = None
        self.last_feedback = None

    @property
    def done(self):
        return self._evt.is_set()

    def wait(self, timeout=None):
        if not self._evt.wait(timeout):
            raise TimeoutError(f"{timeout}s 안에 모션 결과를 받지 못했습니다")
        if self._exc is not None:
            raise self._exc
        return self._result

    def cancel(self):
        pass

    # 테스트 제어용
    def finish(self, result):
        self._result = result
        self._evt.set()

    def fail(self, exc):
        self._exc = exc
        self._evt.set()


class FakeRobot:
    def __init__(self):
        self.state_name = "IDLE"
        self.status_raises = False
        self.reject_with = None          # wait()에서 던질 예외 (실기 경로)
        self.next_handle = FakeFacadeHandle()
        self.doctor_ok = True
        self.doctor_issues = []
        self.play_calls = []

    def play_async(self, motion_id):
        """실기와 동일: 항상 핸들 반환 — 거절도 핸들의 wait()에서 나온다."""
        self.play_calls.append(motion_id)
        if self.reject_with is not None:
            self.next_handle.fail(self.reject_with)
        return self.next_handle

    def status(self, timeout=2.0):
        if self.status_raises:
            raise TimeoutError("status timeout")
        return SimpleNamespace(state_name=self.state_name)

    def motions(self):
        # 실기: Catalog(순회 가능) — 리스트로 충분히 모사된다
        return [SimpleNamespace(id=1, name="approach"),
                SimpleNamespace(id=2, name="insert"),
                SimpleNamespace(id=7, name="insert_slow")]

    def doctor(self):
        return SimpleNamespace(ok=self.doctor_ok, issues=self.doctor_issues)


def make_hal(probe_s=0.05):
    """probe_s: 테스트는 수락 프로브를 짧게 (기본 1.0s는 현장값)."""
    robot = FakeRobot()
    hal = RealPhorceHAL(robot)
    hal._ACCEPT_PROBE_S = probe_s
    return robot, hal


# ------------------------------------------------------------ 연결/import
@pytest.mark.skipif(HAS_PHORCE, reason="phorce가 설치된 환경 — 현장 전용")
def test_connect_without_facade_fails_loud_with_startup_procedure():
    """파사드 부재 시 조용한 실패 금지 — 시동 절차를 말하고 죽는다."""
    with pytest.raises(PhorceError) as ei:
        RealPhorceHAL()
    msg = str(ei.value)
    assert "phorce" in msg
    assert "phorce_monitor" in msg          # 시동 절차 안내 포함
    assert "MockPhorceHAL" in msg           # 목 리허설 대안 안내


# ------------------------------------------------------------ 거절 변환
def test_rejection_codes_convert_by_number_not_string():
    """분류는 숫자 코드로만 — .reason 문자열이 뭐든 결과가 같다."""
    for reason in ("REJECT_QUEUE_FULL", "완전히-다른-문자열", ""):
        exc = _translate_exception(facade_exc("MotionRejected", code=5,
                                              reason=reason))
        assert isinstance(exc, MotionBusy)

    e12 = _translate_exception(facade_exc(
        "MotionRejected", code=12, reason="REJECT_NOT_READY_FOR_MOTION",
        detail="영점을 잡아주세요."))
    assert isinstance(e12, MotionRejected) and e12.needs_operator
    assert e12.detail == "영점을 잡아주세요."      # 한국어 안내 보존

    for code, cat in ((3, "permanent"), (4, "permanent"),
                      (6, "hardware"), (11, "hardware")):
        e = _translate_exception(facade_exc("MotionRejected", code=code))
        assert e.category == cat
    for code in (0, 1, 2, 7, 8, 9, 10):     # 요청/상태 문제 — 전부 fatal
        e = _translate_exception(facade_exc("MotionRejected", code=code))
        assert e.category == "fatal"


def test_facade_busy_type_maps_to_our_busy_regardless_of_code():
    """파사드 MotionBusy(코드가 뭐든) → 우리 MotionBusy — 타입 이름 판별."""
    exc = _translate_exception(facade_exc("MotionBusy", code=5,
                                          reason="GOAL_REJECTED_BUSY",
                                          detail="이미 다른 모션이 실행 중입니다."))
    assert isinstance(exc, MotionBusy)


def test_facade_aborted_is_not_misread_as_rejection():
    """파사드 MotionAborted도 .code를 가진다(실측) — .code 덕 타이핑이면
    거절로 오분류된다. 타입 이름으로 중단으로 분류돼야 한다."""
    exc = _translate_exception(facade_exc(
        "MotionAborted", code=17, reason="REJECT_UNKNOWN_17",
        detail="시작 직후 중단되었습니다."))
    assert isinstance(exc, MotionAborted)
    assert "시작 직후 중단되었습니다." in exc.reason
    assert "17" in exc.reason


def test_classify_unknown_code_is_fatal_not_silent():
    assert classify_reject_code(99) == "fatal"


# ------------------------------------------------------------ 재생/핸들
def test_rejection_from_wait_is_promoted_to_sync_exception():
    """실기: 거절은 wait()에서 나온다 — 수락 프로브가 이를 play_async의
    동기 예외로 승격한다 (감독 루프의 WAITING_OPERATOR 전이가 전제)."""
    robot, hal = make_hal()
    robot.reject_with = facade_exc("MotionRejected", code=12,
                                   reason="REJECT_NOT_READY_FOR_MOTION",
                                   detail="영점 필요")
    with pytest.raises(MotionRejected) as ei:
        hal.play_async(3)
    assert ei.value.needs_operator and ei.value.detail == "영점 필요"
    assert hal._playing is False            # 거절이면 재생 플래그 안 선다


def test_busy_from_wait_is_promoted_to_sync_busy():
    robot, hal = make_hal()
    robot.reject_with = facade_exc("MotionBusy", code=5,
                                   detail="이미 다른 모션이 실행 중입니다.")
    with pytest.raises(MotionBusy):
        hal.play_async(3)


def test_abort_during_probe_goes_through_handle_not_raise():
    """프로브 창 안의 중단은 예외가 아니라 핸들로 전달 — play_async가 던지는
    건 Busy/Rejected/PhorceError뿐이라는 우리 계약 유지."""
    robot, hal = make_hal()
    robot.reject_with = facade_exc("MotionAborted", code=17,
                                   reason="REJECT_UNKNOWN_17",
                                   detail="버튼 복구 직후 중단")
    h = hal.play_async(3)                   # 예외 없이 핸들 반환
    assert h.done()
    with pytest.raises(MotionAborted) as ei:
        h.result()
    assert "버튼 복구 직후 중단" in ei.value.reason


def test_play_success_completes_our_handle_async():
    """프로브를 지나(수락) 비동기 감시 스레드가 완료를 전달한다."""
    robot, hal = make_hal()
    h = hal.play_async(1)                   # 프로브 0.05s 타임아웃 → 진행 중
    assert hal._playing is True
    robot.next_handle.finish(FakeResult(ok=True))
    assert h.wait(timeout=2.0)
    assert h.result() == 1
    assert hal._playing is False


def test_immediate_completion_within_probe():
    """아주 짧은 모션 — 프로브 안에 SUCCEEDED가 오면 동기 완료."""
    robot, hal = make_hal()
    robot.next_handle.finish(FakeResult(ok=True))
    h = hal.play_async(1)
    assert h.done() and h.result() == 1


def test_canceled_result_becomes_motion_aborted_with_detail():
    """status_name은 SUCCEEDED|CANCELED뿐 — REJECTED 분기는 없다.
    CANCELED의 detail(한국어)과 completed_count가 사유에 보존된다."""
    robot, hal = make_hal()
    h = hal.play_async(2)
    robot.next_handle.finish(FakeResult(
        ok=False, status_name="CANCELED",
        detail="E-stop으로 중단되었습니다.", completed_count=0))
    assert h.wait(timeout=2.0)
    with pytest.raises(MotionAborted) as ei:
        h.result()
    assert "CANCELED" in ei.value.reason
    assert "E-stop으로 중단되었습니다." in ei.value.reason
    assert "completed_count=0" in ei.value.reason
    assert hal._playing is False


def test_late_abort_from_wait_becomes_motion_aborted():
    robot, hal = make_hal()
    h = hal.play_async(1)                   # 수락 후 진행 중
    robot.next_handle.fail(facade_exc(
        "MotionAborted", code=0, reason="ACTION_TERMINAL_MISMATCH",
        detail="재생 중 통신이 끊겼습니다."))
    assert h.wait(timeout=2.0)
    with pytest.raises(MotionAborted) as ei:
        h.result()
    assert "재생 중 통신이 끊겼습니다." in ei.value.reason


# ------------------------------------------------------------ busy_state
def test_busy_state_only_idle_is_evidence_of_idle():
    robot, hal = make_hal()
    robot.state_name = "IDLE"
    assert hal.busy_state() == "idle"
    for name in ("STALE", "CONTRACT_INACTIVE", "UNKNOWN", ""):
        robot.state_name = name
        assert hal.busy_state() == "unknown"   # 모름 — idle 추측 금지
    # 실기 PrimaryState 유래 이름들은 전부 busy 취급
    for name in ("EXECUTING", "DISPATCHED", "ACCEPTED", "RECOVERY_REQUIRED"):
        robot.state_name = name
        assert hal.busy_state() == "busy"
    robot.status_raises = True
    assert hal.busy_state() == "unknown"       # 조회 실패도 모름


def test_is_busy_does_not_count_unknown_as_busy():
    robot, hal = make_hal()
    robot.state_name = "STALE"
    assert hal.is_busy() is False              # unknown ≠ busy (판단은 보류가 맡음)


# ------------------------------------------------------------ 카탈로그/진단
def test_catalog_maps_motions_ids():
    robot, hal = make_hal()
    assert hal.catalog() == {1: 0.0, 2: 0.0, 7: 0.0}
    assert hal.motion_names() == {1: "approach", 2: "insert", 7: "insert_slow"}


def test_doctor_maps_ok_and_issues():
    robot, hal = make_hal()
    assert hal.doctor() == {"ok": True, "issues": []}
    robot.doctor_ok = False
    robot.doctor_issues = ["EtherCAT down", "axis 3 fault"]
    d = hal.doctor()
    assert d["ok"] is False and len(d["issues"]) == 2


# ------------------------------------------------------------ 피드백 경로
def _fake_msg(n_axes=6, wkc=3, am_rx_age_ms=0, seq=42,
              recv_monotonic_ns=3_000_000_000, valid_mask=None,
              stale_mask=0, oper_mask=None, fault_mask=0):
    """실기 msg 스키마 (introspect 확정): axis[12] 단수 + 마스크 4종."""
    axis = [SimpleNamespace(
        loop_cnt=i, position_rad=0.1 * i, velocity_rad_s=0.0,
        current_a=1.0 + i, dob_a=0.2, bus_v=48.0, temp_c=40.0 + i,
        pos_ref_echo_rad=0.1 * i, kp_echo=60.0, kd_echo=2.5,
        abs_valid=1, axis_seq=0, age_ms=1,
        oper=True, stale=False, valid=True, fault=False)
        for i in range(n_axes)]
    if valid_mask is None:
        valid_mask = (1 << n_axes) - 1
    if oper_mask is None:
        oper_mask = (1 << n_axes) - 1
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=0, nanosec=0),
        recv_monotonic_ns=recv_monotonic_ns, wkc=wkc, tx_cycle_seq=seq,
        am_rx_seq_echo=0, relay_seq_echo=0,
        axis_valid_mask=valid_mask, axis_stale_mask=stale_mask,
        axis_oper_mask=oper_mask, axis_fault_mask=fault_mask,
        am_rx_age_ms=am_rx_age_ms, status_flags=0, axis=axis)


def test_msg_to_frame_uses_recv_monotonic_not_stamp():
    frame = msg_to_frame(_fake_msg(recv_monotonic_ns=5_500_000_000))
    assert frame.t == pytest.approx(5.5)
    assert frame.seq == 42


def test_msg_to_frame_pads_missing_axes_as_invalid():
    """12칸 중 현 기체 6칸 — 안 온/미장착 축은 무조건 무효."""
    frame = msg_to_frame(_fake_msg(n_axes=6))
    assert frame.position_rad.shape == (N_AXES,)
    assert frame.valid[:6].all()
    assert not frame.valid[6:].any()
    assert frame.usable[:6].all()


def test_msg_to_frame_wkc_mismatch_marks_all_stale():
    frame = msg_to_frame(_fake_msg(wkc=2))     # 정상은 3
    assert frame.stale.all()
    assert not frame.usable.any()              # 판정에서 전 축 제외


def test_msg_to_frame_old_am_rx_marks_all_stale():
    """프레임은 오는데 내용물이 낡은 경우 — 수신 벽시계 워치독이 못 잡는
    유일한 구멍을 am_rx_age_ms로 잡는다."""
    frame = msg_to_frame(_fake_msg(am_rx_age_ms=AM_RX_FRESH_LIMIT_MS + 1))
    assert frame.stale.all()


def test_msg_to_frame_masks_and_flags_combine_conservatively():
    """축 플래그와 프레임 마스크가 어긋나면 보수 쪽: valid는 AND, stale은 OR."""
    frame = msg_to_frame(_fake_msg(n_axes=6, valid_mask=0b000101))
    assert frame.valid[0] and frame.valid[2]
    assert not frame.valid[1]                  # 축 플래그 True여도 마스크가 끈다

    frame2 = msg_to_frame(_fake_msg(n_axes=6, stale_mask=0b000010))
    assert frame2.stale[1]                     # 축 플래그 False여도 마스크가 켠다
    assert not frame2.stale[0]
    assert frame2.usable[0] and not frame2.usable[1]


def test_hal_on_frame_stamps_playing_and_fans_out():
    robot, hal = make_hal()
    got = []
    hal.watch(got.append)
    frame = msg_to_frame(_fake_msg())
    hal._on_frame(frame)
    assert got and got[0].playing is False
    hal.play_async(1)                          # _playing True (수락 후)
    frame2 = msg_to_frame(_fake_msg())
    hal._on_frame(frame2)
    assert frame2.playing is True              # STALL 감지가 이 비트를 전제
    assert hal.latest_feedback() is frame2


def test_on_frame_swallows_callback_errors_but_counts():
    robot, hal = make_hal()

    def bad(frame):
        raise RuntimeError("boom")

    hal.watch(bad)
    hal._on_frame(msg_to_frame(_fake_msg()))
    assert hal.frame_cb_errors == 1            # 수신 스레드는 죽지 않는다


@pytest.mark.skipif(HAS_RCLPY, reason="rclpy가 있는 환경 — 현장 전용")
def test_bridge_without_rclpy_fails_loud():
    bridge = PhorceFeedbackBridge()
    with pytest.raises(RuntimeError) as ei:
        bridge.start(lambda f: None)
    assert "rclpy" in str(ei.value)            # 뭘 해야 하는지 말한다


# ------------------------------------------------------------ 블로킹 재생
def test_play_blocking_wraps_async():
    robot, hal = make_hal()

    def finish_soon():
        time.sleep(0.02)
        robot.next_handle.finish(FakeResult(ok=True))

    threading.Thread(target=finish_soon, daemon=True).start()
    assert hal.play(1) == 1
