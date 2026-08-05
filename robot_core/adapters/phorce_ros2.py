"""phorce ROS 2 피드백 브리지 — /phorce/feedback (1kHz) → PhorceFeedback.

문서 확정 사실 (2026-08):
- 토픽 "/phorce/feedback", msg 타입 agx_msgs/PhorceFeedback
- QoS는 **qos_profile_sensor_data 필수** — 기본(RELIABLE)로 구독하면
  퍼블리셔(BEST_EFFORT)와 불일치로 **에러 없이 0건 수신**한다
- 시간축은 recv_monotonic_ns를 쓴다 (stamp 아님 — 로봇 클록은 스트림이
  죽으면 신뢰할 수 없고, 워치독/판정 창은 수신 단조시계 기준이다)
- 프레임 필드: stamp, recv_monotonic_ns, wkc(정상 3), tx_cycle_seq,
  axis_valid_mask/stale/oper/fault(비트마스크), am_rx_age_ms, status_flags
- 축 필드(12칸, 현 기체는 6칸 사용): position_rad, velocity_rad_s,
  current_a, dob_a, bus_v, temp_c, pos_ref_echo_rad, kp_echo, kd_echo,
  abs_valid, age_ms, valid/stale/oper/fault
- 미사용 축은 valid=False로 들어온다 — 감지기는 usable 마스크로 걸러낸다

rclpy/agx_msgs는 **지연 import** (start()에서만): 개발 환경(rclpy 없음)에서도
모듈 로드와 msg_to_frame 단위 테스트가 가능해야 한다.

수신 콜백 규칙: 변환 + on_frame 호출만. 판단/서비스콜 금지. 예외는 삼키되
cb_errors로 계측한다 (수신 스레드는 죽으면 안 된다).
"""

from __future__ import annotations

import threading

import numpy as np

from robot_core.hal.phorce import N_AXES, PhorceFeedback

TOPIC = "/phorce/feedback"
EXPECTED_WKC = 3            # EtherCAT working counter 정상값 (문서 확정)
AM_RX_FRESH_LIMIT_MS = 1500  # 매뉴얼 kStateFreshLimitMs와 동일 출처


def _axis_array(axes, name, default=0.0):
    out = np.full(N_AXES, float(default))
    for i, ax in enumerate(axes[:N_AXES]):
        out[i] = float(getattr(ax, name, default))
    return out


def _axis_bool(axes, name, default=False, mask: int | None = None):
    """축 불리언: 축 필드 값과 프레임 비트마스크를 OR/AND로 합친다."""
    out = np.full(N_AXES, bool(default))
    for i, ax in enumerate(axes[:N_AXES]):
        out[i] = bool(getattr(ax, name, default))
    if mask is not None:
        bits = np.array([(int(mask) >> i) & 1 for i in range(N_AXES)],
                        dtype=bool)
        return out, bits
    return out


def msg_to_frame(msg) -> PhorceFeedback:
    """agx_msgs/PhorceFeedback → 우리 프레임. 순수 함수 (rclpy 불필요).

    신뢰성 강등 규칙 (수치를 고치지 않고 마스크로만 표현한다):
    - wkc != 3        → 전 축 stale (EtherCAT 사이클이 깨진 프레임)
    - am_rx_age_ms 초과 → 전 축 stale (프레임은 오는데 내용물이 낡은 경우 —
      수신 벽시계 워치독이 못 잡는 유일한 구멍이라 여기서 잡는다)
    - 미장착/미사용 축  → valid=False (msg가 이미 그렇게 준다)
    """
    axes = list(getattr(msg, "axes", []) or [])
    n_present = len(axes)

    valid_axis, valid_bits = _axis_bool(
        axes, "valid", mask=int(getattr(msg, "axis_valid_mask", 0)))
    valid = valid_axis & valid_bits
    if n_present < N_AXES:                       # 안 온 축은 무조건 무효
        valid[n_present:] = False

    stale = _axis_bool(axes, "stale")
    wkc = int(getattr(msg, "wkc", EXPECTED_WKC))
    am_age = float(getattr(msg, "am_rx_age_ms", 0.0))
    if wkc != EXPECTED_WKC or am_age > AM_RX_FRESH_LIMIT_MS:
        stale = np.ones(N_AXES, dtype=bool)

    return PhorceFeedback(
        t=float(getattr(msg, "recv_monotonic_ns", 0)) * 1e-9,  # ★stamp 아님
        seq=int(getattr(msg, "tx_cycle_seq", 0)),
        position_rad=_axis_array(axes, "position_rad"),
        velocity_rad_s=_axis_array(axes, "velocity_rad_s"),
        current_a=_axis_array(axes, "current_a"),
        dob_a=_axis_array(axes, "dob_a"),
        bus_v=_axis_array(axes, "bus_v"),
        temp_c=_axis_array(axes, "temp_c"),
        kp_echo=_axis_array(axes, "kp_echo"),
        kd_echo=_axis_array(axes, "kd_echo"),
        valid=valid,
        oper=_axis_bool(axes, "oper"),
        stale=stale,
        fault=_axis_bool(axes, "fault"),
        axis_valid_mask=int(getattr(msg, "axis_valid_mask", 0)),
        status_flags=int(getattr(msg, "status_flags", 0)),
        playing=False,                 # RealPhorceHAL._on_frame이 스탬프한다
    )


class PhorceFeedbackBridge:
    """rclpy 구독 → on_frame(PhorceFeedback) 팬아웃. 전용 spin 스레드.

    사용:
        bridge = PhorceFeedbackBridge()
        hal.attach_feedback_bridge(bridge)   # hal이 start를 대신 불러준다
        ...
        bridge.stop()
    """

    def __init__(self, topic: str = TOPIC,
                 node_name: str = "roboticus_feedback_bridge") -> None:
        self.topic = topic
        self.node_name = node_name
        self.rx_count = 0
        self.cb_errors = 0
        self._on_frame = None
        self._node = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, on_frame) -> None:
        """구독 시작. rclpy/agx_msgs가 없으면 뭘 해야 하는지 말하고 죽는다."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("bridge already started")
        self._on_frame = on_frame
        try:
            import rclpy
            # ★필수 — 기본 QoS면 에러 없이 0건 수신한다 (문서 명시)
            from rclpy.qos import qos_profile_sensor_data
        except ImportError as e:
            raise RuntimeError(
                "rclpy를 찾을 수 없다 — 젯슨에서 ROS 2 환경을 source했는지 "
                "확인 (개발 PC에서는 MockPhorceHAL로 리허설).") from e
        try:
            from agx_msgs.msg import PhorceFeedback as PhorceFeedbackMsg
        except ImportError as e:
            raise RuntimeError(
                "agx_msgs를 찾을 수 없다 — 워크스페이스 overlay를 source했는지 "
                "확인 (터미널1/2가 뜨는 환경과 같은 셸이어야 한다).") from e

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node(self.node_name)
        self._node.create_subscription(
            PhorceFeedbackMsg, self.topic, self._cb, qos_profile_sensor_data)
        self._stop.clear()

        def spin():
            while not self._stop.is_set() and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.1)

        self._thread = threading.Thread(target=spin, daemon=True,
                                        name="phorce-feedback-spin")
        self._thread.start()

    def _cb(self, msg) -> None:
        """수신 스레드 — 변환 + 팬아웃만. 예외 삼키고 계측 (죽으면 안 됨)."""
        self.rx_count += 1
        try:
            self._on_frame(msg_to_frame(msg))
        except Exception:
            self.cb_errors += 1

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
