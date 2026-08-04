"""phorce ROS 2 어댑터 — 주석 스켈레톤 (rclpy 실제 import 금지).

*** 파사드 우선 사용 권고 ***
phorce는 파이썬 파사드(phorce.connect())를 제공한다. 우리 스택은 파사드를
PhorceHAL로 감싸는 것이 1순위다 — ROS 2 직결은 파사드가 못 하는 것이 있을
때만 (예: 다른 ROS 노드와의 통합이 요구될 때).

--------------------------------------------------------------------------
직결이 필요해지면 아래 스켈레톤을 채운다
--------------------------------------------------------------------------

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import qos_profile_sensor_data      # <-- 이거 필수!
#
# class PhorceFeedbackBridge(Node):
#     '''/phorce/feedback (1kHz) → FeedbackCache.push'''
#
#     def __init__(self, cache):
#         super().__init__("phorce_feedback_bridge")
#         self.cache = cache
#         # *** 주의: QoS를 qos_profile_sensor_data로 하지 않으면 ***
#         # *** (기본 RELIABLE vs 퍼블리셔 BEST_EFFORT 불일치로)   ***
#         # *** 에러도 없이 조용히 0개 수신한다.                    ***
#         self.sub = self.create_subscription(
#             PhorceFeedbackMsg,                       # 실제 msg 타입으로
#             "/phorce/feedback",
#             self._on_frame,
#             qos_profile_sensor_data,
#         )
#
#     def _on_frame(self, msg):
#         # 1kHz 콜백 규칙: 변환 + 저장만. 판단/서비스콜 금지.
#         frame = PhorceFeedback(
#             t=..., seq=msg.seq,
#             position_rad=np.asarray(msg.position_rad), ...
#             valid=..., stale=..., fault=...,
#         )
#         self.cache.push(frame)
#
# play 하행은 파사드로 유지하는 것을 권장 (거절 코드 → 예외 매핑이 이미
# 파사드에 있다). ROS 서비스로 직결해야 하면:
#     self.play_cli = self.create_client(PlayMotion, "/phorce/play")
#     응답의 거절 코드를 hal/phorce.py의 MotionBusy/MotionRejected로 매핑할 것
#     (12/13 → needs_operator=True — supervisor의 WAITING_OPERATOR 경로).
#
# 실행:
#     rclpy.init()
#     bridge = PhorceFeedbackBridge(cache)
#     # 수신 전용 스레드에서 spin (판단 루프와 분리):
#     threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True).start()

체크리스트 (직결 전 확인):
1. ros2 topic hz /phorce/feedback   → 1kHz 나오는지
2. ros2 topic info -v               → 퍼블리셔 QoS 확인 (BEST_EFFORT일 것)
3. 수신 콜백 스레드 확인            → 메인 아님. 저장만 할 것
4. axis_valid_mask / status_flags   → 프레임 필드 이름을 msg 정의와 대조
"""
