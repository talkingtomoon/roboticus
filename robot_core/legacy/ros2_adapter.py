# ROS 2 어댑터 스켈레톤 — 현장에서 채운다. 지금은 rclpy를 import하지 않는다.
#
# 목표: robot_core.graph의 Node/NodeGraphManager를 코드 수정 없이 ROS 2 위에 올린다.
# 우리 노드 인터페이스(update(inputs)->dict, enabled, params)는 rclpy 노드에
# 1:1로 대응되도록 이미 설계돼 있다:
#
#   우리 개념              ROS 2 대응
#   -------------------   ------------------------------------------
#   Node.update()          타이머 콜백 (create_timer)
#   그래프 간선(edge)      토픽 publish/subscribe
#   Node.params            ROS 2 파라미터 서버 (declare_parameter)
#   enable/disable         lifecycle node activate/deactivate
#   SafetyGuard.apply()    on_set_parameters_callback (검증 지점 동일!)
#
# ---------------------------------------------------------------------------
# 채우는 순서
# ---------------------------------------------------------------------------
# 1) 래퍼 클래스 골격
#
#    import rclpy
#    from rclpy.node import Node as RclpyNode
#    from std_msgs.msg import Float64MultiArray   # 또는 커스텀 msg
#
#    class Ros2NodeWrapper(RclpyNode):
#        """robot_core Node 하나를 rclpy 노드로 감싼다."""
#        def __init__(self, core_node, rate_hz=1000.0):
#            super().__init__(core_node.name)
#            self._core = core_node
#            self._inputs = {}          # 토픽에서 모은 최신 입력 캐시
#            # (a) 업스트림 구독: 그래프 간선마다 하나
#            #     self.create_subscription(Float64MultiArray,
#            #         f"/{src_name}/out", self._on_upstream, 10)
#            # (b) 출력 퍼블리셔
#            #     self._pub = self.create_publisher(Float64MultiArray,
#            #         f"/{core_node.name}/out", 10)
#            # (c) 주기 실행
#            #     self.create_timer(1.0 / rate_hz, self._tick)
#
#        def _on_upstream(self, msg):
#            pass  # msg → dict 변환해서 self._inputs.update(...)
#
#        def _tick(self):
#            if not self._core.enabled:
#                return
#            out = self._core.update(self._inputs)
#            # out(dict) → msg 변환 후 self._pub.publish(...)
#
# 2) 파라미터 서버 연결
#    - __init__에서 core_node.params의 각 키를 declare_parameter(k, v)
#    - self.add_on_set_parameters_callback(self._on_param_change) 등록
#    - _on_param_change에서는 **반드시 SafetyGuard를 통과시킬 것**:
#          audit = guard.apply([{"node": self._core.name,
#                                "param": p.name, "value": p.value}],
#                              source="ros2-param")
#          return SetParametersResult(successful=<audit 결과에 따라>)
#      → ros2 param set으로 들어오는 값도 LLM 값과 같은 검증 경로를 탄다.
#
# 3) 그래프 전체 띄우기
#
#    def launch_graph(manager, rate_hz=1000.0):
#        rclpy.init()
#        executor = rclpy.executors.MultiThreadedExecutor()
#        wrappers = [Ros2NodeWrapper(manager.node(n), rate_hz)
#                    for n in manager.node_names()]
#        for w in wrappers: executor.add_node(w)
#        executor.spin()
#
#    주의: 토픽 경유는 노드마다 큐/지연이 생긴다. 1kHz 안쪽 경로(임피던스 지령)는
#    토픽으로 쪼개지 말고 하나의 rclpy 노드 안에서 manager.step()을 통째로 돌리는
#    편이 안전하다. 토픽으로 빼는 건 저주파 경로(모니터링, 회복 루프)만.
#
# 4) enable/disable
#    - 간단 버전: ~/enable 서비스(SetBool) 하나 만들어 core_node.enabled 토글
#    - 정식 버전: LifecycleNode 상속, on_activate/on_deactivate에서 토글
#
# 5) 메시지 타입
#    - 프로토타입은 Float64MultiArray + layout으로 버틴다
#    - 시간 나면 JointState(sensor_msgs)와 커스텀 JointCommand.msg로 교체
#      (q_des, qd_des, tau_ff, kp, kd — robot_core.hal.JointCommand와 필드 동일)

# 이 파일은 의도적으로 실행 코드가 없다. rclpy 없는 환경에서도 임포트돼야 한다.
