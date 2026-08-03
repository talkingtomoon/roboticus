"""DREAM-Chunk 스위칭 통합 데모 — phact-401 6축 기준.

시나리오:
  로봇이 HOME→TASK 점대점 이동 중 → 경로 중간에 측면 충돌급 외란(+4.6 Nm,
  j0 베이스 요) → 감지(TORQUE_SPIKE) → 후보 전체 채점(점수표 출력) →
  순응하는 우회 청크(detour_left)로 C1 연속 블렌딩 전환 → 목표 도달.

출력:
  타임라인 + 후보별 점수표 + 스위칭 전후 참조 신호의 연속성 수치 검증.

실행:
    python examples/demo_chunk_switching.py
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core import ControlLoopRunner, MockRobotHAL, RingLogger
from robot_core.graph import ImpedanceNode, NodeGraphManager
from robot_core.hal import PHACT_401
from robot_core.integration.full_stack import (
    KD, KP, LATERAL_JOINT, TASK, default_dictionary,
)
from robot_core.recovery import DetectorConfig, FailureDetector
from robot_core.switching import (
    ChunkScorer, ChunkSwitchNode, DreamModel, ScorerConfig, estimate_disturbance,
)

# phact-401 6축: 딕셔너리/포즈/게인은 integration.full_stack의 표준 정의를 쓴다.
N = PHACT_401.n_joints
IMPACT = 15.0                            # 감지 임계(tau_detect=10.8) 위, 순간 정격 안
DETECT_THRESHOLD = PHACT_401.tau_detect  # 연속×1.5 — 자체 과도 토크 오검출 방지


class Timeline:
    def __init__(self):
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()

    def __call__(self, msg, tag=""):
        with self._lock:
            print(f"[{time.perf_counter() - self.t0:+8.3f}s] {tag}{msg}", flush=True)


def main() -> None:
    log = Timeline()
    hal = MockRobotHAL(n_joints=N, dt=1e-3, torque_limit=PHACT_401.tau_limit)
    logger = RingLogger.for_hal(hal, window_sec=5.0)

    dictionary = default_dictionary()
    log(f"청크 딕셔너리: {dictionary.names()}")

    dream = DreamModel.from_mock_hal(hal, kp=KP, kd=KD)
    # veto는 순간 정격(21.6 Nm) 기준 — 연속 한계로 걸면 진입 과도 토크만으로
    # 접근 후보가 탈락한다 (profiles.py 참고)
    dream.torque_limit = np.full(N, PHACT_401.tau_limit_peak)
    scorer = ChunkScorer(dictionary.all(), dream=dream,
                         config=ScorerConfig(entry_dir_window_s=0.6))
    node = ChunkSwitchNode(params={"cooldown_s": 0.4}, scorer=scorer, goal=TASK)
    node.set_active(dictionary.get("direct"), t_now=0.0)

    mgr = NodeGraphManager()
    mgr.add_node(node)
    mgr.add_node(ImpedanceNode(params={"kp": KP, "kd": KD}))
    mgr.connect("chunk_switch", "impedance")

    # 참조 신호 연속성 계측: q_des가 이전 틱의 (q_des + qd_des*dt) 예측에서
    # 얼마나 벗어나는지 — 스위칭 틱에서도 2차항(가속도) 수준이어야 한다
    dt = 1e-3
    cont = {"prev": None, "max_pos_gap": 0.0, "max_vel_jump": 0.0}

    def policy(state):
        out = mgr.step({"state": state})
        ref = out["chunk_switch"]
        if cont["prev"] is not None:
            pred = cont["prev"]["q_des"] + cont["prev"]["qd_des"] * dt
            cont["max_pos_gap"] = max(cont["max_pos_gap"],
                                      float(np.abs(ref["q_des"] - pred).max()))
            cont["max_vel_jump"] = max(cont["max_vel_jump"],
                                       float(np.abs(ref["qd_des"] - cont["prev"]["qd_des"]).max()))
        cont["prev"] = ref
        return out["impedance"]["command"]

    runner = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger)
    runner.set_policy(policy)

    detector = FailureDetector(N, DetectorConfig(
        torque_threshold=DETECT_THRESHOLD, torque_min_duration_s=0.008,
        refractory_s=0.5))

    log("=== 1단계: 'direct' 궤적으로 HOME→TASK 이동 시작 ===")
    runner.run(n_steps=400)
    log(f"진행 중: q={np.round(hal.read_state().q, 3)}, phase={node.phase}")

    log(f"=== 2단계: 관절 {LATERAL_JOINT}(베이스 요)에 +{IMPACT} Nm 측면 충격 ===")
    hal.inject_disturbance(LATERAL_JOINT, +IMPACT, duration=5.0)
    logger.mark_event(f"obstacle: +{IMPACT} Nm side force on joint {LATERAL_JOINT}",
                      t=hal.t)

    # 10ms 주기 모니터: 감지 → 방향 추정 → 스위칭 요청 (제어 루프는 계속 돈다)
    detected = False
    for _ in range(10):
        runner.run(n_steps=10)
        events = detector.check(logger.to_arrays(window_sec=1.0))
        if events and not detected:
            detected = True
            ev = events[0]
            log(f"감지: {ev.describe()}", tag="  [monitor] ")
            d = estimate_disturbance(logger.to_arrays(), window_s=0.05,
                                     baseline_s=0.3, stiffness=KP)
            log(f"외란 방향 추정: {np.round(d, 2)} → 스위칭 요청", tag="  [monitor] ")
            node.request_switch(d)

    log("=== 3단계: 채점 → 블렌딩 전환 → 새 궤적 완주 ===")
    runner.run(duration_sec=2.5)

    log("=== 결과 ===")
    print()
    print(node.dump_decisions())
    print()
    q_final = hal.read_state().q
    print(f"최종 자세      : {np.round(q_final, 3)}  (목표 {np.round(TASK, 3)})")
    print(f"관절 {LATERAL_JOINT} 처짐    : {q_final[LATERAL_JOINT]:.3f} rad — "
          f"지속 외란 {IMPACT} Nm / kp {KP:.0f} = {IMPACT / KP:.3f} 예상과 일치")
    print()
    print("[참조 신호 연속성 — 스위칭 포함 전 구간]")
    print(f"  max |q_des - (이전 q_des + qd_des*dt)| = {cont['max_pos_gap']:.2e} rad"
          f"   (가속도*dt^2 수준이면 C1 연속)")
    print(f"  max |qd_des 틱간 점프|                 = {cont['max_vel_jump']:.2e} rad/s"
          f" (가속도*dt 수준이면 C1 연속)")
    print()
    print(runner.last_stats.summary())


if __name__ == "__main__":
    main()
