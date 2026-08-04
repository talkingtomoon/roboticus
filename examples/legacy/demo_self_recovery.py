"""자가 회복 루프 통합 데모.

MockRobotHAL에 외란/jam을 주입해 실패를 유발하고,
감지 → LLM 질의(또는 규칙 폴백) → SafetyGuard 검증 → 파라미터 갱신 → 회복
전 과정을 타임라인으로 출력한다.

실행:
    python examples/demo_self_recovery.py            # ANTHROPIC_API_KEY 있으면 실제 LLM,
                                                     # 없으면 시뮬레이트된 LLM (라벨 표시)
    python examples/demo_self_recovery.py --no-llm   # 규칙 기반 폴백 경로만
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core import ControlLoopRunner, MockRobotHAL, RingLogger
from robot_core.graph import ImpedanceNode, NodeGraphManager, TargetNode
from robot_core.recovery import (
    AnthropicChatClient, DetectorConfig, FailureDetector, LLMConfig,
    LLMRecoveryAgent, ParamSpec, RecoverySupervisor, RuleBasedRecovery, SafetyGuard,
)

N_JOINTS = 3


class SimulatedLLMClient:
    """오프라인 데모용 가짜 LLM. 실제 API 형식의 JSON을 0.3초 지연 후 돌려준다.

    (비동기 경로 + JSON 파싱 + SafetyGuard 검증을 실제와 동일하게 태우기 위한 것)
    """

    def __call__(self, system: str, user: str) -> str:
        time.sleep(0.3)
        if "STALL" in user:
            return json.dumps({
                "diagnosis": "Joint stalled against resistance; stiffness too low to overcome it.",
                "actions": [{"node": "impedance", "param": "kp", "value": 60.0}],
                "confidence": 0.85,
            })
        if "TORQUE_SPIKE" in user:
            return json.dumps({
                "diagnosis": "Sustained torque saturation suggests mechanical jam; retreat and soften.",
                "actions": [
                    {"node": "target", "param": "retreat", "value": 0.08},
                    {"node": "impedance", "param": "kp", "value": 20.0},
                ],
                "confidence": 0.9,
            })
        return json.dumps({
            "diagnosis": "Oscillation from underdamped gains; add damping.",
            "actions": [{"node": "impedance", "param": "kd", "value": 3.5}],
            "confidence": 0.8,
        })


class Timeline:
    """[ +2.134s] 형식의 스레드 안전 콘솔 타임라인."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()

    def __call__(self, msg: str, tag: str = "") -> None:
        with self._lock:
            print(f"[{time.perf_counter() - self.t0:+8.3f}s] {tag}{msg}", flush=True)


def build_system(use_llm: bool, log: Timeline):
    hal = MockRobotHAL(n_joints=N_JOINTS, dt=1e-3, torque_limit=20.0,
                       coulomb_friction=0.15, viscous_friction=0.05)
    logger = RingLogger.for_hal(hal, window_sec=5.0, threshold_frac=0.75)

    mgr = NodeGraphManager()
    mgr.add_node(TargetNode(params={"depth": 0.4, "retreat": 0.0}))
    mgr.add_node(ImpedanceNode(params={"kp": 25.0, "kd": 1.5}))
    mgr.connect("target", "impedance")

    guard = SafetyGuard(mgr, {
        "impedance.kp": ParamSpec(1.0, 120.0, max_rel_step=2.5),
        "impedance.kd": ParamSpec(0.1, 8.0, max_rel_step=2.5),
        "target.depth": ParamSpec(0.0, 1.0, max_rel_step=None, max_abs_step=0.2),
        "target.retreat": ParamSpec(0.0, 0.3, max_rel_step=None, max_abs_step=0.1),
    })
    rules = RuleBasedRecovery(mgr)

    config = LLMConfig(timeout_s=3.0, cooldown_s=2.0)
    if not use_llm:
        client, label = None, "LLM 비활성 (--no-llm) → 규칙 기반 폴백만 사용"
    elif AnthropicChatClient.available(config):
        client, label = AnthropicChatClient(config), f"실제 Anthropic API 사용 ({config.model})"
    else:
        client, label = SimulatedLLMClient(), "API 키 없음 → 시뮬레이트된 LLM (오프라인 데모 모드)"
    log(f"LLM 경로: {label}")

    agent = LLMRecoveryAgent(guard, rules, logger=logger, config=config, client=client,
                             on_log=lambda m: log(m, tag="  [agent] "))
    detector = FailureDetector(N_JOINTS, DetectorConfig(
        torque_threshold=hal.torque_limits * 0.75,
        stall_err_threshold=0.08, stall_min_duration_s=0.25, refractory_s=1.5))
    supervisor = RecoverySupervisor(detector, agent, logger,
                                    on_log=lambda m: log(m, tag="  [monitor] "))

    runner = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger)
    runner.set_policy(lambda st: mgr.step({"state": st})["impedance"]["command"])
    return hal, logger, mgr, guard, agent, supervisor, runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true", help="규칙 기반 폴백 경로만 사용")
    args = parser.parse_args()

    log = Timeline()
    hal, logger, mgr, guard, agent, supervisor, runner = build_system(not args.no_llm, log)

    supervisor.start()  # 모니터 20Hz + LLM 워커 시작 (제어 루프와 별도 스레드)
    try:
        log("=== 1단계: 정상 접근 — 관절 0을 depth 0.4로 ===")
        runner.run(duration_sec=0.5)
        log(f"도달: q[0]={hal.read_state().q[0]:.3f} (목표 0.4)")

        log("=== 2단계: 지속 저항 외란 -8 Nm 주입 → STALL 유발 ===")
        log("    (kp=25로는 토크가 임계치에 못 미쳐 STALL 감지만이 잡을 수 있다)")
        hal.inject_disturbance(0, -8.0, duration=60.0)
        runner.run(duration_sec=1.5)  # 이 사이 모니터가 감지→회복을 수행한다
        err = abs(hal.read_state().q[0] - 0.4)
        kp = mgr.get_params("impedance")["kp"]
        log(f"회복 1차 결과: kp={kp:.1f}, 오차={err:.3f} rad")
        runner.run(duration_sec=1.5)  # 재발 시 2차 회복 (이전 시도가 프롬프트에 포함됨)
        err = abs(hal.read_state().q[0] - 0.4)
        kp = mgr.get_params("impedance")["kp"]
        log(f"회복 2차 결과: kp={kp:.1f}, 오차={err:.3f} rad")
        hal.clear_disturbances()

        log("=== 3단계: 관절 1 물리적 jam + 접근 지령 → TORQUE_SPIKE 유발 ===")
        hal.inject_jam(1)
        # jam된 관절에 지령을 줘야 토크가 쌓인다: 관절 1에 강한 홀드 오차를 만든다
        hal.q[1] = -0.5  # 걸린 위치가 목표(0)에서 멀어진 상황 재현
        hal._q_link[1] = -0.5
        runner.run(duration_sec=1.5)
        log(f"현재 파라미터: {mgr.get_params('impedance')}, target={mgr.get_params('target')}")
        hal.clear_jam()
        runner.run(duration_sec=0.5)

        agent.wait_idle(timeout=10.0)
        log("=== 최종 보고 ===")
        print()
        print(guard.dump_audit_text())
        print()
        print(logger.dump_text(window_sec=2.0))
        print()
        print(runner.last_stats.summary())
        print()
        stats = dict(agent.stats)
        log(f"에이전트 통계: {stats}")
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
