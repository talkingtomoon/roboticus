"""스위칭 방식 비교: 채점기 vs 무작위 vs 항상-첫-후보 (DREAM-Chunk 베이스라인).

S2형 시나리오(이동 중 측면 충격 → 스위칭)를 3방식 × N시드로 돌려
목표 도달률 / 도달 시간 / 최대 토크를 비교하고 markdown 표로 저장한다.

    python scripts/baseline_switch_comparison.py --seeds 5

발표 자료 원본: docs/baseline_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core import MockRobotHAL, RingLogger
from robot_core.graph import ImpedanceNode, NodeGraphManager
from robot_core.hal import PHACT_401
from robot_core.integration.full_stack import (
    KD, KP, LATERAL_JOINT, TASK, default_dictionary,
)
from robot_core.recovery import DetectorConfig, FailureDetector, FailureType
from robot_core.switching import (
    ChunkScorer, ChunkSwitchNode, DreamModel, ScorerConfig, estimate_disturbance,
)
from robot_core.switching.baselines import FirstChunkSelector, RandomChunkSelector

# phact-401 6축 기준 — 감지 임계는 표준 tau_detect (연속×1.5 = 10.8 Nm)
N_JOINTS = PHACT_401.n_joints
TOL = 0.07
SIM_STEPS = 3500
DETECT_THRESHOLD = PHACT_401.tau_detect


def make_selector(method: str, chunks, hal, seed: int):
    if method == "scorer":
        dream = DreamModel.from_mock_hal(hal, kp=KP, kd=KD)
        # veto는 순간 정격 기준 — 연속 한계(5.76)로 걸면 진입 과도 토크만으로
        # 접근 후보 전원이 탈락해 retreat만 남는다 (FullStack.build와 동일 배선)
        dream.torque_limit = np.full(N_JOINTS, PHACT_401.tau_limit_peak)
        return ChunkScorer(chunks, dream=dream,
                           config=ScorerConfig(entry_dir_window_s=0.6))
    if method == "random":
        return RandomChunkSelector(chunks, seed=seed)
    if method == "first":
        return FirstChunkSelector(chunks)
    raise ValueError(method)


def run_once(method: str, seed: int) -> dict:
    """시드별로 충격 시점/크기를 살짝 바꿔 공정 비교."""
    rng = np.random.default_rng(seed)
    impact_step = 350 + int(rng.integers(0, 120))
    # 임계(10.8) 위, 순간 정격(21.6) 안 — kd 감쇠를 견디려면 임계의 1.4배+
    impact_mag = 14.5 + float(rng.uniform(0.0, 3.0))

    hal = MockRobotHAL(n_joints=N_JOINTS, dt=1e-3, torque_limit=PHACT_401.tau_clamp)
    logger = RingLogger.for_hal(hal, window_sec=3.0)
    dictionary = default_dictionary()
    # 시드별로 후보 순서를 섞는다 (전 방식 동일 순서 = 공정).
    # 채점기는 순서 불변이지만 '항상 첫 후보'는 순서라는 우연에 지배된다 —
    # 그 우연성이 베이스라인의 본질적 약점이다.
    chunks = dictionary.all()
    rng.shuffle(chunks)
    node = ChunkSwitchNode(params={"cooldown_s": 0.4},
                           scorer=make_selector(method, chunks, hal, seed),
                           goal=TASK)
    node.set_active(dictionary.get("direct"), t_now=0.0)

    mgr = NodeGraphManager()
    mgr.add_node(node)
    mgr.add_node(ImpedanceNode(params={"kp": KP, "kd": KD}))
    mgr.connect("chunk_switch", "impedance")

    detector = FailureDetector(N_JOINTS, DetectorConfig(
        torque_threshold=DETECT_THRESHOLD, torque_min_duration_s=0.01,
        refractory_s=1.0))

    errs, taus = [], []
    switched = False
    for k in range(SIM_STEPS):
        state = hal.read_state()
        cmd = mgr.step({"state": state})["impedance"]["command"]
        hal.send_command(cmd)
        logger.log(state, cmd=cmd, loop_dt=1e-4, wall_time=state.timestamp)
        s = hal.read_state()
        errs.append(float(np.abs(s.q - TASK).max()))
        taus.append(float(np.abs(s.tau_measured).max()))

        if k == impact_step:
            hal.inject_disturbance(LATERAL_JOINT, impact_mag, duration=0.3)
        if not switched and k > impact_step and k % 20 == 0:
            events = detector.check(logger.to_arrays(window_sec=1.0))
            if any(e.type == FailureType.TORQUE_SPIKE for e in events):
                d = estimate_disturbance(logger.to_arrays(window_sec=1.0),
                                         window_s=0.05, baseline_s=0.3, stiffness=KP)
                node.request_switch(d)
                switched = True

    errs = np.array(errs)
    beyond = np.flatnonzero(errs > TOL)
    reached = errs[-1] < TOL
    t_goal = (beyond[-1] + 1) * 1e-3 if reached and beyond.size else (
        0.0 if reached else float("nan"))
    chosen = next((d.chosen for d in node.decisions if d.chosen), "(no switch)")
    return {"reached": reached, "t_goal": t_goal, "max_tau": max(taus),
            "chosen": chosen}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="docs/baseline_comparison.md")
    args = p.parse_args()

    methods = ["scorer", "random", "first"]
    rows = {}
    detail_lines = []
    for method in methods:
        runs = [run_once(method, seed) for seed in range(args.seeds)]
        reach = [r["reached"] for r in runs]
        t_ok = [r["t_goal"] for r in runs if r["reached"]]
        rows[method] = {
            "rate": 100.0 * sum(reach) / len(runs),
            "t_mean": float(np.mean(t_ok)) if t_ok else float("nan"),
            "tau_mean": float(np.mean([r["max_tau"] for r in runs])),
            "tau_max": float(np.max([r["max_tau"] for r in runs])),
        }
        for seed, r in enumerate(runs):
            detail_lines.append(
                f"| {method} | {seed} | {'O' if r['reached'] else 'X'} | "
                f"{r['t_goal']:.2f} | {r['max_tau']:.1f} | {r['chosen']} |")
        print(f"{method:8s}: 도달 {rows[method]['rate']:.0f}%  "
              f"시간 {rows[method]['t_mean']:.2f}s  "
              f"최대토크 평균 {rows[method]['tau_mean']:.1f} Nm")

    md = [
        "# 청크 스위칭: 채점기 vs 베이스라인 비교",
        "",
        f"phact-401 6축, S2형 시나리오 (이동 중 측면 충격 14.5~17.5 Nm, 0.3 s) "
        f"× {args.seeds} 시드.",
        "베이스라인은 DREAM-Chunk 원문의 비교 기준 차용: 무작위 선택 / 항상 첫 후보.",
        "",
        "| 방식 | 목표 도달률 | 평균 도달 시간 [s] | 최대 토크 평균 [Nm] | 최대 토크 최악 [Nm] |",
        "|---|---|---|---|---|",
    ]
    label = {"scorer": "**채점기 (본 구현)**", "random": "무작위 선택",
             "first": "항상 첫 후보"}
    for m in methods:
        r = rows[m]
        md.append(f"| {label[m]} | {r['rate']:.0f}% | {r['t_mean']:.2f} | "
                  f"{r['tau_mean']:.1f} | {r['tau_max']:.1f} |")
    md += ["", "## 시드별 상세", "",
           "| 방식 | 시드 | 도달 | 시간 [s] | 최대토크 [Nm] | 선택 청크 |",
           "|---|---|---|---|---|---|"] + detail_lines + [""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
