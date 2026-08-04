"""선택기 비교: 지능 선곡 vs 무작위 vs 항상-첫-슬롯 (발표 자료 원본).

S2형 시나리오(재생 중 충격 → 다음 경계에서 재선곡)를 3방식 × N시드로 돌려
계획 완주율 / 완주 틱 수 / 헛선곡(계획 밖 재생) 횟수를 비교한다.

    python scripts/baseline_selector_comparison.py --seeds 7
결과: docs/baseline_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core.integration.scenarios import (   # noqa: E402
    IMPACT_DOB, TICK_STEPS, build_world,
)
from robot_core.supervisor import SupervisorState  # noqa: E402
from robot_core.switching.baselines import (       # noqa: E402
    FirstMotionSelector, RandomMotionSelector,
)
from robot_core.switching.selector import MotionSelector  # noqa: E402

MAX_TICKS = 30


def run_once(selector_cls, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    impact_tick = 2 + int(rng.integers(0, 3))
    impact_mag = IMPACT_DOB + float(rng.uniform(-0.5, 1.5))

    hal, cache, sup, plan, _ = build_world(selector_cls=selector_cls, seed=seed)
    ticks_done = MAX_TICKS
    for k in range(MAX_TICKS):
        if k == impact_tick:
            hal.inject_disturbance(int(rng.integers(0, 4)), impact_mag, 0.25)
        sup.tick()
        hal.step(TICK_STEPS)
        if plan.done and sup.state == SupervisorState.DONE:
            ticks_done = k + 1
            break

    plays = [t for t in (e.text for e in cache.events())
             if t.startswith("play: motion")]
    return {"done": plan.done, "ticks": ticks_done, "plays": len(plays),
            "busy": sup.busy_rejections}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=7)
    p.add_argument("--out", default="docs/baseline_comparison.md")
    args = p.parse_args()

    methods = [("selector", MotionSelector), ("random", RandomMotionSelector),
               ("first", FirstMotionSelector)]
    rows = {}
    detail = []
    for name, cls in methods:
        runs = [run_once(cls, s) for s in range(args.seeds)]
        rate = 100.0 * sum(r["done"] for r in runs) / len(runs)
        t_ok = [r["ticks"] for r in runs if r["done"]]
        rows[name] = {
            "rate": rate,
            "ticks": float(np.mean(t_ok)) if t_ok else float("nan"),
            "plays": float(np.mean([r["plays"] for r in runs])),
        }
        for s, r in enumerate(runs):
            detail.append(f"| {name} | {s} | {'O' if r['done'] else 'X'} | "
                          f"{r['ticks']} | {r['plays']} |")
        print(f"{name:9s}: 완주 {rate:.0f}%  평균 {rows[name]['ticks']:.1f}틱  "
              f"재생 {rows[name]['plays']:.1f}회")

    label = {"selector": "**선택기 (본 구현)**", "random": "무작위 선곡",
             "first": "항상 첫 슬롯"}
    md = [
        "# 모션 선곡: 선택기 vs 베이스라인 비교",
        "",
        f"phorce 목 12축, S2형 시나리오 (재생 중 충격 → 경계 재선곡) × {args.seeds} 시드.",
        f"판단 2Hz, 최대 {MAX_TICKS}틱({MAX_TICKS // 2}초) 안에 계획"
        f"(approach→insert→finish) 완주 여부.",
        "",
        "| 방식 | 계획 완주율 | 평균 완주 틱 | 평균 재생 횟수 |",
        "|---|---|---|---|",
    ] + [f"| {label[m]} | {rows[m]['rate']:.0f}% | {rows[m]['ticks']:.1f} | "
         f"{rows[m]['plays']:.1f} |" for m, _ in methods] + [
        "",
        "베이스라인은 태그/진입 자세/외란을 전부 무시한다 — 엉뚱한 자세에서",
        "엉뚱한 모션을 트는 것이 수치로 드러난다.",
        "",
        "## 시드별 상세", "",
        "| 방식 | 시드 | 완주 | 틱 | 재생 횟수 |", "|---|---|---|---|---|",
    ] + detail + [""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
