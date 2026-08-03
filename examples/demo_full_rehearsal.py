"""전체 시스템 리허설 — 캠프 첫날 돌릴 통합 데모.

세 지름길(회복/스위칭/보정)을 한 그래프에 올리고 시나리오 5종을 순서대로
실행한다. S5의 통합 타임라인이 최종 산출물이다.

    python examples/demo_full_rehearsal.py
    python examples/demo_full_rehearsal.py --scenario S5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core.integration.scenarios import SCENARIOS, run_all  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    p.add_argument("--no-timeline", action="store_true",
                   help="판정 요약만 출력 (타임라인 생략)")
    args = p.parse_args()
    if args.scenario:
        res = SCENARIOS[args.scenario]()
        print(res.summary())
        if not args.no_timeline:
            print(res.timeline)
        sys.exit(0 if res.ok else 1)
    results = run_all(show_timeline=not args.no_timeline)
    sys.exit(0 if all(r.ok for r in results) else 1)
