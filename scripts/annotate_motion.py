"""교시 직후 모션 메타데이터 반자동 생성 — 실물에서 1회 재생으로 주석 완성.

phorce Studio로 교시한 모션은 SD카드에만 있다. 선곡에 필요한 메타데이터
(start_pose, initial_direction, duration)는 이 스크립트가 재생 1회를 녹화해
자동 추출하고, 이름/태그만 사람이 입력한다.

    python scripts/annotate_motion.py --id 3 --name approach --tags approach,slow \\
        --catalog catalog.json

    # 목으로 흐름 리허설:
    python scripts/annotate_motion.py --mock --id 1 --name test --tags approach
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core.catalog.motion_catalog import (   # noqa: E402
    MotionCatalog, annotate_from_recording,
)
from robot_core.logging.feedback_cache import FeedbackCache  # noqa: E402


def make_hal(mock: bool):
    if mock:
        from robot_core.hal.phorce import MockMotion, MockPhorceHAL, N_AXES
        home = np.zeros(N_AXES)
        pose = np.zeros(N_AXES); pose[:3] = [0.4, -0.2, 0.5]

        def traj(t):
            s = min(t / 1.0, 1.0)
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            return home + s * (pose - home)
        return MockPhorceHAL({1: MockMotion(1.0, traj)})
    # TODO(현장): phorce 파사드 래퍼로 교체
    #   from robot_core.hal.phorce_real import RealPhorceHAL
    #   return RealPhorceHAL(phorce.connect())
    raise SystemExit("실물 HAL 미구현 — 현장에서 RealPhorceHAL 연결 후 사용. "
                     "목 리허설은 --mock")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", type=int, required=True, help="모션 슬롯 id (1~50)")
    p.add_argument("--name", required=True)
    p.add_argument("--tags", required=True, help="쉼표 구분 (예: approach,slow)")
    p.add_argument("--notes", default="")
    p.add_argument("--catalog", default="catalog.json", help="갱신할 카탈로그 JSON")
    p.add_argument("--mock", action="store_true", help="목으로 흐름 리허설")
    args = p.parse_args()

    hal = make_hal(args.mock)
    loaded = hal.catalog()
    if args.id not in loaded:
        raise SystemExit(f"슬롯 {args.id}가 적재돼 있지 않다. 적재 목록: "
                         f"{sorted(loaded)}")

    cache = FeedbackCache()
    hal.watch(cache.push)

    print(f"모션 {args.id} 재생 + 녹화 중... (로봇이 움직인다 — 주변 확인)")
    hal.play(args.id)          # 블로킹: 완주까지
    if args.mock:
        hal.step(50)           # 꼬리 프레임 몇 개

    meta = annotate_from_recording(
        args.id, args.name, [t.strip() for t in args.tags.split(",") if t.strip()],
        cache.to_arrays(window_sec=None), notes=args.notes)

    path = Path(args.catalog)
    cat = MotionCatalog.from_json(path) if path.exists() else MotionCatalog()
    if args.id in cat:
        print(f"기존 항목 {args.id} 교체")
        cat._motions.pop(args.id)
    cat.add(meta)
    cat.to_json(path)

    print(f"\n주석 완성 → {path}")
    print(f"  name={meta.name!r} tags={meta.tags}")
    print(f"  duration={meta.duration_s:.2f}s")
    print(f"  start_pose[:4]={np.round(meta.start_pose[:4], 3)}")
    print(f"  initial_direction[:4]={np.round(meta.initial_direction[:4], 3)}")


if __name__ == "__main__":
    main()
