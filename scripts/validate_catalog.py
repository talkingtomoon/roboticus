"""카탈로그·계획 정적 검증 — 교시 세션 **직후에** 돌릴 것.

교시는 되돌리기 제일 비싼 작업이다. 문제를 캠프 후반에 발견하면 다시 가르칠
시간이 없다 — 이 CLI가 교시 지침(짧게 / 변주를 / 시작자세 일관)을 자동
검사로 바꾼다.

    python scripts/validate_catalog.py --catalog catalog.json \\
        --plan approach,insert,finish

검사 항목:
  [ERROR] 계획 태그에 대응 모션 없음 / rest·retreat 부재 / 방향 없는 모션
  [WARN ] 태그 오타 의심(편집거리 1) / 회피가 한 방향뿐 /
          단계 간 시작자세 거리 초과(근사) / 모션 길이 > 반응성 기준
종료 코드: ERROR 있으면 1.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core.catalog.motion_catalog import MotionCatalog  # noqa: E402


def _edit1(a: str, b: str) -> bool:
    """편집거리 1 이내인가 (오타 의심 판정용, 소형 구현)."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:                      # 치환 1
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    for i in range(lb):               # 삽입 1
        if a == b[:i] + b[i + 1:]:
            return True
    return False


def validate(catalog: MotionCatalog, plan_tags: list[str],
             entry_max_dist: float = 0.8,
             responsiveness_s: float = 2.0) -> tuple[list[str], list[str]]:
    """(errors, warnings). 라이브 HAL 대조(reconcile)는 호출측이 별도로."""
    errors: list[str] = []
    warnings: list[str] = []
    motions = catalog.all()
    all_tags = catalog.all_tags()

    if not motions:
        return (["카탈로그가 비었다 — 교시/주석이 안 됐다"], [])

    # ---- 계획 태그 커버리지 --------------------------------------------
    for tag in plan_tags:
        if not catalog.by_tag(tag):
            errors.append(f"계획 태그 {tag!r}에 대응하는 모션이 없다")

    # ---- 안전 태그 존재 -------------------------------------------------
    for tag, why in (("rest", "과열/이상 시 휴지"), ("retreat", "충격 시 회피")):
        if not catalog.by_tag(tag):
            errors.append(f"{tag!r} 태그 모션이 없다 — {why}가 불가능해진다")

    # ---- 태그 오타 의심 -------------------------------------------------
    counts = {t: len(catalog.by_tag(t)) for t in all_tags}
    for a, b in combinations(sorted(all_tags), 2):
        if _edit1(a, b):
            rare, common = (a, b) if counts[a] <= counts[b] else (b, a)
            warnings.append(
                f"태그 오타 의심: {rare!r}({counts[rare]}개) ↔ {common!r}"
                f"({counts[common]}개) — 편집거리 1")

    # ---- 회피 방향 다양성 -----------------------------------------------
    retreats = [m for m in catalog.by_tag("retreat")
                if float(np.linalg.norm(m.initial_direction)) > 1e-6]
    if len(retreats) >= 1:
        one_sided = True
        for m1, m2 in combinations(retreats, 2):
            if float(m1.initial_direction @ m2.initial_direction) < 0.0:
                one_sided = False
                break
        if one_sided:
            warnings.append(
                f"회피(retreat) 모션 {len(retreats)}개가 전부 같은 반구 방향 — "
                f"반대로 밀리면 대항하게 된다. 양방향을 가르칠 것")

    # ---- 모션별 검사 ----------------------------------------------------
    for m in motions:
        if m.duration_s > responsiveness_s:
            warnings.append(
                f"모션 {m.id}({m.name}): {m.duration_s:.1f}s > "
                f"{responsiveness_s:.0f}s — 재생 중 개입 불가라 반응 지연이 "
                f"그만큼 길어진다. 쪼개서 가르치는 것을 권장")
        if ("rest" not in m.tags
                and float(np.linalg.norm(m.initial_direction)) < 1e-6):
            errors.append(
                f"모션 {m.id}({m.name}): initial_direction이 0 — "
                f"annotate_motion.py 재실행 필요 (순응 채점 불가)")

    # ---- 단계 간 연결 (근사) --------------------------------------------
    # 정확한 검사에는 이전 단계의 end_pose가 필요하다 (주석 확장 예정).
    # 근사: 연속 계획 태그의 모션 시작자세들끼리 최소 거리가 진입 필터
    # 임계를 넘으면, 단계 사이를 이어줄 후보가 없을 가능성이 높다.
    for tag_a, tag_b in zip(plan_tags, plan_tags[1:]):
        ms_a, ms_b = catalog.by_tag(tag_a), catalog.by_tag(tag_b)
        if not ms_a or not ms_b:
            continue
        dmin = min(float(np.linalg.norm(mb.start_pose - ma.start_pose))
                   for ma in ms_a for mb in ms_b)
        if dmin > entry_max_dist:
            warnings.append(
                f"단계 {tag_a!r}→{tag_b!r}: 시작자세 최소 거리 {dmin:.2f} > "
                f"진입 임계 {entry_max_dist} (근사 검사 — end_pose 주석 후 "
                f"정밀화). 연결이 안 될 수 있다")

    return errors, warnings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="catalog.json")
    p.add_argument("--plan", required=True, help="쉼표 구분 태그 시퀀스")
    p.add_argument("--entry-max-dist", type=float, default=0.8)
    p.add_argument("--responsiveness-s", type=float, default=2.0,
                   help="이보다 긴 모션은 반응성 경고")
    args = p.parse_args()

    catalog = MotionCatalog.from_json(args.catalog)
    plan_tags = [t.strip() for t in args.plan.split(",") if t.strip()]
    errors, warnings = validate(catalog, plan_tags,
                                entry_max_dist=args.entry_max_dist,
                                responsiveness_s=args.responsiveness_s)

    print(f"카탈로그 {args.catalog}: 모션 {len(catalog)}개, "
          f"태그 {sorted(catalog.all_tags())}")
    print(f"계획: {' → '.join(plan_tags)}")
    print()
    for e in errors:
        print(f"  [ERROR] {e}")
    for w in warnings:
        print(f"  [WARN ] {w}")
    if not errors and not warnings:
        print("  전부 통과 — 교시 세션 마감 가능")
    elif not errors:
        print(f"\nERROR 0 / WARN {len(warnings)} — 진행 가능하되 경고 검토")
    else:
        print(f"\nERROR {len(errors)} / WARN {len(warnings)} — "
              f"**교시 세션이 끝나기 전에** 고칠 것")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
