"""모션 선택기 — 스플라인 채점기의 phorce 판. 출력은 궤적이 아니라 motion_id.

채점 항목 (가중합, 낮을수록 좋음):
  (a) tag        : 요청된 의도 태그와 모션 태그 매칭 (불일치 = 1)
                   + modifier 태그(예: "slow") 보유 시 감점 혜택
  (b) entry      : 현재 자세 ↔ 모션 시작 자세 거리 (valid 축만)
  (c) compliance : 마지막 dob_a 방향과 모션 초기 이동 방향의 정합
                   (1-cos)/2 — 순응 0 < 무관 0.5 < 정면 대항 1
  (d) progress   : 임무 계획(태그 시퀀스)에서 목표까지 남는 단계 수

dream veto는 없다 — 토크 예측이 불가능한 인터페이스다. 대신 하드 필터:
  - 적재 안 된 id 제외 (재생하면 코드 4 거절 — 애초에 후보에서 뺀다)
  - 시작 자세 거리 > entry_max_dist 제외 (엉뚱한 자세에서 재생 방지.
    MotionAborted 후 임의 자세 시나리오를 이 필터가 받는다)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_core.catalog.motion_catalog import MotionCatalog, MotionMeta


@dataclass
class SelectorWeights:
    tag: float = 2.0
    entry: float = 1.0
    compliance: float = 1.0
    progress: float = 0.6
    modifier_bonus: float = 0.5   # modifier 태그(예: "slow") 일치당 감점


@dataclass
class MissionPlan:
    """임무 = 순서 있는 의도 태그 시퀀스. 그 태그를 단 모션이 완주하면 전진."""

    tag_sequence: list[str]
    cursor: int = 0

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.tag_sequence)

    @property
    def next_tag(self) -> str | None:
        return None if self.done else self.tag_sequence[self.cursor]

    def remaining_after(self, motion: MotionMeta) -> int:
        rem = len(self.tag_sequence) - self.cursor
        return rem - 1 if (self.next_tag in motion.tags) else rem

    def advance_if_matches(self, motion: MotionMeta) -> bool:
        if not self.done and self.next_tag in motion.tags:
            self.cursor += 1
            return True
        return False


@dataclass
class CandidateRow:
    motion_id: int
    name: str
    tag_cost: float
    entry_cost: float
    compliance_cost: float
    progress_cost: float
    total: float
    excluded: str | None = None    # 하드 필터 사유 (None = 후보 생존)

    def line(self) -> str:
        if self.excluded:
            return f"{self.motion_id:3d} {self.name:20s} EXCLUDED: {self.excluded}"
        return (f"{self.motion_id:3d} {self.name:20s} total={self.total:6.3f}  "
                f"tag={self.tag_cost:4.2f} entry={self.entry_cost:5.3f} "
                f"comply={self.compliance_cost:4.2f} prog={self.progress_cost:4.2f}")


@dataclass
class SelectionReport:
    rows: list[CandidateRow]
    best_id: int | None
    intent_tag: str
    modifiers: tuple = ()

    @property
    def best_row(self) -> CandidateRow | None:
        if self.best_id is None:
            return None
        return next(r for r in self.rows if r.motion_id == self.best_id)

    def table(self, top: int | None = None) -> str:
        alive = sorted([r for r in self.rows if not r.excluded], key=lambda r: r.total)
        dead = [r for r in self.rows if r.excluded]
        lines = [f"[SELECT] intent={self.intent_tag!r}"
                 + (f" mod={list(self.modifiers)}" if self.modifiers else "")
                 + f"  candidates={len(alive)} excluded={len(dead)}"]
        for i, r in enumerate(alive[: top or len(alive)], 1):
            mark = " <-- chosen" if r.motion_id == self.best_id else ""
            lines.append(f"  {i:2d}. {r.line()}{mark}")
        for r in dead:
            lines.append(f"   x. {r.line()}")
        return "\n".join(lines)


class MotionSelector:
    """카탈로그 + 적재 정본으로 후보를 채점해 다음 motion_id를 고른다."""

    def __init__(self, catalog: MotionCatalog, loaded_ids,
                 weights: SelectorWeights | None = None,
                 entry_max_dist: float = 0.8) -> None:
        self.catalog = catalog
        self.loaded = {int(i) for i in loaded_ids}
        self.w = weights or SelectorWeights()
        self.entry_max_dist = float(entry_max_dist)

    def select(self, position: np.ndarray, valid: np.ndarray,
               dob: np.ndarray, intent_tag: str, plan: MissionPlan,
               modifiers: tuple = ()) -> SelectionReport:
        """position/valid/dob: 마지막 valid 피드백에서. intent_tag: LLM/규칙/계획."""
        position = np.asarray(position, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        dob = np.where(valid, np.asarray(dob, dtype=float), 0.0)
        dob_norm = float(np.linalg.norm(dob))
        total_steps = max(1, len(plan.tag_sequence))

        rows: list[CandidateRow] = []
        for m in self.catalog.all():
            # ---- 하드 필터 ----
            if m.id not in self.loaded:
                rows.append(CandidateRow(m.id, m.name, 0, 0, 0, 0, 0,
                                         excluded="not loaded (code 4 예방)"))
                continue
            diff = np.where(valid, m.start_pose - position, 0.0)
            entry_dist = float(np.linalg.norm(diff))
            if entry_dist > self.entry_max_dist:
                rows.append(CandidateRow(
                    m.id, m.name, 0, entry_dist, 0, 0, 0,
                    excluded=f"entry dist {entry_dist:.2f} > {self.entry_max_dist}"))
                continue

            # ---- 채점 ----
            tag_cost = 0.0 if intent_tag in m.tags else 1.0
            tag_cost -= self.w.modifier_bonus * sum(1 for t in modifiers if t in m.tags)

            if dob_norm > 1e-6:
                cos = float(m.initial_direction @ (dob / dob_norm))
                comply = (1.0 - cos) / 2.0
            else:
                comply = 0.0

            prog = plan.remaining_after(m) / total_steps

            total = (self.w.tag * tag_cost + self.w.entry * entry_dist
                     + self.w.compliance * comply + self.w.progress * prog)
            rows.append(CandidateRow(m.id, m.name, tag_cost, entry_dist,
                                     comply, prog, total))

        alive = [r for r in rows if not r.excluded]
        best = min(alive, key=lambda r: r.total).motion_id if alive else None
        return SelectionReport(rows=rows, best_id=best, intent_tag=intent_tag,
                               modifiers=tuple(modifiers))
