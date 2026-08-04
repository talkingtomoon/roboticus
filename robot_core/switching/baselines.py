"""선택기 베이스라인 — 무작위 선곡 / 항상 첫 후보 (비교표용).

MotionSelector와 같은 select() 인터페이스. 하드 필터 중 '적재 여부'만
공유한다 (미적재 재생은 즉시 코드 4 거절이라 비교 자체가 안 된다).
진입 거리 필터는 일부러 없다 — 엉뚱한 자세에서 재생하는 것이
베이스라인의 본질적 약점이고, 그게 비교표에 드러나야 한다.
"""

from __future__ import annotations

import numpy as np

from robot_core.catalog.motion_catalog import MotionCatalog
from robot_core.switching.selector import (
    CandidateRow, MissionPlan, SelectionReport,
)


class _BaselineSelector:
    def __init__(self, catalog: MotionCatalog, loaded_ids) -> None:
        self.catalog = catalog
        self.loaded = {int(i) for i in loaded_ids}

    def _pick(self, ids: list[int]) -> int:
        raise NotImplementedError

    def select(self, position, valid, dob, intent_tag: str, plan: MissionPlan,
               modifiers: tuple = ()) -> SelectionReport:
        rows = []
        playable = []
        for m in self.catalog.all():
            if m.id not in self.loaded:
                rows.append(CandidateRow(m.id, m.name, 0, 0, 0, 0, 0,
                                         excluded="not loaded"))
            else:
                playable.append(m.id)
                rows.append(CandidateRow(m.id, m.name, 0, 0, 0, 0, 0))
        best = self._pick(playable) if playable else None
        for r in rows:
            if r.motion_id == best:
                r.total = 0.0
            elif not r.excluded:
                r.total = 1.0
        return SelectionReport(rows=rows, best_id=best, intent_tag=intent_tag,
                               modifiers=tuple(modifiers))


class RandomMotionSelector(_BaselineSelector):
    def __init__(self, catalog, loaded_ids, seed: int = 0) -> None:
        super().__init__(catalog, loaded_ids)
        self._rng = np.random.default_rng(seed)

    def _pick(self, ids: list[int]) -> int:
        return int(self._rng.choice(ids))


class FirstMotionSelector(_BaselineSelector):
    """항상 (정렬상) 첫 후보 — 성능이 슬롯 번호라는 우연에 지배됨을 보여준다."""

    def _pick(self, ids: list[int]) -> int:
        return int(ids[0])
