"""스위칭 베이스라인 — DREAM-Chunk 원문 레포의 비교 기준 차용.

채점기와 같은 인터페이스(score/chunks)를 내놓지만:
- RandomChunkSelector : 무작위 후보 선택 (시드 고정 가능)
- FirstChunkSelector  : 항상 첫 후보 선택

발표용 비교표(scripts/baseline_switch_comparison.py)에서 "채점기가 정말
나은가"를 수치로 보여주는 용도다. Dream veto도 없다 — 베이스라인이니까.
"""

from __future__ import annotations

import numpy as np

from robot_core.chunks.format import MotionChunk
from robot_core.switching.scorer import CandidateScore, ScoreReport


class _SelectorBase:
    """ChunkSwitchNode가 기대하는 최소 인터페이스: .chunks, .score(...)."""

    def __init__(self, chunks: list[MotionChunk]) -> None:
        if not chunks:
            raise ValueError("need at least one candidate chunk")
        self.chunks = list(chunks)

    def _pick(self) -> int:
        raise NotImplementedError

    def score(self, q, qd, disturbance, goal) -> ScoreReport:
        best = self._pick()
        entries = [
            CandidateScore(name=c.name, connection=0.0, resistance=0.0,
                           progress=0.0, total=0.0 if i == best else 1.0,
                           vetoed=False, peak_tau_frac=0.0)
            for i, c in enumerate(self.chunks)
        ]
        return ScoreReport(entries=entries, best_index=best, elapsed_ms=0.0)


class RandomChunkSelector(_SelectorBase):
    """무작위 선택. seed 고정 시 결정적."""

    def __init__(self, chunks: list[MotionChunk], seed: int = 0) -> None:
        super().__init__(chunks)
        self._rng = np.random.default_rng(seed)

    def _pick(self) -> int:
        return int(self._rng.integers(0, len(self.chunks)))


class FirstChunkSelector(_SelectorBase):
    """항상 첫 후보 (딕셔너리 정렬 순)."""

    def _pick(self) -> int:
        return 0
