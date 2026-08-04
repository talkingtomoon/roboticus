"""구 스위칭 모듈 (스플라인 채점기·블렌더·time_scale 매니저) — 참고용."""

from robot_core.legacy.switching.scorer import (
    CandidateScore, ChunkScorer, DreamModel, ScoreReport, ScorerConfig,
    ScoreWeights, estimate_disturbance,
)
from robot_core.legacy.switching.blender import (
    BlendConfig, Blender, ChunkPlan, TransitionPlan, quintic_coeffs,
)
from robot_core.legacy.switching.manager import ChunkSwitchNode, SwitchDecision

__all__ = [
    "ChunkScorer", "ScorerConfig", "ScoreWeights", "DreamModel",
    "ScoreReport", "CandidateScore", "estimate_disturbance",
    "Blender", "BlendConfig", "TransitionPlan", "ChunkPlan", "quintic_coeffs",
    "ChunkSwitchNode", "SwitchDecision",
]
