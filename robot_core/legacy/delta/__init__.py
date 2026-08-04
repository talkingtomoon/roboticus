"""잔차(Delta) 보정 — 현장 30분 캘리브레이션 파이프라인.

여기 궤적 생성 → 데이터 수집 → 모델 2벌 학습/비교 → 보정 노드 적용.
모델 파일은 항상 현장 데이터에서만 나온다 (사전 학습 금지).
"""

from robot_core.legacy.delta.excitation import (
    ExcitationConfig, ExcitationPlan, RampCycleSegment, ReversalSegment,
    SineSegment, build_excitation,
)
from robot_core.legacy.delta.collector import CalibrationData, Collector, SafetyLimits
from robot_core.legacy.delta.models import (
    MLPDeltaModel, PhysicsDeltaModel, compare_models, smooth_sign,
)
from robot_core.legacy.delta.corrector_node import DeltaCorrectorNode

__all__ = [
    "build_excitation", "ExcitationPlan", "ExcitationConfig",
    "SineSegment", "ReversalSegment", "RampCycleSegment",
    "Collector", "CalibrationData", "SafetyLimits",
    "PhysicsDeltaModel", "MLPDeltaModel", "compare_models", "smooth_sign",
    "DeltaCorrectorNode",
]
