"""목 리허설 = 전체 파이프라인 회귀 테스트.

현장 직전에 한 번 더 돌려볼 테스트:
    python -m pytest tests/test_field_rehearsal.py -v

MockRobotHAL에 알려진 마찰(tau_c=0.4, b=0.15)과 백래시(0.02)를 심고
원커맨드 스크립트 전체를 실행해:
  1) 물리 모델이 심어둔 파라미터를 10% 이내로 복원하는지
  2) 보정 on 시 추종 오차가 off 대비 유의미하게 감소하는지 (>30%)
  3) --resume 이 수집을 건너뛰고 재학습만 하는지
를 고정한다.
"""

import argparse
import re

import numpy as np
import pytest

from scripts.field_calibration import run_pipeline

# make_hal("mock")에 심어둔 값과 일치해야 한다
TAU_C, B = 0.4, 0.15


def make_args(tmp_path, budget_min=2.0, resume=False):
    # 관절당 40초 — 현장 기본(8분/3관절 = 160초/관절)의 축소판.
    # 1분(관절당 20초)은 파라미터 복원이 10% 경계에 걸린다.
    return argparse.Namespace(
        hal="mock", budget_min=budget_min, joints="0,1,2",
        out_dir=str(tmp_path / "calib_out"), resume=resume, yes=True, seed=0)


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    """전체 파이프라인 1회 실행 (모듈 공유 — 제일 비싼 픽스처)."""
    tmp = tmp_path_factory.mktemp("rehearsal")
    result = run_pipeline(make_args(tmp))
    result["tmp"] = tmp
    return result


def test_pipeline_completes(pipeline_result):
    assert pipeline_result["status"] == "ok"
    out_dir = pipeline_result["out_dir"]
    assert (out_dir / "calibration_data.npz").exists()
    assert (out_dir / "physics_model.npz").exists()
    assert (out_dir / "report.txt").exists()


def test_physics_recovers_planted_friction(pipeline_result):
    """심어둔 tau_c, b 복원 오차 < 10% (요구사항)."""
    physics = pipeline_result["physics"]
    for j in range(3):
        p = physics.params[j]
        assert abs(p.coulomb - TAU_C) / TAU_C < 0.10, \
            f"joint {j}: tau_c {p.coulomb:.4f} vs {TAU_C}"
        assert abs(p.viscous - B) / B < 0.10, \
            f"joint {j}: b {p.viscous:.4f} vs {B}"


def test_correction_reduces_tracking_error(pipeline_result):
    """검증 리포트의 관절별 개선율이 전부 +30% 이상."""
    report = pipeline_result["validation"]
    improvements = [float(m) for m in re.findall(r"([+-]\d+\.\d+)%", report)]
    assert len(improvements) == 3
    assert all(imp > 30.0 for imp in improvements), report


def test_corrector_inference_within_budget(pipeline_result):
    assert "OK" in pipeline_result["validation"].split("budget")[1].splitlines()[0]


def test_resume_reuses_collected_data(pipeline_result):
    """--resume: 수집을 건너뛰고 데이터 파일을 재사용한다."""
    tmp = pipeline_result["tmp"]
    data_path = pipeline_result["out_dir"] / "calibration_data.npz"
    mtime_before = data_path.stat().st_mtime_ns

    result2 = run_pipeline(make_args(tmp, resume=True))
    assert result2["status"] == "ok"
    assert data_path.stat().st_mtime_ns == mtime_before, "resume인데 데이터를 다시 수집했다"
    # 재학습 결과도 동일 수준으로 복원
    p = result2["physics"].params[0]
    assert abs(p.coulomb - TAU_C) / TAU_C < 0.10
