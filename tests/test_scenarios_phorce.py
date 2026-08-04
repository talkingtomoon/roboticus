"""통합 리허설 — 시나리오 5종 자동 판정 + 단일 타임라인."""

from robot_core.integration.scenarios import (
    s1_peace, s2_impact_retreat, s3_stall_retry, s4_overheat_rest, s5_full_battle,
)


def _assert_ok(result):
    assert result.ok, "\n" + result.summary() + "\n\n" + result.timeline


def test_s1_peace():
    _assert_ok(s1_peace())


def test_s2_impact_retreat():
    _assert_ok(s2_impact_retreat())


def test_s3_stall_retry_llm_path():
    _assert_ok(s3_stall_retry())


def test_s4_overheat_rest_and_resume():
    _assert_ok(s4_overheat_rest())


def test_s5_full_battle_single_timeline():
    result = s5_full_battle()
    _assert_ok(result)
    assert "TIMELINE" in result.timeline
    assert "WAITING_OPERATOR" in result.timeline or "operator" in result.timeline
