"""캠프 데모용 시나리오 5종 — phorce "관찰 → 판단 → 선곡" 판.

전부 결정적: 시뮬 스텝 수 기준, 시드 고정, 벽시계 비의존.
실행: python -m robot_core.integration.scenarios [--scenario S3]

공통 월드: 12축 목 phorce 로봇 + 교시 모션 8개(접근/삽입/복귀/우회/저속/휴지).
판단 틱과 시뮬 스텝을 2Hz로 교차(tick 1회 ↔ 0.5s = 500스텝).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from robot_core.catalog.motion_catalog import MotionCatalog, MotionMeta
from robot_core.hal.phorce import MockMotion, MockPhorceHAL, N_AXES
from robot_core.integration.timeline import (
    assert_monotonic, build_timeline, format_timeline,
)
from robot_core.logging.feedback_cache import FeedbackCache
from robot_core.recovery import (
    DetectorConfig, FailureDetector, FailureType, LLMConfig, LLMRecoveryAgent,
    TagRuleFallback, TagSafetyGuard,
)
from robot_core.supervisor import Supervisor, SupervisorConfig, SupervisorState
from robot_core.switching.selector import MissionPlan, MotionSelector

# ------------------------------------------------------------------ 월드 정의
HOME = np.zeros(N_AXES)
POSE_A = np.zeros(N_AXES); POSE_A[:3] = [0.5, -0.3, 0.6]     # 접근 완료 자세
POSE_B = POSE_A.copy(); POSE_B[3] = 0.4                       # 삽입 완료 자세
POSE_SIDE = POSE_A.copy(); POSE_SIDE[0] += 0.35               # 옆으로 비켜선 자세

TICK_STEPS = 500          # 판단 틱당 시뮬 스텝 (0.5s = 2Hz)
IMPACT_DOB = 5.0          # [A] 충격 크기 (임계 3.0의 1.7배)


def _minjerk(p0, p1, T):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)

    def fn(t):
        s = min(max(t / T, 0.0), 1.0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        return p0 + s * (p1 - p0)
    return fn


def _hold(p):
    p = np.asarray(p, float)
    return lambda t: p


def _meta(mid, name, tags, p0, p1, T, notes=""):
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = float(np.linalg.norm(d))
    return MotionMeta(mid, name, tags, p0, d / n if n > 1e-6 else d, T, notes,
                      end_pose=np.asarray(p1, float))


def build_world(*, client=None, selector_cls=MotionSelector,
                plan_tags=("approach", "insert", "finish"), seed: int = 0,
                dob_threshold: float = 3.0):
    """목 로봇 + 카탈로그 + 전체 스택 조립. 시나리오/테스트/비교표 공용."""
    mock = {
        1: MockMotion(1.0, _minjerk(HOME, POSE_A, 1.0)),
        2: MockMotion(1.0, _minjerk(POSE_A, POSE_B, 1.0)),
        3: MockMotion(0.8, _minjerk(POSE_B, HOME, 0.8)),
        4: MockMotion(1.5, _hold(POSE_A)),                       # rest (홀드)
        5: MockMotion(1.2, _minjerk(POSE_A, POSE_SIDE, 1.2)),    # 우회(퇴피)
        6: MockMotion(1.0, _minjerk(POSE_SIDE, POSE_A, 1.0)),    # 복귀
        7: MockMotion(2.0, _minjerk(POSE_A, POSE_B, 2.0)),       # 저속 삽입
        8: MockMotion(1.6, _minjerk(HOME, POSE_A, 1.6)),         # 저속 접근
    }
    hal = MockPhorceHAL(mock)

    catalog = MotionCatalog([
        _meta(1, "approach", ["approach"], HOME, POSE_A, 1.0),
        _meta(2, "insert", ["insert"], POSE_A, POSE_B, 1.0),
        _meta(3, "go_home", ["finish"], POSE_B, HOME, 0.8),
        _meta(4, "rest_hold", ["rest"], POSE_A, POSE_A, 1.5),
        _meta(5, "side_step", ["retreat"], POSE_A, POSE_SIDE, 1.2),
        _meta(6, "side_return", ["approach", "retry"], POSE_SIDE, POSE_A, 1.0),
        _meta(7, "insert_slow", ["insert", "retry", "slow"], POSE_A, POSE_B, 2.0),
        _meta(8, "approach_slow", ["approach", "slow"], HOME, POSE_A, 1.6),
    ])
    usable, warns = catalog.reconcile(hal.catalog())

    cache = FeedbackCache()
    detector = FailureDetector(N_AXES, DetectorConfig(
        impact_dob_threshold=dob_threshold,
        overheat_temp_c=70.0, overheat_release_c=62.0))
    if selector_cls is MotionSelector:
        selector = MotionSelector(catalog, usable, entry_max_dist=1.0)
    else:
        selector = selector_cls(catalog, usable, seed=seed) \
            if selector_cls.__name__ == "RandomMotionSelector" \
            else selector_cls(catalog, usable)
    guard = TagSafetyGuard(catalog.all_tags(), time_fn=lambda: hal.t)
    agent = LLMRecoveryAgent(
        guard, TagRuleFallback(), cache=cache,
        config=LLMConfig(timeout_s=2.0, cooldown_s=2.0), client=client,
        time_fn=lambda: hal.t)
    plan = MissionPlan(list(plan_tags))
    sup = Supervisor(hal, cache, detector, selector, agent, catalog, plan,
                     guard=guard)
    for w in warns:
        cache.mark_event("catalog: " + w)
    return hal, cache, sup, plan, guard


def run_ticks(hal, sup, n_ticks: int, on_tick=None):
    for k in range(n_ticks):
        if on_tick:
            on_tick(k)
        sup.tick()
        hal.step(TICK_STEPS)


def scripted_llm(system: str, user: str) -> str:
    """리허설용 결정적 LLM 대역 — 실패 타입을 프롬프트에서 읽어 태그 선정."""
    if "PLAYBACK_STALL" in user:
        out = {"intent_tag": "retry", "urgency": "slow",
               "reasoning": "blocked mid-motion; retry slowly", "confidence": 0.85}
    elif "IMPACT" in user:
        out = {"intent_tag": "retreat", "urgency": "normal",
               "reasoning": "external impact; step aside", "confidence": 0.9}
    elif "OVERHEAT" in user:
        out = {"intent_tag": "rest", "urgency": "stop",
               "reasoning": "thermal budget exceeded; rest", "confidence": 0.9}
    else:
        out = {"intent_tag": "rest", "urgency": "stop",
               "reasoning": "unknown failure; hold safe", "confidence": 0.5}
    return json.dumps(out)


# ---------------------------------------------------------------- 결과 구조
@dataclass
class ScenarioResult:
    name: str
    title: str
    checks: list = field(default_factory=list)
    timeline: str = ""
    notes: list = field(default_factory=list)

    def check(self, desc: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((desc, bool(ok), detail))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def summary(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        lines = [f"[{self.name}] {self.title}  ->  {mark}"]
        for desc, ok, detail in self.checks:
            lines.append(f"  {'v' if ok else 'X'} {desc}"
                         + (f"  ({detail})" if detail else ""))
        for n in self.notes:
            lines.append(f"  - {n}")
        return lines and "\n".join(lines)


def _finish(r: ScenarioResult, cache, guard) -> ScenarioResult:
    entries = build_timeline(logger=cache, guard=guard)
    assert_monotonic(entries)
    r.timeline = format_timeline(entries, title=f"{r.name} TIMELINE")
    return r


def _texts(cache) -> list[str]:
    return [e.text for e in cache.events()]


# ---------------------------------------------------------------- S1: 평화
def s1_peace() -> ScenarioResult:
    r = ScenarioResult("S1", "평화 — 모션 3개 순차 재생, 오발동 0")
    hal, cache, sup, plan, guard = build_world()
    run_ticks(hal, sup, 14)

    texts = _texts(cache)
    r.check("계획 완주 (approach→insert→finish)", plan.done and
            sup.state == SupervisorState.DONE, f"cursor={plan.cursor}")
    r.check("감지 이벤트 0", not any(t.startswith("DETECTED") for t in texts))
    r.check("LLM 질의 0", sup.agent.stats.get("submitted", 0) == 0)
    r.check("BUSY 폭주 0 (재생 중 play 시도 없음)", sup.busy_rejections == 0)
    q = hal.latest_feedback().position_rad
    r.check("최종 자세 = HOME", float(np.abs(q - HOME).max()) < 0.05,
            f"|err|max={np.abs(q - HOME).max():.3f}")
    return _finish(r, cache, guard)


# ------------------------------------------------------- S2: 충격 → 우회
def s2_impact_retreat() -> ScenarioResult:
    r = ScenarioResult("S2", "충격→우회 — dob_a 이력 보고 retreat 태그 선곡")
    hal, cache, sup, plan, guard = build_world()

    def on_tick(k):
        if k == 3:   # motion 1 재생 중간(t≈1.5s 구간)에 측면 충격
            hal.inject_disturbance(0, IMPACT_DOB, duration=0.25)
            cache.mark_event(f"INJECT impact dob={IMPACT_DOB:+.1f} A on axis 0",
                             t=hal.t)
    run_ticks(hal, sup, 20, on_tick)

    texts = _texts(cache)
    r.check("IMPACT 감지", any("DETECTED IMPACT" in t for t in texts))
    r.check("규칙 폴백 → retreat 태그", any(
        "intent='retreat'" in t for t in texts))
    r.check("우회 모션(side_step) 재생", any(
        "play: motion 5" in t for t in texts))
    r.check("이후 계획 복귀·완주", plan.done, f"cursor={plan.cursor}")
    r.check("BUSY 폭주 0", sup.busy_rejections == 0)
    return _finish(r, cache, guard)


# ------------------------------------------------------ S3: 막힘 → 재시도
def s3_stall_retry() -> ScenarioResult:
    r = ScenarioResult("S3", "막힘→재시도 — PLAYBACK_STALL → LLM 경로 → slow 재선곡")
    hal, cache, sup, plan, guard = build_world(client=scripted_llm)

    # 삽입(motion 2)이 움직이는 축(3)을 물체가 물리적으로 막는다.
    # 지속 '힘'으로는 스톨이 안 난다 — PD는 오프셋 진 채 계속 추종한다.
    # 진행 자체가 불가능한 상황 = jam이 맞는 물리다.
    def on_tick(k):
        if k == 4:                     # t=2.0, insert(1.5..2.5s) 한복판
            hal.inject_jam(3)
            cache.mark_event("INJECT jam on axis 3 (object blocks insertion)",
                             t=hal.t)
        if k == 8:                     # t=4.0, 물체가 치워짐
            hal.clear_jam(3)
            cache.mark_event("jam cleared (obstacle removed)", t=hal.t)
    run_ticks(hal, sup, 26, on_tick)

    texts = _texts(cache)
    r.check("PLAYBACK_STALL 감지", any("DETECTED PLAYBACK_STALL" in t for t in texts))
    r.check("LLM 경로 사용 (source=llm)", any(
        "recovery[llm]" in t and "retry" in t for t in texts))
    r.check("slow 변주(insert_slow=7) 재선곡", any(
        "play: motion 7" in t for t in texts))
    r.check("막힘 해제 후 계획 완주", plan.done, f"cursor={plan.cursor}")
    r.check("BUSY 폭주 0", sup.busy_rejections == 0)
    return _finish(r, cache, guard)


# ----------------------------------------------------------- S4: 과열
def s4_overheat_rest() -> ScenarioResult:
    r = ScenarioResult("S4", "과열 — OVERHEAT → 휴지 삽입 → 해제 후 복귀")
    hal, cache, sup, plan, guard = build_world()

    def on_tick(k):
        if k == 3:                       # 재생 중 과열 주입
            for ax in (1, 2):
                hal.set_temperature(ax, 75.0)
            cache.mark_event("INJECT overheat 75C on axes 1,2", t=hal.t)
        if k == 9:                       # 냉각 (해제 임계 62C 아래로)
            for ax in (1, 2):
                hal.set_temperature(ax, 55.0)
            cache.mark_event("cooldown to 55C", t=hal.t)
    run_ticks(hal, sup, 26, on_tick)

    texts = _texts(cache)
    r.check("OVERHEAT 감지", any("DETECTED OVERHEAT joint" in t
                              or "DETECTED OVERHEAT " in t for t in texts))
    r.check("휴지(rest_hold=4) 삽입", any("play: motion 4" in t for t in texts))
    r.check("해제(OVERHEAT_CLEARED) 수신", any(
        "OVERHEAT_CLEARED" in t for t in texts))
    r.check("휴지 해제 후 계획 복귀·완주", plan.done, f"cursor={plan.cursor}")
    r.check("BUSY 폭주 0", sup.busy_rejections == 0)
    return _finish(r, cache, guard)


# ----------------------------------------------------------- S5: 총력전
def s5_full_battle() -> ScenarioResult:
    r = ScenarioResult("S5", "총력전 — 충격→우회 + 거절 12→WAITING_OPERATOR + 완주")
    hal, cache, sup, plan, guard = build_world(client=scripted_llm)

    def on_tick(k):
        if k == 3:
            hal.inject_disturbance(0, IMPACT_DOB, duration=0.25)
            cache.mark_event(f"INJECT impact dob={IMPACT_DOB:+.1f} A on axis 0",
                             t=hal.t)
        if k == 7:                       # 다음 선곡 직전에 NOT_READY 거절 무장
            hal.set_rejection(12)
            cache.mark_event("INJECT rejection code 12 (NOT_READY)", t=hal.t)
    run_ticks(hal, sup, 12, on_tick)

    waited = sup.state == SupervisorState.WAITING_OPERATOR
    r.check("1단계: IMPACT → retreat 우회", any(
        "play: motion 5" in t for t in _texts(cache)))
    r.check("2단계: 거절 12 → WAITING_OPERATOR", waited, f"state={sup.state}")

    plays_before = sum(1 for t in _texts(cache) if t.startswith("play: motion"))
    run_ticks(hal, sup, 4)              # 대기 중 — 자동 재시도 없어야 함
    plays_after = sum(1 for t in _texts(cache) if t.startswith("play: motion"))
    r.check("대기 중 자동 재시도 0", plays_after == plays_before)

    hal.clear_rejection()               # 사람이 영점 버튼을 눌렀다
    sup.operator_cleared()
    cache.mark_event("operator: zero-button done (human)", t=hal.t)
    run_ticks(hal, sup, 18)

    r.check("개입 후 계획 완주", plan.done and sup.state == SupervisorState.DONE,
            f"cursor={plan.cursor} state={sup.state}")
    r.check("BUSY 폭주 0", sup.busy_rejections == 0)

    _finish(r, cache, guard)
    texts = r.timeline
    sources = {line.split("]")[1].split("|")[0].strip()
               for line in texts.splitlines() if line.startswith("[t=")}
    r.check("(단일 타임라인) 소스 다양성", {"detect", "recovery", "select",
                                     "play", "operator"} <= sources,
            f"sources={sorted(sources)}")
    return r


SCENARIOS = {
    "S1": s1_peace,
    "S2": s2_impact_retreat,
    "S3": s3_stall_retry,
    "S4": s4_overheat_rest,
    "S5": s5_full_battle,
}


def run_all(show_timeline: bool = True) -> list[ScenarioResult]:
    results = []
    for name, fn in SCENARIOS.items():
        print(f"\n{'#' * 74}\n# {name} 실행 중...\n{'#' * 74}")
        res = fn()
        results.append(res)
        print(res.summary())
        if show_timeline and (name == "S5" or not res.ok):
            print()
            print(res.timeline)
    print("\n" + "=" * 74)
    n_ok = sum(1 for r in results if r.ok)
    print(f"통합 리허설 결과: {n_ok}/{len(results)} 시나리오 통과"
          + ("  — 전체 시스템 정상" if n_ok == len(results) else "  ** 실패 있음"))
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    p.add_argument("--no-timeline", action="store_true")
    args = p.parse_args()
    if args.scenario:
        res = SCENARIOS[args.scenario]()
        print(res.summary())
        if not args.no_timeline:
            print(res.timeline)
        raise SystemExit(0 if res.ok else 1)
    results = run_all(show_timeline=not args.no_timeline)
    raise SystemExit(0 if all(r.ok for r in results) else 1)
