"""ChunkSwitchNode — 상태기계, 쿨다운, 감사 로그, MockRobotHAL 통합."""

import numpy as np
import pytest

from robot_core import ControlLoopRunner, JointState, MockRobotHAL, RingLogger
from robot_core.chunks import generator as G
from robot_core.graph import ImpedanceNode, NodeGraphManager
from robot_core.recovery import DetectorConfig, FailureDetector, FailureType
from robot_core.switching import (
    Blender, ChunkScorer, ChunkSwitchNode, DreamModel, ScorerConfig,
    estimate_disturbance,
)

N = 3
START = np.zeros(N)
GOAL = np.array([0.8, 0.0, 0.4])


def make_dictionary():
    direct = G.min_jerk("direct", START, GOAL, 1.2, tags=["approach"])
    return [
        direct,
        G.with_detour(direct, t_via=0.6, offset=np.array([0.0, +0.35, 0.0]),
                      name="detour_up"),
        G.with_detour(direct, t_via=0.6, offset=np.array([0.0, -0.35, 0.0]),
                      name="detour_down"),
        G.time_scaled(direct, 1.6, name="direct_slow"),
        G.retreat("retreat", GOAL * 0.5, START, 1.0),
    ]


def make_node(hal=None, cooldown=0.4, with_dream=True):
    hal = hal or MockRobotHAL(n_joints=N, dt=1e-3, torque_limit=20.0)
    dream = DreamModel.from_mock_hal(hal, kp=60.0, kd=2.5) if with_dream else None
    # entry_dir_window을 우회 범프가 보이는 길이로 (범프는 t=0.3부터 시작)
    scorer = ChunkScorer(make_dictionary(), dream=dream,
                         config=ScorerConfig(entry_dir_window_s=0.6))
    node = ChunkSwitchNode(params={"cooldown_s": cooldown}, scorer=scorer, goal=GOAL)
    return node, hal


def fake_state(q, qd, t):
    q = np.asarray(q, dtype=float)
    return JointState(q=q, qd=np.asarray(qd, dtype=float),
                      tau_measured=np.zeros_like(q), timestamp=float(t))


# ------------------------------------------------------------ 기본 거동
def test_idle_without_active_chunk_holds_current_pose():
    node, _ = make_node()
    out = node.update({"state": fake_state([0.3, -0.1, 0.2], np.zeros(N), 0.0)})
    assert np.allclose(out["q_des"], [0.3, -0.1, 0.2])
    assert np.allclose(out["qd_des"], 0.0)
    assert node.phase == "IDLE"


def test_executing_follows_chunk_then_holds_end():
    node, _ = make_node()
    node.set_active(G.min_jerk("d", START, GOAL, 1.0), t_now=0.0)
    out_mid = node.update({"state": fake_state(START, np.zeros(N), 0.5)})
    assert 0.0 < out_mid["q_des"][0] < 0.8
    assert node.phase == "EXECUTING"

    out_end = node.update({"state": fake_state(GOAL, np.zeros(N), 5.0)})
    assert np.allclose(out_end["q_des"], GOAL)
    assert np.allclose(out_end["qd_des"], 0.0)
    assert node.phase == "DONE"


def test_switch_request_triggers_blend_and_audit():
    node, _ = make_node()
    node.set_active(G.min_jerk("direct", START, GOAL, 1.2), t_now=0.0)
    node.update({"state": fake_state(START, np.zeros(N), 0.3)})

    node.request_switch(np.array([0.0, 6.0, 0.0]))
    node.update({"state": fake_state(START, np.zeros(N), 0.301)})

    assert node.phase == "BLENDING"
    assert len(node.decisions) == 1
    d = node.decisions[0]
    assert d.chosen is not None
    assert d.report is not None and len(d.report.entries) == 5
    assert "SWITCH to" in d.line()
    text = node.dump_decisions()
    assert "[SWITCH AUDIT]" in text and "total=" in text


def test_reference_continuity_across_switch_tick():
    """스위칭이 일어난 틱 전후로 참조 신호가 C1 연속이어야 한다 (핵심 요구사항)."""
    node, _ = make_node()
    node.set_active(G.min_jerk("direct", START, GOAL, 1.2), t_now=0.0)

    dt = 1e-3
    prev = None
    max_pos_gap = 0.0
    max_vel_jump = 0.0
    for k in range(800):
        t = k * dt
        if k == 400:
            node.request_switch(np.array([0.0, 6.0, 0.0]))
        out = node.update({"state": fake_state(START, np.zeros(N), t)})
        if prev is not None:
            # 위치: 이전 위치 + 이전 속도*dt 로 예측한 값과의 차이 (2차항 수준이어야)
            pred = prev["q_des"] + prev["qd_des"] * dt
            max_pos_gap = max(max_pos_gap, np.abs(out["q_des"] - pred).max())
            max_vel_jump = max(max_vel_jump, np.abs(out["qd_des"] - prev["qd_des"]).max())
        prev = out

    assert max_pos_gap < 5e-5, f"위치 불연속 {max_pos_gap:.2e}"
    assert max_vel_jump < 0.05, f"속도 점프 {max_vel_jump:.2e} rad/s (가속도*dt 수준이어야)"
    assert any(d.chosen for d in node.decisions)


def test_cooldown_blocks_chattering():
    # 쿨다운 로직만 검증한다 — 시간이 지나 참조가 전진하면 dream veto가
    # 후보를 정당하게 탈락시키므로, 여기서는 veto를 끈다
    node, _ = make_node(cooldown=0.5, with_dream=False)
    node.set_active(G.min_jerk("direct", START, GOAL, 1.2), t_now=0.0)

    node.request_switch(np.array([0.0, 6.0, 0.0]))
    node.update({"state": fake_state(START, np.zeros(N), 0.2)})
    assert node.decisions[-1].chosen is not None

    # 쿨다운 안: 기각 + 사유 기록
    node.request_switch(np.array([0.0, -6.0, 0.0]))
    node.update({"state": fake_state(START, np.zeros(N), 0.3)})
    assert node.decisions[-1].chosen is None
    assert "cooldown" in node.decisions[-1].reason

    # 쿨다운 지나면 다시 허용
    node.request_switch(np.array([0.0, -6.0, 0.0]))
    node.update({"state": fake_state(START, np.zeros(N), 0.9)})
    assert node.decisions[-1].chosen is not None


def test_reswitch_during_blend_keeps_reference_continuous():
    node, _ = make_node(cooldown=0.05)
    node.set_active(G.min_jerk("direct", START, GOAL, 1.2), t_now=0.0)

    dt = 1e-3
    prev = None
    max_vel_jump = 0.0
    for k in range(900):
        t = k * dt
        if k == 300:
            node.request_switch(np.array([0.0, 6.0, 0.0]))   # 1차 스위칭
        if k == 380:                                          # 블렌딩 도중 (min 0.15s)
            node.request_switch(np.array([6.0, 0.0, 0.0]))   # 2차 스위칭
        out = node.update({"state": fake_state(START, np.zeros(N), t)})
        if prev is not None:
            max_vel_jump = max(max_vel_jump, np.abs(out["qd_des"] - prev["qd_des"]).max())
        prev = out

    switches = [d for d in node.decisions if d.chosen is not None]
    assert len(switches) == 2, node.dump_decisions()
    assert max_vel_jump < 0.08, f"재스위칭에서 속도 점프 {max_vel_jump:.2e}"


def test_all_vetoed_keeps_current_plan():
    hal = MockRobotHAL(n_joints=N, dt=1e-3, torque_limit=20.0)
    dream = DreamModel.from_mock_hal(hal, kp=60.0, kd=2.5)
    far = [G.min_jerk(f"far{i}", np.full(N, 5.0 + i), np.full(N, 6.0 + i), 1.0)
           for i in range(3)]
    node = ChunkSwitchNode(scorer=ChunkScorer(far, dream=dream), goal=GOAL)
    active = G.min_jerk("direct", START, GOAL, 1.2)
    node.set_active(active, t_now=0.0)

    node.request_switch(np.array([0.0, 6.0, 0.0]))
    node.update({"state": fake_state(START, np.zeros(N), 0.3)})

    assert node.decisions[-1].chosen is None
    assert "vetoed" in node.decisions[-1].reason
    assert node._plan.chunk is active  # 기존 계획 유지


def test_request_before_set_active_is_harmless():
    node, _ = make_node()
    node.request_switch(np.array([1.0, 0.0, 0.0]))
    out = node.update({"state": fake_state(START, np.zeros(N), 0.0)})
    assert np.allclose(out["q_des"], START)


# ------------------------------------------------- MockRobotHAL 통합
def test_full_pipeline_switches_and_reaches_goal():
    """실행 중 외란 → FailureDetector 트리거 → 방향 추정 → 스위칭 → 목표 도달."""
    hal = MockRobotHAL(n_joints=N, dt=1e-3, torque_limit=20.0)
    logger = RingLogger.for_hal(hal, window_sec=3.0)
    node, _ = make_node(hal)
    node.set_active(G.min_jerk("direct", START, GOAL, 1.2), t_now=0.0)

    mgr = NodeGraphManager()
    mgr.add_node(node)
    mgr.add_node(ImpedanceNode(params={"kp": 60.0, "kd": 2.5}))
    mgr.connect("chunk_switch", "impedance")

    # 충돌 과도구간은 짧다: kd 댐핑이 ms 단위로 반응해 토크 신호를 ~15ms 안에
    # 상쇄한다. 그래서 감지 창(min_duration)을 10ms로 짧게, 폴링도 10ms 주기로.
    detector = FailureDetector(N, DetectorConfig(
        torque_threshold=4.0, torque_min_duration_s=0.01, refractory_s=0.5))
    runner = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger)
    runner.set_policy(lambda st: mgr.step({"state": st})["impedance"]["command"])

    runner.run(n_steps=400)                        # 이동 시작
    hal.inject_disturbance(1, +10.0, duration=5.0)  # 경로 중간에 측면 충돌 수준의 힘

    # 실제 모니터처럼: 폴링하다가 첫 감지 즉시 방향 추정 → 스위칭 요청.
    # (요청이 늦으면 참조가 전진해서 처음부터 시작하는 후보들이 전부
    #  '진입 토크 한계 초과'로 veto되는 게 맞다 — 그래서 즉시성이 중요하다)
    events = []
    for _ in range(10):
        runner.run(n_steps=10)
        new = detector.check(logger.to_arrays(window_sec=1.0))
        if new and not events:
            d = estimate_disturbance(logger.to_arrays(), window_s=0.05, baseline_s=0.3)
            assert d[1] > 2.0
            node.request_switch(d)
        events += new
    assert any(e.type == FailureType.TORQUE_SPIKE and e.joint_idx == 1 for e in events)

    runner.run(n_steps=2500)                      # 스위칭 + 새 궤적 완주
    switches = [dec for dec in node.decisions if dec.chosen is not None]
    assert len(switches) == 1
    # +방향으로 미는 외란: 순응하는 detour_up이 선택돼야 한다
    # (맞서는 detour_down은 저항 비용, 중립인 direct는 순응보다 비싸다)
    assert switches[0].chosen == "detour_up", node.dump_decisions()

    # 목표 도달. 관절 1은 지속 외란(+10Nm)에 의해 예측 가능한 처짐(d/kp)이 남는다
    q_final = hal.read_state().q
    assert abs(q_final[0] - GOAL[0]) < 0.06 and abs(q_final[2] - GOAL[2]) < 0.06, \
        f"목표 미도달: {q_final}"
    assert q_final[1] == pytest.approx(10.0 / 60.0, abs=0.06)
