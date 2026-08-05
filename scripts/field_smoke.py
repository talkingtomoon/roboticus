"""현장 스모크 — 실물 전용, 안전 우선. 시뮬레이터는 없다 (문서 확정).

기본 실행(A1~A5)은 전부 **읽기**다 — 로봇이 움직이지 않는다.
A6(쓰기: 모션 1개 재생)은 --allow-motion 플래그와 대화형 확인이 **둘 다**
있어야 실행된다.

    python scripts/field_smoke.py                     # A1~A5 읽기 전용
    python scripts/field_smoke.py --allow-motion 4    # + A6 (로봇이 움직인다!)

검증 순서 (고정):
  A1 [읽기] phorce.connect() 성공 (PhorceUnavailable 처리)
  A2 [읽기] robot.doctor() → ok/issues
  A3 [읽기] robot.status() → state_name, 2초 간격 3회
  A4 [읽기] robot.motions() → 적재 id 목록 (카탈로그 정본)
  A5 [읽기] 피드백 30초 수집 → 실측 Hz, valid 축 인덱스,
            dob_a/current_a 분포 → 감지 임계 자동 제안, temp_c 기준선
  A6 [쓰기] ★사람 확인 후에만★ 지정 안전 모션 1개 재생 →
            핸들 완료 지연 실측, result.ok/detail 확인

각 항목은 PASS/FAIL/실측치와 함께, 실패 시 그 가정이 박혀 있는 코드
위치를 리포트에 명시한다 — 깨진 가정만 그 자리에서 고치면 된다.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core.hal.phorce import (  # noqa: E402
    MotionAborted, PhorceError,
)
from robot_core.hal.real_phorce import RealPhorceHAL  # noqa: E402


class Report:
    def __init__(self):
        self.rows = []

    def add(self, item, status, measured, assumption_loc):
        self.rows.append((item, status, measured, assumption_loc))
        print(f"  [{status}] {item}: {measured}")

    def render(self) -> str:
        lines = ["=" * 78, "PHORCE 현장 스모크 리포트 (실물)", "=" * 78,
                 f"{'항목':30s} {'판정':6s} 실측치 / 가정 코드 위치", "-" * 78]
        for item, status, measured, loc in self.rows:
            lines.append(f"{item:30s} {status:6s} {measured}")
            lines.append(f"{'':30s}        가정: {loc}")
        n_fail = sum(1 for r in self.rows if r[1] == "FAIL")
        lines.append("-" * 78)
        lines.append(f"결과: {len(self.rows) - n_fail}/{len(self.rows)} PASS"
                     + ("" if n_fail == 0 else
                        f"  ** FAIL {n_fail}건 — 위 코드 위치 수정"))
        lines.append("=" * 78)
        return "\n".join(lines)


def stage_a1_connect(rep: Report) -> RealPhorceHAL | None:
    try:
        hal = RealPhorceHAL()
        rep.add("A1 phorce.connect()", "PASS", "연결됨",
                "hal/real_phorce.py _connect_facade()")
        return hal
    except PhorceError as e:
        rep.add("A1 phorce.connect()", "FAIL", str(e).splitlines()[0],
                "hal/real_phorce.py _connect_facade() — 시동 절차(README) 확인")
        return None


def stage_a2_doctor(rep: Report, hal: RealPhorceHAL) -> None:
    try:
        d = hal.doctor()
        status = "PASS" if d["ok"] else "FAIL"
        issues = "; ".join(d["issues"]) if d["issues"] else "(이슈 없음)"
        rep.add("A2 robot.doctor()", status, f"ok={d['ok']} {issues}",
                "hal/real_phorce.py RealPhorceHAL.doctor() / supervisor.preflight")
    except Exception as e:
        rep.add("A2 robot.doctor()", "FAIL", f"{type(e).__name__}: {e}",
                "hal/real_phorce.py RealPhorceHAL.doctor()")


def stage_a3_status(rep: Report, hal: RealPhorceHAL) -> None:
    names = []
    for i in range(3):
        st = hal.status()
        names.append(st.get("state_name", st.get("error", "?")))
        if i < 2:
            time.sleep(2.0)
    ok = all("error" not in str(n).lower() for n in names)
    rep.add("A3 robot.status() x3 (2s 간격)", "PASS" if ok else "FAIL",
            f"state_name={names} — 'IDLE'만 유휴의 증거 "
            f"(STALE/CONTRACT_INACTIVE/UNKNOWN은 '모름')",
            "hal/real_phorce.py busy_state() / supervisor._check_boundary 폴백")


def stage_a4_motions(rep: Report, hal: RealPhorceHAL) -> dict[int, str]:
    try:
        names = hal.motion_names()
        ok = bool(names)
        listing = ", ".join(f"{i}:{n}" for i, n in sorted(names.items()))
        rep.add("A4 robot.motions()", "PASS" if ok else "FAIL",
                f"적재 {len(names)}개 [{listing}]" if ok else
                "적재 모션 0개 — SD 카드/교시 확인",
                "hal/real_phorce.py catalog() / catalog.reconcile (id만 대조)")
        return names
    except Exception as e:
        rep.add("A4 robot.motions()", "FAIL", f"{type(e).__name__}: {e}",
                "hal/real_phorce.py catalog()")
        return {}


def stage_a5_feedback(rep: Report, hal: RealPhorceHAL, seconds: float) -> None:
    try:
        from robot_core.adapters.phorce_ros2 import PhorceFeedbackBridge
        bridge = PhorceFeedbackBridge()
    except Exception as e:
        rep.add("A5 피드백 수집", "FAIL", f"브리지 생성 실패: {e}",
                "adapters/phorce_ros2.py PhorceFeedbackBridge")
        return

    frames_t, dobs, curs = [], [], []
    last = {"frame": None}

    def collect(frame):
        frames_t.append(frame.t)
        u = frame.usable
        dobs.append(float(np.where(u, np.abs(frame.dob_a), 0.0).max()))
        curs.append(float(np.where(u, np.abs(frame.current_a), 0.0).max()))
        last["frame"] = frame

    try:
        hal.watch(collect)
        hal.attach_feedback_bridge(bridge)
    except RuntimeError as e:
        rep.add("A5 피드백 수집", "FAIL", str(e),
                "adapters/phorce_ros2.py start() — rclpy/agx_msgs source 확인")
        return
    print(f"  ... 피드백 {seconds:.0f}초 수집 중 (로봇은 움직이지 않음)")
    time.sleep(seconds)
    bridge.stop()

    if len(frames_t) < 100:
        rep.add("A5 피드백 수집", "FAIL",
                f"{len(frames_t)}프레임뿐 ({seconds:.0f}s) — "
                "qos_profile_sensor_data 미적용이면 에러 없이 0건이 된다",
                "adapters/phorce_ros2.py start() QoS")
        return

    hz = 1.0 / float(np.median(np.diff(frames_t)))
    fb = last["frame"]
    valid_idx = [int(i) for i in np.flatnonzero(fb.usable)]
    temps = ", ".join(f"ax{i}:{fb.temp_c[i]:.1f}" for i in valid_idx)
    dob_p99 = float(np.percentile(dobs, 99))
    cur_p99 = float(np.percentile(curs, 99))
    rep.add("A5 피드백 실측", "PASS" if hz > 500 else "WARN",
            f"{hz:.0f} Hz (median), {len(frames_t)}프레임, "
            f"valid {len(valid_idx)}/12: {valid_idx}",
            "adapters/phorce_ros2.py msg_to_frame (recv_monotonic_ns 시간축)")
    rep.add("A5 감지 임계 제안", "PASS",
            f"|dob| p50={statistics.median(dobs):.2f} p99={dob_p99:.2f} A, "
            f"|cur| p99={cur_p99:.2f} A → impact_dob_threshold="
            f"{max(dob_p99 * 3, 1.0):.1f}, "
            f"stall_current_floor={max(cur_p99 * 0.5, 0.5):.1f}",
            "recovery/detector.py DetectorConfig")
    rep.add("A5 온도 기준선", "PASS", temps or "(valid 축 없음)",
            "recovery/detector.py 과열 임계 대비 기준선")


def stage_a6_play(rep: Report, hal: RealPhorceHAL, motion_id: int,
                  names: dict[int, str]) -> None:
    """유일한 쓰기 단계. --allow-motion + 대화형 확인 이중 게이트."""
    name = names.get(motion_id, "?")
    print(f"\n  ** A6는 로봇을 실제로 움직입니다: motion {motion_id} ({name})")
    print("  ** 로봇 주변에 사람과 물건이 없는지 확인하세요.")
    try:
        answer = input("  계속하려면 'yes' 입력 (그 외 입력/Ctrl+C = 건너뜀): ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip().lower() != "yes":
        rep.add("A6 재생 (쓰기)", "SKIP", "사람 확인 없음 — 재생 안 함",
                "scripts/field_smoke.py stage_a6_play (이중 게이트)")
        return

    t0 = time.perf_counter()
    try:
        handle = hal.play_async(motion_id)
    except PhorceError as e:
        rep.add("A6 재생 (쓰기)", "FAIL", f"{type(e).__name__}: {e}",
                "hal/real_phorce.py play_async — 거절코드 분류표 "
                "(hal/phorce.py classify_reject_code)")
        return
    handle.wait(timeout=120.0)
    elapsed = time.perf_counter() - t0
    if not handle.done():
        rep.add("A6 재생 (쓰기)", "FAIL", f"{elapsed:.1f}s 지나도 핸들 미완료",
                "hal/real_phorce.py _await_result (핸들 완료 = 경계 1차 수단)")
        return
    try:
        handle.result()
        rep.add("A6 재생 (쓰기)", "PASS",
                f"완료까지 {elapsed:.2f}s, result ok",
                "supervisor._check_boundary (핸들 완료 = 경계 감지 1차 수단)")
    except MotionAborted as e:
        # 알려진 현상: 버튼 복구 직후 첫 재생이 error=17로 중단될 수 있음 —
        # 한 번 더 보내면 된다 (guidance.py에도 번역돼 있다)
        rep.add("A6 재생 (쓰기)", "FAIL",
                f"ABORTED after {elapsed:.2f}s: {e.reason} "
                f"(error=17이면 한 번 더 보내면 된다 — 알려진 현상)",
                "hal/real_phorce.py _await_result / ui/guidance.py error=17")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feedback-sec", type=float, default=30.0,
                   help="A5 피드백 수집 시간 (기본 30초)")
    p.add_argument("--allow-motion", type=int, default=None, metavar="ID",
                   help="A6 활성화: 이 id의 안전 모션 1개를 재생한다 "
                        "(대화형 확인도 통과해야 실행)")
    p.add_argument("--out", default="field_smoke_report.txt")
    args = p.parse_args()

    rep = Report()
    hal = stage_a1_connect(rep)
    if hal is None:
        print("\n" + rep.render())
        sys.exit(1)

    stage_a2_doctor(rep, hal)
    stage_a3_status(rep, hal)
    names = stage_a4_motions(rep, hal)
    stage_a5_feedback(rep, hal, args.feedback_sec)

    if args.allow_motion is not None:
        if args.allow_motion in names:
            stage_a6_play(rep, hal, args.allow_motion, names)
        else:
            rep.add("A6 재생 (쓰기)", "FAIL",
                    f"--allow-motion {args.allow_motion}은 적재 목록에 없다 "
                    f"(적재: {sorted(names)})",
                    "scripts/field_smoke.py — A4 적재 목록이 정본")
    else:
        print("\n  (A6 건너뜀 — 재생하려면 --allow-motion <id>. 기본은 읽기 전용)")

    print()
    report_text = rep.render()
    print(report_text)
    Path(args.out).write_text(report_text, encoding="utf-8")
    print(f"\n저장: {args.out}")
    if any(r[1] == "FAIL" for r in rep.rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
