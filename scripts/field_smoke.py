"""현장 호환성 스모크 — 캠프 1일차에 sim:demo 대상으로 파사드 가정을 검증.

우리 코드는 phorce 파사드에 대한 가정 위에 서 있다. 이 스크립트는 가정을
하나씩 실측해 PASS/FAIL/실측치를 한 장 리포트로 만든다. 깨지는 가정이 있으면
리포트에 적힌 코드 위치만 고치면 된다.

    python scripts/field_smoke.py --mock            # 목으로 리포트 형식 리허설
    python scripts/field_smoke.py                    # 현장: RealPhorceHAL 연결 후

검증 항목:
  A1 connect/status/motions 왕복
  A2 watch() 콜백 실측 주파수 + 호출 스레드 식별 (10초 측정)
  A3 play 완료 → 핸들 이벤트 지연 실측
  A4 재생 중 play 시도 → MotionBusy 발생 확인
  A5 12축 중 valid 축 인덱스
  A6 dob_a/current_a 30초 분포 → 감지 임계 자동 제안
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core.hal.phorce import MotionBusy, N_AXES  # noqa: E402


class Report:
    def __init__(self):
        self.rows = []

    def add(self, item, status, measured, assumption_loc):
        self.rows.append((item, status, measured, assumption_loc))
        print(f"  [{status}] {item}: {measured}")

    def render(self) -> str:
        lines = ["=" * 78, "PHORCE 현장 호환성 스모크 리포트", "=" * 78,
                 f"{'항목':30s} {'판정':6s} 실측치 / 가정 코드 위치", "-" * 78]
        for item, status, measured, loc in self.rows:
            lines.append(f"{item:30s} {status:6s} {measured}")
            lines.append(f"{'':30s}        가정: {loc}")
        n_fail = sum(1 for r in self.rows if r[1] == "FAIL")
        lines.append("-" * 78)
        lines.append(f"결과: {len(self.rows) - n_fail}/{len(self.rows)} PASS"
                     + ("" if n_fail == 0 else f"  ** FAIL {n_fail}건 — 위 코드 위치 수정"))
        lines.append("=" * 78)
        return "\n".join(lines)


def make_hal(mock: bool, sim_thread=None):
    if mock:
        from robot_core.hal.phorce import MockMotion, MockPhorceHAL
        home = np.zeros(N_AXES)
        pose = np.zeros(N_AXES); pose[:3] = [0.4, -0.2, 0.5]

        def traj(t):
            s = min(t / 1.0, 1.0)
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            return home + s * (pose - home)
        hal = MockPhorceHAL({1: MockMotion(1.0, traj),
                             2: MockMotion(0.8, lambda t: pose)})
        # 목에는 수신 스레드가 없으므로 하나 돌려서 실물 조건을 흉내낸다
        stop = threading.Event()

        def sim():
            while not stop.is_set():
                hal.step(1)
                time.sleep(0.001)
        th = threading.Thread(target=sim, daemon=True, name="mock-feedback")
        th.start()
        hal._smoke_stop = stop
        return hal
    # TODO(현장): phorce 파사드 래퍼
    #   import phorce; robot = phorce.connect()
    #   from robot_core.hal.phorce_real import RealPhorceHAL
    #   return RealPhorceHAL(robot)
    raise SystemExit("실물 HAL 미구현 — RealPhorceHAL 연결 후 사용. "
                     "리포트 형식 확인은 --mock")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--watch-sec", type=float, default=10.0)
    p.add_argument("--dist-sec", type=float, default=30.0)
    p.add_argument("--out", default="field_smoke_report.txt")
    args = p.parse_args()

    rep = Report()
    hal = make_hal(args.mock)

    # ---- A1: connect/status/motions 왕복 --------------------------------
    try:
        cat = hal.catalog()
        fb = hal.latest_feedback()
        ok = bool(cat) and fb is not None
        rep.add("A1 connect/motions/feedback", "PASS" if ok else "FAIL",
                f"적재 {len(cat)}개 slot={sorted(cat)}, feedback seq={getattr(fb, 'seq', None)}",
                "hal/phorce.py PhorceHAL.catalog()/latest_feedback()")
    except Exception as e:
        rep.add("A1 connect/motions/feedback", "FAIL", f"{type(e).__name__}: {e}",
                "hal/phorce.py PhorceHAL")
        print(rep.render())
        return

    # ---- A2: watch() 실측 주파수 + 스레드 --------------------------------
    stamps: list[float] = []
    threads: set[str] = set()

    def cb(frame):
        stamps.append(time.perf_counter())
        threads.add(threading.current_thread().name)

    hal.watch(cb)
    time.sleep(args.watch_sec if not args.mock else min(args.watch_sec, 3.0))
    if len(stamps) >= 10:
        periods = np.diff(stamps)
        hz = 1.0 / float(np.median(periods))
        status = "PASS" if hz > 200 else "WARN"
        rep.add("A2 watch() 주파수/스레드", status,
                f"{hz:.0f} Hz (median), 호출 스레드={sorted(threads)} — "
                f"메인 스레드가 아니면 콜백은 저장만 (규칙 확인)",
                "supervisor.py Supervisor.__init__ hal.watch(cache.push)")
    else:
        rep.add("A2 watch() 주파수/스레드", "FAIL", f"콜백 {len(stamps)}회뿐",
                "hal/phorce.py PhorceHAL.watch")

    # ---- A3: play 완료 → 핸들 이벤트 지연 --------------------------------
    first_id = sorted(cat)[0]
    dur = cat[first_id]
    t0 = time.perf_counter()
    h = hal.play_async(first_id)
    if args.mock:
        while not h.done():
            time.sleep(0.005)
    else:
        h.wait(timeout=dur + 5.0)
    latency = time.perf_counter() - t0 - dur
    ok = h.done()
    rep.add("A3 play 완료→핸들 지연", "PASS" if ok else "FAIL",
            f"공칭 {dur:.2f}s, 핸들 완료까지 {time.perf_counter() - t0:.2f}s "
            f"(지연 {latency * 1e3:+.0f} ms)",
            "supervisor.py _check_boundary (핸들 완료 = 경계 감지 1차 수단)")

    # ---- A4: 재생 중 play → MotionBusy ----------------------------------
    h2 = hal.play_async(first_id)
    try:
        hal.play_async(first_id)
        rep.add("A4 재생 중 play → BUSY", "FAIL", "예외 없이 통과 — 큐가 있나?",
                "hal/phorce.py MotionBusy + supervisor.py BUSY 원칙")
    except MotionBusy:
        rep.add("A4 재생 중 play → BUSY", "PASS", "MotionBusy 발생 (매뉴얼 일치)",
                "hal/phorce.py MotionBusy")
    except Exception as e:
        rep.add("A4 재생 중 play → BUSY", "FAIL", f"다른 예외: {type(e).__name__}: {e}",
                "hal/phorce.py 예외 매핑")
    if args.mock:
        while not h2.done():
            time.sleep(0.005)
    else:
        h2.wait(timeout=dur + 5.0)

    # ---- A5: valid 축 인덱스 --------------------------------------------
    fb = hal.latest_feedback()
    valid_idx = [int(i) for i in np.flatnonzero(fb.usable)]
    rep.add("A5 valid 축", "PASS" if valid_idx else "FAIL",
            f"{len(valid_idx)}/12 valid: {valid_idx}",
            "recovery/detector.py — valid=False 축은 판정 제외")

    # ---- A6: dob/current 분포 → 임계 제안 --------------------------------
    dobs, curs = [], []

    def dist_cb(frame):
        dobs.append(np.where(frame.usable, np.abs(frame.dob_a), 0.0).max())
        curs.append(np.where(frame.usable, np.abs(frame.current_a), 0.0).max())

    hal.watch(dist_cb)
    time.sleep(args.dist_sec if not args.mock else min(args.dist_sec, 3.0))
    if dobs:
        dob_p99 = float(np.percentile(dobs, 99))
        cur_p99 = float(np.percentile(curs, 99))
        rep.add("A6 dob/current 분포", "PASS",
                f"|dob| p50={statistics.median(dobs):.2f} p99={dob_p99:.2f} A, "
                f"|cur| p99={cur_p99:.2f} A → 제안: impact_dob_threshold="
                f"{max(dob_p99 * 3, 1.0):.1f}, stall_current_floor={max(cur_p99 * 0.5, 0.5):.1f}",
                "recovery/detector.py DetectorConfig")
    else:
        rep.add("A6 dob/current 분포", "FAIL", "프레임 없음",
                "hal/phorce.py watch")

    if hasattr(hal, "_smoke_stop"):
        hal._smoke_stop.set()

    print()
    report_text = rep.render()
    print(report_text)
    Path(args.out).write_text(report_text, encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
