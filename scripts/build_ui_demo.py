"""UI 공유 데모 빌더 — GitHub Pages용 정적 페이지 생성.

실제 ui/static/index.html·op.html을 **바이트 그대로** 읽어, 네트워크 폴링만
녹화 재생으로 패치해 한 장짜리 정적 페이지로 만든다. 녹화는 목 로봇으로
결정적 시나리오를 돌려 그 자리에서 뜬다 — 팀원이 보는 픽셀 = 실물 UI.

    python scripts/build_ui_demo.py        # → docs/demo/index.html

배포: docs/를 GitHub Pages 소스로 지정하면
    https://talkingtomoon.github.io/roboticus/demo/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot_core.integration.scenarios import TICK_STEPS, build_world  # noqa: E402
from robot_core.intent import IntentInterpreter, TypedSource  # noqa: E402
from robot_core.ui.guidance import compute_guidance  # noqa: E402
from robot_core.ui.server import build_inputs_block  # noqa: E402

OUT = ROOT / "docs" / "demo" / "index.html"


# ====================================================================== 녹화
def record_session() -> dict:
    """목 로봇으로 결정적 시나리오를 돌려 UI API 응답을 프레임 단위로 캡처."""
    hal, cache, sup, plan, guard = build_world(
        plan_tags=("approach", "insert", "finish"))
    source = TypedSource()
    interp = IntentInterpreter(sup, source=source, client=None)

    frames: list[dict] = []
    notes: dict[int, str] = {}

    def snap(note=None):
        ev = [{"seq": i + 1, "t": round(e.t, 3), "text": e.text}
              for i, e in enumerate(cache.events())]
        s = sup.snapshot()
        # 서버의 /api/state와 동일하게 입력 블록을 붙인다 (UI 입력 패널이 쓴다)
        s["inputs"] = build_inputs_block(s, interp)
        frames.append({"state": s, "events": ev,
                       "guidance": compute_guidance(s, ev)})
        if note:
            notes[len(frames) - 1] = note

    def step(n=1, note=None):
        for i in range(n):
            sup.tick()
            hal.step(TICK_STEPS)
            snap(note if i == 0 else None)

    # ── 각본: 리허설 시나리오의 축약판 (전 상태를 한 번씩 지나간다) ──
    step(2, "임무 시작 — 계획의 첫 태그로 모션을 고르고 재생합니다.")
    step(1, "재생 중. 모식도의 '재생' 노드가 켜집니다.")

    hal.inject_disturbance(0, 6.0, duration=0.3)
    step(1, "로봇을 옆에서 밀었습니다 (dob 스파이크 주입).")
    step(2, "감지 → 회복 판단 → 가드 통과 → '순응하는' 우회 모션 선곡.")
    step(2, "우회 후 계획으로 복귀합니다.")

    source.put("천천히 다시 해봐")
    interp.process_pending()
    step(1, "사람이 말을 걸었습니다 — 'slow'로 해석돼 가드를 통과.")

    source.put("저지 동작 해줘")
    interp.process_pending()
    step(1, "STT 오인식 흉내('저지'). 가드가 미지 태그를 거부합니다 — 빨강.")

    hal.set_rejection(12)          # 재생 중 무장 — 다음 경계의 선곡이 거절
    step(4, "로봇이 영점(코드 12)을 요구 — 사람 개입 대기로 전환.")
    assert sup.state.value == "WAITING_OPERATOR", sup.state
    step(2, "대기 중 — 자동 재시도는 하지 않습니다.")
    hal.clear_rejection()
    sup.operator_cleared()
    snap("담당자가 절차를 마치고 [해결했어요]를 눌렀습니다.")
    step(8, "남은 계획을 완주합니다.")

    sup.halt("운용자 페이지 멈춰 버튼")
    step(2, "정지 버튼을 눌렀습니다 — 모식도가 흐려지고 안내가 바뀝니다.")

    states = {f["state"]["state"] for f in frames}
    required = {"PLAYING", "WAITING_OPERATOR", "DONE", "HALTED"}
    assert required <= states, f"각본이 상태를 못 지나감: {required - states}"
    return {"frames": frames, "notes": notes}


# ================================================================ 페이지 조립
DASH_SHIM = """
// ==== 리뷰 데모 셈: 폴링 대신 부모가 프레임을 밀어넣는다 ====
for (const id of ["btnHalt", "btnResume", "btnSend", "txt"]) {
  const el = document.getElementById(id);
  el.disabled = true;
  el.title = "리뷰 데모 — 조작은 실제 서버에서만 됩니다";
  el.style.opacity = 0.4; el.style.cursor = "not-allowed";
}
window.renderFrame = function (frame, animate) {
  renderState(frame.state);
  const seen = new Set(allEvents.map(e => e.seq));
  const evs = frame.events.map(e => ({...e, info: classify(e.text)}));
  if (animate) {
    for (const e of evs)
      if (!seen.has(e.seq) && e.info.node)
        flash(e.info.node, e.info.blocked, e.info.label);
  }
  allEvents = evs;
  renderTimeline();
  renderScore();
};
"""

OP_SHIM = """
// ==== 리뷰 데모 셈 ====
for (const b of document.querySelectorAll("button, #txt")) {
  if (b.id !== "btnResolve") { b.disabled = true; }
  b.title = "리뷰 데모 — 조작은 실제 서버에서만 됩니다";
  b.style.opacity = 0.55; b.style.cursor = "not-allowed";
  b.onclick = null;
}
document.getElementById("txt").disabled = true;
window.renderFrame = function (frame) {
  const g = frame.guidance;
  const h = document.getElementById("headline");
  h.textContent = g.headline;
  h.className = g.severity;
  document.getElementById("detail").textContent = g.detail || "";
  const steps = document.getElementById("steps");
  const list = document.getElementById("stepList");
  if (g.steps && g.steps.length) {
    list.innerHTML = g.steps.map(s => `<li>${esc(s)}</li>`).join("");
    steps.classList.add("show");
  } else { steps.classList.remove("show"); }
  const rb = document.getElementById("btnResolve");
  rb.disabled = true;                               // 리뷰 모드: 클릭 불가
  rb.style.opacity = g.show_resolve ? 1.0 : 0.35;   // 활성 '표시'만 재현
};
"""

POLL_NEEDLE = "poll();\nsetInterval(poll, 1000);"


def _embed(html: str) -> str:
    """<script> 안 템플릿 리터럴로 안전하게 넣기 위한 이스케이프."""
    return (html.replace("\\", "\\\\").replace("`", "\\`")
                .replace("</script>", "<\\/script>").replace("${", "\\${"))


PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>robot_core 운용 UI — 리뷰 데모</title>
<style>
  :root {
    --bg:#eef1f4; --panel:#ffffff; --line:#d5dbe1; --fg:#1c2430;
    --dim:#5c6b7a; --accent:#2f6feb; --accent-fg:#ffffff; --note-bg:#e6ecf5;
  }
  @media (prefers-color-scheme: dark) { :root {
    --bg:#0a0d10; --panel:#12161b; --line:#242b33; --fg:#e6e9ec;
    --dim:#8a949e; --accent:#58a6ff; --accent-fg:#0a0d10; --note-bg:#16202e;
  } }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--fg);
         font:15px/1.55 system-ui, -apple-system, "Malgun Gothic", sans-serif; }
  .wrap { max-width:1060px; margin:0 auto; padding:20px 16px 40px; }
  header h1 { font-size:21px; font-weight:800; text-wrap:balance; }
  header .sub { color:var(--dim); font-size:13px; margin-top:4px; }
  header .sub code { font:12px ui-monospace, Consolas, monospace;
                     background:var(--note-bg); padding:1px 6px; border-radius:5px; }
  .player { background:var(--panel); border:1px solid var(--line);
            border-radius:12px; padding:14px 16px; margin:16px 0 12px;
            position:sticky; top:8px; z-index:5;
            box-shadow:0 4px 14px rgba(0,0,0,0.08); }
  .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .controls button { min-width:44px; min-height:40px; font-size:16px;
                     border:1px solid var(--line); border-radius:9px;
                     background:var(--panel); color:var(--fg); cursor:pointer; }
  .controls button:focus-visible { outline:2px solid var(--accent); }
  #btnPlay { background:var(--accent); color:var(--accent-fg); border:0;
             min-width:88px; font-weight:700; }
  #scrub { flex:1; min-width:160px; accent-color:var(--accent); }
  #pos { font:13px ui-monospace, Consolas, monospace; color:var(--dim);
         min-width:56px; text-align:right; font-variant-numeric:tabular-nums; }
  #note { margin-top:10px; padding:9px 12px; border-radius:8px;
          background:var(--note-bg); font-size:14px; min-height:38px; }
  #note b { color:var(--accent); }
  .tabs { display:flex; gap:8px; margin-bottom:10px; }
  .tabs button { padding:9px 16px; font-size:14px; font-weight:700;
                 border:1px solid var(--line); border-radius:9px 9px 0 0;
                 border-bottom:0; background:transparent; color:var(--dim);
                 cursor:pointer; }
  .tabs button.on { background:var(--panel); color:var(--fg);
                    border-color:var(--accent); }
  .pane { display:none; background:var(--panel); border:1px solid var(--line);
          border-radius:0 12px 12px 12px; padding:10px; }
  .pane.on { display:block; }
  .pane iframe { width:100%; border:0; border-radius:8px; background:#111418;
                 height:min(74vh, 860px); }
  #paneOp iframe { max-width:410px; display:block; margin:0 auto;
                   box-shadow:0 0 0 10px #1a2027, 0 0 0 12px var(--line);
                   border-radius:22px; }
  .footnote { color:var(--dim); font-size:12.5px; margin-top:14px;
              line-height:1.6; }
  @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>robot_core 운용 UI — 공유 리뷰 데모</h1>
  <div class="sub">실제 <code>ui/static/index.html · op.html</code> 코드 그대로,
  목 로봇 세션 녹화를 재생합니다
  · <a href="https://github.com/talkingtomoon/roboticus">저장소</a>
  · 실배포는 젯슨에서 <code>python -m robot_core.ui.server</code></div>
</header>

<div class="player">
  <div class="controls">
    <button id="btnFirst" aria-label="처음으로">⏮</button>
    <button id="btnPrev" aria-label="이전">◀</button>
    <button id="btnPlay">▶ 재생</button>
    <button id="btnNext" aria-label="다음">▶︎▶︎</button>
    <input id="scrub" type="range" min="0" max="0" value="0" step="1">
    <span id="pos">0/0</span>
  </div>
  <div id="note">시나리오: 임무 수행 중 충격 → 우회, 사람 발화("천천히"),
  STT 오인식 거부, 영점 요구(코드 12) → 사람 대기 → 완주 → 정지.</div>
</div>

<div class="tabs" role="tablist">
  <button id="tabDash" class="on" role="tab">운용 대시보드 (/)</button>
  <button id="tabOp" role="tab">운용자 페이지 (/op) — 팀원용</button>
</div>
<div class="pane on" id="paneDash"><iframe id="ifDash" title="대시보드"></iframe></div>
<div class="pane" id="paneOp"><iframe id="ifOp" title="운용자 페이지"></iframe></div>

<div class="footnote">
버튼·입력은 리뷰 모드라 비활성입니다 (실서버에서만 동작).
대시보드의 모식도 노드는 클릭하면 타임라인이 그 종류로 필터됩니다 —
실물과 동일하게 동작합니다. ◀/▶ 키보드로도 넘길 수 있어요.
이 페이지는 <code>python scripts/build_ui_demo.py</code>로 재생성됩니다.
</div>
</div>

<script>
"use strict";
const FRAMES = __FRAMES__;
const NOTES = __NOTES__;
const DASH_SRC = `__DASH__`;
const OP_SRC = `__OP__`;

const ifDash = document.getElementById("ifDash");
const ifOp = document.getElementById("ifOp");
ifDash.srcdoc = DASH_SRC;
ifOp.srcdoc = OP_SRC;

let i = 0, playing = null, ready = 0;
const scrub = document.getElementById("scrub");
scrub.max = FRAMES.length - 1;

function show(idx, animate) {
  i = Math.max(0, Math.min(FRAMES.length - 1, idx));
  scrub.value = i;
  document.getElementById("pos").textContent = (i + 1) + "/" + FRAMES.length;
  const f = FRAMES[i];
  try { ifDash.contentWindow.renderFrame(f, animate); } catch (e) {}
  try { ifOp.contentWindow.renderFrame(f); } catch (e) {}
  let note = null;
  for (let k = i; k >= 0; k--) if (NOTES[k]) { note = NOTES[k]; break; }
  document.getElementById("note").innerHTML =
    "<b>" + f.state.state + "</b> — " + (note || "");
}

function onReady() { ready++; if (ready >= 2) show(0, false); }
ifDash.addEventListener("load", onReady);
ifOp.addEventListener("load", onReady);

function stop() {
  if (playing) { clearInterval(playing); playing = null;
    document.getElementById("btnPlay").textContent = "▶ 재생"; }
}
document.getElementById("btnPlay").onclick = () => {
  if (playing) { stop(); return; }
  if (i >= FRAMES.length - 1) show(0, false);
  document.getElementById("btnPlay").textContent = "⏸ 일시정지";
  playing = setInterval(() => {
    if (i >= FRAMES.length - 1) { stop(); return; }
    show(i + 1, true);
  }, 1300);
};
document.getElementById("btnFirst").onclick = () => { stop(); show(0, false); };
document.getElementById("btnPrev").onclick = () => { stop(); show(i - 1, false); };
document.getElementById("btnNext").onclick = () => { stop(); show(i + 1, true); };
scrub.oninput = () => { stop(); show(+scrub.value, false); };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") { stop(); show(i + 1, true); }
  if (e.key === "ArrowLeft") { stop(); show(i - 1, false); }
  if (e.key === " " && e.target === document.body) {
    e.preventDefault(); document.getElementById("btnPlay").click(); }
});

const panes = {tabDash: "paneDash", tabOp: "paneOp"};
for (const [tid, pid] of Object.entries(panes)) {
  document.getElementById(tid).onclick = () => {
    for (const [t2, p2] of Object.entries(panes)) {
      document.getElementById(t2).classList.toggle("on", t2 === tid);
      document.getElementById(p2).classList.toggle("on", p2 === pid);
    }
    show(i, false);
  };
}
</script>
</body>
</html>
"""


def build() -> Path:
    session = record_session()
    dash = (ROOT / "robot_core/ui/static/index.html").read_text(encoding="utf-8")
    op = (ROOT / "robot_core/ui/static/op.html").read_text(encoding="utf-8")
    assert POLL_NEEDLE in dash and POLL_NEEDLE in op, \
        "UI의 폴링 코드가 바뀌었다 — POLL_NEEDLE을 갱신할 것"
    dash = dash.replace(POLL_NEEDLE, DASH_SHIM)
    op = op.replace(POLL_NEEDLE, OP_SHIM)

    page = (PAGE
            .replace("__FRAMES__", json.dumps(session["frames"],
                                              ensure_ascii=False,
                                              separators=(",", ":")))
            .replace("__NOTES__", json.dumps(session["notes"],
                                             ensure_ascii=False))
            .replace("__DASH__", _embed(dash))
            .replace("__OP__", _embed(op)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8", newline="\n")
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"written {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"frames 검증 포함)")
