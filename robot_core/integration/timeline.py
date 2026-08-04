"""통합 감사 로그 — 회복/스위칭/보정/감지 이벤트를 단일 시간축으로 병합.

전제: FullStack.build()가 SafetyGuard와 LLMRecoveryAgent의 time_fn을
로봇 클록(hal.t)으로 잡아 두므로, 세 소스의 타임스탬프가 같은 축이다.
(이걸 안 하면 벽시계·로봇시계가 섞여 사후 디버깅이 불가능해진다 — 문제 (D))
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimelineEntry:
    t: float
    source: str   # detect | interlock | switch | recovery | guard | event
    text: str

    def line(self) -> str:
        return f"[t={self.t:8.3f}] {self.source:9s} | {self.text}"


def build_timeline(logger=None, switch_node=None, guard=None,
                   extra: list[TimelineEntry] | None = None) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = list(extra or [])

    if logger is not None:
        for ev in logger.events():
            text = ev.text
            if text.startswith("DETECTED"):
                src = "detect"
            elif text.startswith("interlock:"):
                src, text = "interlock", text[len("interlock:"):].strip()
            elif text.startswith("recovery"):
                src = "recovery"
            elif text.startswith("select:"):
                src, text = "select", text[len("select:"):].strip()
            elif text.startswith("play"):
                src = "play"
            elif text.startswith("operator"):
                src = "operator"
            else:
                src = "event"
            entries.append(TimelineEntry(t=ev.t, source=src, text=text))

    if switch_node is not None:
        for d in switch_node.decisions:
            entries.append(TimelineEntry(t=d.t, source="switch", text=d.line()))
            if d.report is not None and d.chosen is not None:
                top = d.report.table(top=3).splitlines()[1:]
                for row in top:
                    entries.append(TimelineEntry(t=d.t, source="switch",
                                                 text="  " + row.strip()))

    if guard is not None:
        for a in guard.audit:
            entries.append(TimelineEntry(t=a.wall_t, source="guard", text=a.line()))

    entries.sort(key=lambda e: (e.t, _source_rank(e.source)))
    return entries


def _source_rank(source: str) -> int:
    # 같은 시각 내 순서: 감지 → 회복 → 가드 → 나머지(캐시 삽입 순서 보존 —
    # 파이썬 정렬은 stable이라 rank가 같으면 원래 순서가 유지된다)
    order = {"detect": 0, "interlock": 1, "recovery": 2, "guard": 3,
             "select": 5, "switch": 5, "play": 5, "operator": 5, "event": 5}
    return order.get(source, 9)


def format_timeline(entries: list[TimelineEntry], title: str = "TIMELINE") -> str:
    lines = ["=" * 74, f"[{title}] {len(entries)} events (robot clock, 단일 시간축)",
             "=" * 74]
    lines += [e.line() for e in entries]
    lines.append("=" * 74)
    return "\n".join(lines)


def assert_monotonic(entries: list[TimelineEntry]) -> None:
    """타임라인 일관성 검사: 시간이 뒤로 가면 병합이 잘못된 것."""
    for a, b in zip(entries, entries[1:]):
        if b.t < a.t - 1e-9:
            raise AssertionError(f"timeline not monotonic: {a.t} -> {b.t}")
