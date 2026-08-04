"""SafetyGuard — 위험한 결정의 유일한 관문.

phorce 피벗 이후의 현행 가드는 **TagSafetyGuard** (파일 하단):
LLM은 의도 태그만 내고, 태그는 카탈로그에 존재하는 것만 통과한다.
motion_id 최종 결정은 선택기가 한다 (실행 경로 분리 원칙).

아래 SafetyGuard/ParamSpec은 임피던스 시절의 파라미터 가드 —
robot_core/legacy 참고용 코드가 쓴다.
"""

from __future__ import annotations

# [legacy] SafetyGuard(파라미터 관문) 원칙 — 참고용 코드가 쓰는 구 가드:
# - 화이트리스트에 없는 (node, param) → 거부
# - 범위 밖 값 → [min, max] 클램프 + 경고 (부분적으로라도 회복하게)
# - 변화율 제한 (×max_rel_step / ±max_abs_step — kp 10배 튀기기 방지)
# - 숫자 아니거나 NaN/inf → 거부, 모든 시도를 감사 로그에

import math
import threading
import time
from dataclasses import dataclass, field

from robot_core.graph.manager import NodeGraphManager


@dataclass
class ParamSpec:
    """조정 가능한 파라미터 하나의 안전 범위."""

    min: float
    max: float
    max_rel_step: float | None = 2.0   # 1회 적용당 배율 상한 (현재값>0일 때). None=제한 없음
    max_abs_step: float | None = None  # 1회 적용당 절대 변화 상한. None=제한 없음

    def __post_init__(self) -> None:
        if not (self.min < self.max):
            raise ValueError(f"invalid range [{self.min}, {self.max}]")
        if self.max_rel_step is not None and self.max_rel_step <= 1.0:
            raise ValueError("max_rel_step must be > 1")
        if self.max_abs_step is not None and self.max_abs_step <= 0:
            raise ValueError("max_abs_step must be > 0")


@dataclass
class AuditEntry:
    wall_t: float          # time.monotonic() 기준
    source: str            # "llm" / "rules:<이유>" / "ros2-param" 등
    node: str
    param: str
    requested: object      # 원래 요청값 (숫자가 아닐 수도 있으니 object)
    applied: float | None  # 실제 적용값. 거부면 None
    status: str            # "applied" | "clamped" | "rate-limited" | "rejected"
    reason: str = ""

    def line(self) -> str:
        req = f"{self.requested:.4g}" if isinstance(self.requested, (int, float)) else repr(self.requested)
        app = "-" if self.applied is None else f"{self.applied:.4g}"
        return (
            f"[{self.wall_t:9.3f}] {self.status.upper():12s} {self.node}.{self.param}: "
            f"req={req} applied={app} src={self.source}"
            + (f" ({self.reason})" if self.reason else "")
        )


class SafetyGuard:
    def __init__(
        self,
        manager: NodeGraphManager,
        whitelist: dict[str, ParamSpec],
        time_fn=time.monotonic,
    ) -> None:
        """whitelist 키는 "node.param" 형식. 예: {"impedance.kp": ParamSpec(1, 120)}."""
        for key in whitelist:
            if key.count(".") != 1:
                raise ValueError(f'whitelist key {key!r} must be "node.param"')
        self.manager = manager
        self.whitelist = dict(whitelist)
        self._time = time_fn
        self._audit: list[AuditEntry] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 적용
    def apply(self, actions: list[dict], source: str) -> list[AuditEntry]:
        """액션 리스트를 검증·클램프해서 적용. 각 액션의 감사 엔트리를 돌려준다.

        액션 형식: {"node": str, "param": str, "value": number}
        한 액션의 거부가 다른 액션을 막지 않는다 (독립 처리).
        """
        entries = [self._apply_one(a, source) for a in actions]
        with self._lock:
            self._audit.extend(entries)
        return entries

    def _apply_one(self, action: dict, source: str) -> AuditEntry:
        now = self._time()

        def reject(node, param, requested, reason):
            return AuditEntry(now, source, node, param, requested, None, "rejected", reason)

        if not isinstance(action, dict):
            return reject("?", "?", action, "action is not a dict")
        node = action.get("node")
        param = action.get("param")
        value = action.get("value")
        if not isinstance(node, str) or not isinstance(param, str):
            return reject(str(node), str(param), value, "node/param must be strings")

        key = f"{node}.{param}"
        spec = self.whitelist.get(key)
        if spec is None:
            return reject(node, param, value, f"{key!r} not in whitelist")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return reject(node, param, value, "value is not a number")
        value = float(value)
        if not math.isfinite(value):
            return reject(node, param, value, "value is not finite")

        try:
            current = float(self.manager.get_params(node)[param])
        except KeyError as e:
            return reject(node, param, value, f"node/param missing in graph: {e}")

        # 1) 변화율 제한 (rate limit)
        lo_rate, hi_rate = -math.inf, math.inf
        if spec.max_rel_step is not None and current > 0:
            lo_rate = current / spec.max_rel_step
            hi_rate = current * spec.max_rel_step
        if spec.max_abs_step is not None:
            lo_rate = max(lo_rate, current - spec.max_abs_step)
            hi_rate = min(hi_rate, current + spec.max_abs_step)
        rate_limited = value < lo_rate or value > hi_rate
        applied = min(max(value, lo_rate), hi_rate)

        # 2) 절대 범위 클램프
        range_clamped = applied < spec.min or applied > spec.max
        applied = min(max(applied, spec.min), spec.max)

        if applied == current:
            status = "rejected" if (rate_limited or range_clamped) else "applied"
            reason = "no-op after clamping" if status == "rejected" else "value unchanged"
            return AuditEntry(now, source, node, param, value, current, status, reason)

        self.manager.set_params(node, {param: applied})

        if rate_limited:
            return AuditEntry(
                now, source, node, param, value, applied, "rate-limited",
                f"step limited from {current:.4g} (rel≤{spec.max_rel_step}, abs≤{spec.max_abs_step})",
            )
        if range_clamped:
            return AuditEntry(
                now, source, node, param, value, applied, "clamped",
                f"clamped to [{spec.min:.4g}, {spec.max:.4g}]",
            )
        return AuditEntry(now, source, node, param, value, applied, "applied")

    # ------------------------------------------------------------- 조회/보고
    @property
    def audit(self) -> list[AuditEntry]:
        with self._lock:
            return list(self._audit)

    def dump_audit_text(self, last_n: int | None = None) -> str:
        entries = self.audit
        if last_n is not None:
            entries = entries[-last_n:]
        if not entries:
            return "[SAFETY AUDIT] (empty)"
        return "[SAFETY AUDIT]\n" + "\n".join("  " + e.line() for e in entries)

    def describe_whitelist(self) -> str:
        """LLM 프롬프트에 넣을 허용 파라미터 명세. 현재값 포함."""
        lines = []
        for key, spec in sorted(self.whitelist.items()):
            node, param = key.split(".")
            try:
                cur = self.manager.get_params(node)[param]
                cur_s = f"{cur:.4g}"
            except KeyError:
                cur_s = "?"
            limits = []
            if spec.max_rel_step is not None:
                limits.append(f"max x{spec.max_rel_step:g} per change")
            if spec.max_abs_step is not None:
                limits.append(f"max +/-{spec.max_abs_step:g} per change")
            lines.append(
                f"- {key} = {cur_s}  (allowed range [{spec.min:g}, {spec.max:g}]"
                + (", " + ", ".join(limits) if limits else "") + ")"
            )
        return "\n".join(lines)


# ==========================================================================
# TagSafetyGuard - phorce 피벗 이후의 현행 가드
# ==========================================================================
@dataclass
class TagAuditEntry:
    wall_t: float
    source: str            # "llm" / "rules:<이유>" / "plan"
    requested: str         # 요청된 의도 태그
    accepted: str | None   # 통과한 태그. 거부면 None
    status: str            # "accepted" | "rejected"
    reason: str = ""

    def line(self) -> str:
        acc = self.accepted if self.accepted is not None else "-"
        return (f"[{self.wall_t:9.3f}] {self.status.upper():9s} "
                f"tag={self.requested!r} -> {acc!r} src={self.source}"
                + (f" ({self.reason})" if self.reason else ""))


class TagSafetyGuard:
    """의도 태그의 유일한 관문 - 화이트리스트 = 카탈로그에 존재하는 태그 집합.

    LLM이 미지의 태그를 지어내면 여기서 거부되고 폴백 태그로 간다.
    motion_id를 직접 고르는 경로는 아예 없다 (선택기만 고른다).
    """

    def __init__(self, allowed_tags, time_fn=time.monotonic) -> None:
        self.allowed = {str(t) for t in allowed_tags}
        self._time = time_fn
        self._audit: list[TagAuditEntry] = []
        self._lock = threading.Lock()

    @property
    def audit(self) -> list[TagAuditEntry]:
        with self._lock:
            return list(self._audit)

    def validate(self, tag, source: str) -> str | None:
        """통과한 태그 또는 None. 모든 시도를 감사 로그에 남긴다."""
        now = self._time()
        if not isinstance(tag, str) or not tag:
            entry = TagAuditEntry(now, source, repr(tag), None, "rejected",
                                  "not a non-empty string")
        elif tag not in self.allowed:
            entry = TagAuditEntry(now, source, tag, None, "rejected",
                                  "unknown tag (not in catalog)")
        else:
            entry = TagAuditEntry(now, source, tag, tag, "accepted")
        with self._lock:
            self._audit.append(entry)
        return entry.accepted

    def describe_whitelist(self) -> str:
        return "Available intent tags: " + ", ".join(sorted(self.allowed))
