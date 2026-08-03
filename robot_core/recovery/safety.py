"""SafetyGuard — 파라미터 변경의 유일한 관문.

LLM이든 규칙 폴백이든 ROS 2 파라미터 서버든, 파라미터 변경은 전부 여기를
통과해야 한다. 원칙:

- 화이트리스트에 없는 (node, param) → 거부
- 범위 밖 값 → [min, max]로 클램프하고 경고 (거부가 아니라 클램프 —
  부분적으로라도 회복하게)
- 변화율 제한: 한 번의 적용에서 현재값 대비 ×max_rel_step / ÷max_rel_step,
  ±max_abs_step 이상 못 바꾼다 (kp 10배 튀기기 방지)
- 값이 숫자가 아니거나 NaN/inf → 거부
- 모든 시도(적용/클램프/거부)를 감사 로그에 남긴다
"""

from __future__ import annotations

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
