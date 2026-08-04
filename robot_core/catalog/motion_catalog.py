"""모션 카탈로그 — 교시된 모션의 메타데이터 (젯슨 측 JSON).

모션 자체는 phorce Studio 교시로 SD카드에 있다 (코드로 못 만든다).
우리가 관리하는 것은 **선곡에 필요한 메타데이터**뿐이다:

    {id: {name, tags[], start_pose[12], initial_direction[12], duration_s, notes}}

- start_pose / initial_direction은 scripts/annotate_motion.py가 실물에서
  1회 재생으로 자동 추출한다 (교시 직후 바로 주석 완성)
- robot.motions()(적재 슬롯 정본)와 reconcile()로 대조한다:
  JSON에만 있는 id는 경고 후 제외, 슬롯에만 있는 id는 "미등록" 경고
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robot_core.hal.phorce import MOTION_ID_MAX, MOTION_ID_MIN, N_AXES


@dataclass
class MotionMeta:
    id: int
    name: str
    tags: list[str]
    start_pose: np.ndarray          # (12,) 교시 시작 자세 [rad]
    initial_direction: np.ndarray   # (12,) 초기 이동 방향 (정규화, 정지 출발 대응)
    duration_s: float
    notes: str = ""

    def __post_init__(self):
        self.id = int(self.id)
        if not (MOTION_ID_MIN <= self.id <= MOTION_ID_MAX):
            raise ValueError(f"motion id {self.id}: 1~50 범위 밖")
        self.tags = [str(t) for t in self.tags]
        self.start_pose = np.asarray(self.start_pose, dtype=float)
        self.initial_direction = np.asarray(self.initial_direction, dtype=float)
        for name, arr in (("start_pose", self.start_pose),
                          ("initial_direction", self.initial_direction)):
            if arr.shape != (N_AXES,):
                raise ValueError(f"motion {self.id} {name}: shape {arr.shape} != ({N_AXES},)")

    def to_dict(self) -> dict:
        return {"name": self.name, "tags": list(self.tags),
                "start_pose": [round(float(x), 5) for x in self.start_pose],
                "initial_direction": [round(float(x), 5) for x in self.initial_direction],
                "duration_s": round(float(self.duration_s), 3), "notes": self.notes}


class MotionCatalog:
    def __init__(self, motions: list[MotionMeta] | None = None) -> None:
        self._motions: dict[int, MotionMeta] = {}
        for m in motions or []:
            self.add(m)

    def add(self, m: MotionMeta) -> MotionMeta:
        if m.id in self._motions:
            raise ValueError(f"duplicate motion id {m.id}")
        self._motions[m.id] = m
        return m

    def get(self, motion_id: int) -> MotionMeta:
        return self._motions[int(motion_id)]

    def __contains__(self, motion_id: int) -> bool:
        return int(motion_id) in self._motions

    def __len__(self) -> int:
        return len(self._motions)

    def ids(self) -> list[int]:
        return sorted(self._motions)

    def all(self) -> list[MotionMeta]:
        return [self._motions[i] for i in self.ids()]

    def by_tag(self, tag: str) -> list[MotionMeta]:
        return [m for m in self.all() if tag in m.tags]

    def all_tags(self) -> set[str]:
        """카탈로그에 존재하는 태그 집합 — LLM 화이트리스트의 정본."""
        tags: set[str] = set()
        for m in self.all():
            tags.update(m.tags)
        return tags

    # ------------------------------------------------------------------ 대조
    def reconcile(self, loaded_ids) -> tuple[set[int], list[str]]:
        """적재 슬롯 정본(robot.motions() 결과의 id들)과 대조.

        반환: (선곡에 써도 되는 id 집합, 경고 리스트).
        JSON에만 있는 id → 경고 + 제외 (재생하면 코드 4 거절이므로 애초에 후보 제외).
        슬롯에만 있는 id → "미등록" 경고 (선곡 불가 — 메타데이터가 없어서 못 고른다).
        """
        loaded = {int(i) for i in loaded_ids}
        mine = set(self._motions)
        warnings = []
        for i in sorted(mine - loaded):
            warnings.append(f"카탈로그에만 있는 모션 {i}({self._motions[i].name!r}) — "
                            f"슬롯에 미적재. 선곡에서 제외한다")
        for i in sorted(loaded - mine):
            warnings.append(f"슬롯 {i}은 적재돼 있으나 카탈로그에 미등록 — "
                            f"scripts/annotate_motion.py로 주석을 만들 것")
        return mine & loaded, warnings

    # ------------------------------------------------------------------ I/O
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(m.id): m.to_dict() for m in self.all()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "MotionCatalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cat = cls()
        for sid, d in data.items():
            cat.add(MotionMeta(
                id=int(sid), name=d["name"], tags=d["tags"],
                start_pose=d["start_pose"], initial_direction=d["initial_direction"],
                duration_s=float(d["duration_s"]), notes=d.get("notes", "")))
        return cat


def annotate_from_recording(motion_id: int, name: str, tags: list[str],
                            arrays: dict, direction_window_frac: float = 0.2,
                            notes: str = "") -> MotionMeta:
    """재생 1회 녹화(FeedbackCache.to_arrays)에서 메타데이터 자동 추출.

    - start_pose        : 재생 시작 시점의 valid 위치
    - initial_direction : 처음 direction_window_frac 구간의 변위 (정규화)
    - duration_s        : 재생 구간 길이
    scripts/annotate_motion.py가 실물에서 이 함수를 쓴다.
    """
    playing = arrays["playing"]
    idx = np.flatnonzero(playing)
    if idx.size < 5:
        raise ValueError("녹화에 재생 구간이 없다 — 재생 중에 녹화했는지 확인")
    s, e = int(idx[0]), int(idx[-1])
    t = arrays["t"]
    duration = float(t[e] - t[s])

    start_pose = arrays["position"][s].copy()
    k = s + max(2, int((e - s) * direction_window_frac))
    disp = arrays["position"][k] - start_pose
    # invalid 축은 방향 정보 없음 → 0
    valid = arrays["valid"][s] & arrays["valid"][k]
    disp = np.where(valid, disp, 0.0)
    norm = float(np.linalg.norm(disp))
    direction = disp / norm if norm > 1e-6 else np.zeros_like(disp)

    return MotionMeta(id=motion_id, name=name, tags=tags, start_pose=start_pose,
                      initial_direction=direction, duration_s=duration, notes=notes)
