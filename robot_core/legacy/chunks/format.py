"""MotionChunk — 시간 파라미터화된 관절 궤적.

내부 표현은 구간별 큐빅 스플라인 계수다 (waypoint 배열 금지 이유:
임의 시점 샘플링과 해석적 미분(qd, qdd)이 스위칭 채점·블렌딩에 필요하다).

세그먼트 i (times[i] <= t < times[i+1]), dt = t - times[i]:
    q(t)   = c0 + c1*dt + c2*dt^2 + c3*dt^3
    qd(t)  = c1 + 2*c2*dt + 3*c3*dt^2
    qdd(t) = 2*c2 + 6*c3*dt

coeffs.shape = (K, n_joints, 4), times.shape = (K+1,)

npz 포맷 (MuJoCo 등 외부 궤적 임포트를 염두에 둔 호환 설계):
    times, coeffs, name, tags, fmt_version
외부 궤적은 waypoint로 받아 from_waypoints()로 스플라인 피팅하면 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FMT_VERSION = 1


class MotionChunk:
    def __init__(self, name: str, times: np.ndarray, coeffs: np.ndarray,
                 tags: list[str] | None = None) -> None:
        times = np.asarray(times, dtype=float)
        coeffs = np.asarray(coeffs, dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times must be 1-D with at least 2 knots")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing")
        if coeffs.ndim != 3 or coeffs.shape[0] != len(times) - 1 or coeffs.shape[2] != 4:
            raise ValueError(f"coeffs shape {coeffs.shape}, expected (K={len(times)-1}, n, 4)")
        self.name = str(name)
        self.tags = list(tags or [])
        self.times = times - times[0]  # 항상 t=0 시작으로 정규화
        self.coeffs = coeffs

    # ---------------------------------------------------------------- 속성
    @property
    def n_joints(self) -> int:
        return self.coeffs.shape[1]

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def q_start(self) -> np.ndarray:
        return self.sample(0.0)[0]

    @property
    def qd_start(self) -> np.ndarray:
        return self.sample(0.0)[1]

    @property
    def q_end(self) -> np.ndarray:
        return self.sample(self.duration)[0]

    @property
    def qd_end(self) -> np.ndarray:
        return self.sample(self.duration)[1]

    # -------------------------------------------------------------- 샘플링
    def sample(self, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, qd, qdd)를 해석적으로 계산. t는 스칼라 또는 (T,) 배열.

        범위 밖 t는 양끝에 클램프된다 (끝나면 마지막 자세 홀드 — qd/qdd는
        그 지점의 스플라인 값이므로, '정지 홀드'가 필요하면 호출측에서 처리).
        스칼라 → (n,), 배열 → (T, n).
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        t_clip = np.clip(t_arr, 0.0, self.duration)
        idx = np.clip(np.searchsorted(self.times, t_clip, side="right") - 1,
                      0, len(self.times) - 2)
        dt = (t_clip - self.times[idx])[:, None]          # (T, 1)
        c = self.coeffs[idx]                              # (T, n, 4)
        c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
        q = c0 + c1 * dt + c2 * dt**2 + c3 * dt**3
        qd = c1 + 2 * c2 * dt + 3 * c3 * dt**2
        qdd = 2 * c2 + 6 * c3 * dt
        if np.isscalar(t) or np.ndim(t) == 0:
            return q[0], qd[0], qdd[0]
        return q, qd, qdd

    # ------------------------------------------------------------ 생성자들
    @classmethod
    def from_waypoints(
        cls, name: str, times, waypoints, *,
        qd_start=None, qd_end=None, tags: list[str] | None = None,
    ) -> "MotionChunk":
        """경유점을 지나는 클램프드 큐빅 스플라인 (양끝 속도 경계조건 지정 가능).

        times: (K+1,), waypoints: (K+1, n). qd_start/qd_end 기본값 0 (정지).
        내부는 C2 연속이다.
        """
        times = np.asarray(times, dtype=float)
        wp = np.atleast_2d(np.asarray(waypoints, dtype=float))
        if wp.shape[0] != len(times):
            raise ValueError(f"waypoints rows {wp.shape[0]} != len(times) {len(times)}")
        n = wp.shape[1]
        v0 = np.zeros(n) if qd_start is None else np.asarray(qd_start, dtype=float)
        vK = np.zeros(n) if qd_end is None else np.asarray(qd_end, dtype=float)

        coeffs = _clamped_cubic_coeffs(times, wp, v0, vK)
        return cls(name, times, coeffs, tags)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        np.savez(
            path,
            fmt_version=np.int64(FMT_VERSION),
            name=np.str_(self.name),
            tags=np.array(self.tags, dtype="U64"),
            times=self.times,
            coeffs=self.coeffs,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MotionChunk":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            name=str(data["name"]),
            times=data["times"],
            coeffs=data["coeffs"],
            tags=[str(t) for t in data["tags"]],
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (f"MotionChunk({self.name!r}, {self.n_joints}j, {self.duration:.2f}s, "
                f"tags={self.tags})")


def _clamped_cubic_coeffs(times: np.ndarray, wp: np.ndarray,
                          v0: np.ndarray, vK: np.ndarray) -> np.ndarray:
    """클램프드 큐빅 스플라인 계수. 모멘트(2차 미분) 방법, 전 관절 동시 해."""
    K = len(times) - 1
    h = np.diff(times)                      # (K,)
    n = wp.shape[1]

    # A @ M = rhs,  M: (K+1, n) 각 매듭의 2차 미분
    A = np.zeros((K + 1, K + 1))
    rhs = np.zeros((K + 1, n))
    A[0, 0] = h[0] / 3.0
    A[0, 1] = h[0] / 6.0
    rhs[0] = (wp[1] - wp[0]) / h[0] - v0
    for i in range(1, K):
        A[i, i - 1] = h[i - 1] / 6.0
        A[i, i] = (h[i - 1] + h[i]) / 3.0
        A[i, i + 1] = h[i] / 6.0
        rhs[i] = (wp[i + 1] - wp[i]) / h[i] - (wp[i] - wp[i - 1]) / h[i - 1]
    A[K, K - 1] = h[K - 1] / 6.0
    A[K, K] = h[K - 1] / 3.0
    rhs[K] = vK - (wp[K] - wp[K - 1]) / h[K - 1]

    M = np.linalg.solve(A, rhs)             # (K+1, n)

    coeffs = np.zeros((K, n, 4))
    for i in range(K):
        hi = h[i]
        coeffs[i, :, 0] = wp[i]
        coeffs[i, :, 1] = (wp[i + 1] - wp[i]) / hi - hi * (2 * M[i] + M[i + 1]) / 6.0
        coeffs[i, :, 2] = M[i] / 2.0
        coeffs[i, :, 3] = (M[i + 1] - M[i]) / (6.0 * hi)
    return coeffs


class ChunkDictionary:
    """청크 모음. 디렉토리 하나 = 딕셔너리 하나 (청크당 npz 파일 하나)."""

    def __init__(self, chunks: list[MotionChunk] | None = None) -> None:
        self._chunks: dict[str, MotionChunk] = {}
        for c in chunks or []:
            self.add(c)

    def add(self, chunk: MotionChunk) -> MotionChunk:
        if chunk.name in self._chunks:
            raise ValueError(f"duplicate chunk name {chunk.name!r}")
        self._chunks[chunk.name] = chunk
        return chunk

    def get(self, name: str) -> MotionChunk:
        if name not in self._chunks:
            raise KeyError(f"no chunk {name!r} (have: {sorted(self._chunks)})")
        return self._chunks[name]

    def names(self) -> list[str]:
        return sorted(self._chunks)

    def all(self) -> list[MotionChunk]:
        return [self._chunks[k] for k in self.names()]

    def by_tag(self, tag: str) -> list[MotionChunk]:
        return [c for c in self.all() if tag in c.tags]

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, name: str) -> bool:
        return name in self._chunks

    # ------------------------------------------------------------------ I/O
    def save_dir(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for chunk in self.all():
            chunk.save(directory / f"{chunk.name}.npz")
        return directory

    @classmethod
    def load_dir(cls, directory: str | Path) -> "ChunkDictionary":
        directory = Path(directory)
        d = cls()
        for path in sorted(directory.glob("*.npz")):
            d.add(MotionChunk.load(path))
        return d
