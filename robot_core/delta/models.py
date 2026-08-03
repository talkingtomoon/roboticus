"""델타 모델 두 벌 — 물리 모델(최소제곱) vs 소형 MLP(PyTorch 학습, numpy 추론).

공통 인터페이스:
    fit(data, joints)                      # 현장 데이터로만 학습 (사전 학습 금지)
    predict(q, qd, q_des, qd_des) -> (n,)  # 1kHz 루프용, 관절당 ~µs
    save(path) / load(path)

학습 타깃 정의:
    관측 토크 = M*qdd + (마찰+백래시 항).  M*qdd(관성)는 보정 대상이 아니므로
    타깃 Δτ = tau_measured - M̂*qdd 로 잡는다 (M̂은 최소제곱으로 함께 추정).
    보정 노드는 Δτ를 tau_ff에 더해 PD가 마찰과 싸우는 부담을 덜어준다.

정렬(alignment) 주의 — 피팅 정확도의 핵심:
    이산 제어에서 스텝 k에 적용된 토크 tau[k]는 스텝 시작 상태(q[k-1], qd[k-1])로
    계산되고, 로그의 q[k], qd[k]는 스텝 종료 상태다. 따라서 회귀는
        y = tau[k]  ~  qdd=(qd[k]-qd[k-1])/dt,  qd[k-1],  sign(qd[k-1])
    로 정렬해야 한다. 한 스텝 어긋나면 파라미터가 크게 뒤틀린다.

행 선별:
    - 물리 모델: |qd_des| > 임계인 '움직이는' 행만 (정지 스틱션 행은
      sign(qd)≈0인데 토크≠0이라 회귀를 오염시킨다)
    - MLP: 움직이는 행 + 정지 행 일부(스트라이드 샘플) — "정지면 0을 내라"도 학습
    - train/val 분리는 관절별 선별 행 안에서 시간순 75/25
      (순차 여기 계획에서 전체 시계열을 자르면 val이 통째로 다른 관절 구간이 된다)

MLP는 학습만 PyTorch를 쓰고, 추론은 가중치를 뽑아 전 관절 일괄 einsum으로 한다
(1kHz 루프에서 torch 호출 오버헤드 제거 + 현장 런타임의 torch 의존 제거).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_core.delta.collector import CalibrationData

_SIGN_EPS = 0.02  # smooth sign: tanh(qd/eps) — 0 근처 채터링 방지


def smooth_sign(v):
    return np.tanh(np.asarray(v, dtype=float) / _SIGN_EPS)


def backlash_feature(q: np.ndarray, qd_des: np.ndarray, width: float,
                     deadband: float = 0.01) -> np.ndarray:
    """반전 직후 |q - q_reversal| < width 구간에서 sign(qd_des), 그 외 0. (N,)

    백래시(데드존)는 방향 반전 직후 짧은 이동 구간에서만 토크 특징을 남긴다.
    """
    sign = np.where(qd_des > deadband, 1, np.where(qd_des < -deadband, -1, 0))
    feature = np.zeros(len(q))
    last_sign = 0
    q_rev = q[0]
    for i in range(len(q)):
        s = sign[i]
        if s != 0:
            if last_sign != 0 and s != last_sign:
                q_rev = q[i]
            last_sign = s
            if abs(q[i] - q_rev) < width:
                feature[i] = s
    return feature


def _moving_rows(data: CalibrationData, j: int, thresh: float = 0.01,
                 rest_stride: int = 0) -> np.ndarray:
    """관절 j가 '지령상 움직이는' 행 인덱스 (k>=1 보장 — k-1 정렬용).

    rest_stride > 0 이면 정지 행도 그 간격으로 섞는다 (MLP용).
    """
    moving = np.abs(data.qd_des[:, j]) > thresh
    idx = np.flatnonzero(moving)
    if rest_stride > 0:
        rest = np.flatnonzero(~moving)[::rest_stride]
        idx = np.sort(np.concatenate([idx, rest]))
    return idx[idx >= 1]


def _aligned_features(data: CalibrationData, j: int, idx: np.ndarray):
    """정렬된 (y, qdd, qd_prev, q_prev, q_des_prev). y=tau[k], 나머지는 k-1 기준."""
    rate = data.rate_hz
    y = data.tau[idx, j]
    qd_prev = data.qd[idx - 1, j]
    qdd = (data.qd[idx, j] - qd_prev) * rate
    q_prev = data.q[idx - 1, j]
    q_des_prev = data.q_des[idx - 1, j]
    return y, qdd, qd_prev, q_prev, q_des_prev


def _split(idx: np.ndarray, val_frac: float = 0.25):
    n_val = max(1, int(len(idx) * val_frac))
    return idx[:-n_val], idx[-n_val:]


def _trimmed_lstsq(X: np.ndarray, y: np.ndarray, rounds: int = 3,
                   k_sigma: float = 4.0) -> np.ndarray:
    """이상치에 강건한 최소제곱: 잔차가 큰 행을 반복 제거하며 재피팅.

    백래시 재체결 순간 부하측 속도가 한 스텝에 점프한다(기어 슬램) —
    이 소수의 거대한 qdd 이상치가 일반 L2 회귀를 통째로 무너뜨린다.
    잔차의 MAD 기반 σ로 |r| > k·σ 행을 버리고 다시 피팅한다.
    """
    keep = np.ones(len(y), dtype=bool)
    theta = None
    for _ in range(rounds):
        theta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        r = y - X @ theta
        sigma = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-12
        new_keep = np.abs(r) < k_sigma * sigma
        if new_keep.sum() < max(50, X.shape[1] * 5) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return theta


# ------------------------------------------------------------------ 물리 모델
@dataclass
class _JointParams:
    inertia: float = 0.0
    viscous: float = 0.0
    coulomb: float = 0.0
    k_backlash: float = 0.0
    backlash_width: float = 0.0
    val_rms: float = float("nan")


class PhysicsDeltaModel:
    """tau ≈ M*qdd + b*qd + tau_c*sign(qd) + k_bl*백래시항. 관절당 파라미터 ~4.

    선형 최소제곱 (백래시 폭만 그리드 탐색). 해석 가능, 과적합 불가.
    """

    model_type = "physics"

    def __init__(self, n_joints: int) -> None:
        self.n_joints = n_joints
        self.params: list[_JointParams] = [_JointParams() for _ in range(n_joints)]
        self.afc_state = "unknown"   # fit() 시 데이터에서 복사 — AFC 가드용
        self._rt_last_sign = np.zeros(n_joints)
        self._rt_q_rev = np.zeros(n_joints)
        self._rt_has_rev = np.zeros(n_joints, dtype=bool)  # 첫 반전 관측 전엔 킥 금지

    # ---------------------------------------------------------------- 학습
    def fit(self, data: CalibrationData, joints: list[int] | None = None,
            val_frac: float = 0.25, engaged_qd: float = 0.02,
            backlash_widths=(0.005, 0.01, 0.02, 0.04)) -> "PhysicsDeltaModel":
        """2단계 피팅.

        1단계: 링크가 실제로 움직이는 '깨끗한' 행(|qd_prev| > engaged_qd)에서
               M, b, tau_c를 하드 sign으로 최소제곱.
               (백래시 데드존을 건너는 동안은 측정 속도가 0인데 토크는 램프한다 —
                이 행들을 넣으면 세 파라미터가 전부 뒤틀린다. 부드러운 sign도
                저속 행에서 tau_c를 과소평가하므로 피팅에는 하드 sign을 쓴다.)
        2단계: 전체 이동 행의 잔차에 백래시 킥 항만 폭 그리드로 피팅.
               val 잔차가 실제로 줄어들 때만 채택한다.
        """
        joints = joints if joints is not None else data.joints
        self.afc_state = data.afc_state
        for j in joints:
            rows_all = _moving_rows(data, j)
            if len(rows_all) < 100:
                raise ValueError(f"joint {j}: 움직이는 샘플 {len(rows_all)}개 — 데이터 부족")
            _, _, qd_prev_all, _, _ = _aligned_features(data, j, rows_all)
            rows_clean = rows_all[np.abs(qd_prev_all) > engaged_qd]
            if len(rows_clean) < 100:
                raise ValueError(f"joint {j}: 링크 이동 샘플 {len(rows_clean)}개 — 데이터 부족")

            # --- 1단계: M, b, tau_c ---
            tr, va = _split(rows_clean, val_frac)

            def design(idx):
                y, qdd, qd_prev, _, _ = _aligned_features(data, j, idx)
                return np.column_stack([qdd, qd_prev, np.sign(qd_prev)]), y

            X_tr, y_tr = design(tr)
            theta = _trimmed_lstsq(X_tr, y_tr)
            M_hat, b_hat, tc_hat = (float(x) for x in theta)

            # --- 2단계: 백래시 킥 (전체 이동 행의 잔차에서) ---
            tr_a, va_a = _split(rows_all, val_frac)

            def residual(idx):
                y, qdd, qd_prev, _, _ = _aligned_features(data, j, idx)
                return y - (M_hat * qdd + b_hat * qd_prev + tc_hat * np.sign(qd_prev))

            r_tr, r_va = residual(tr_a), residual(va_a)
            rms_no_bl = float(np.sqrt(np.mean(r_va ** 2)))
            best_bl = (rms_no_bl, 0.0, 0.0)
            for w in backlash_widths:
                bl = backlash_feature(data.q[:, j], data.qd_des[:, j], w)
                f_tr, f_va = bl[tr_a - 1], bl[va_a - 1]
                denom = float(f_tr @ f_tr)
                if denom < 1e-9:
                    continue
                k = float(f_tr @ r_tr) / denom
                rms = float(np.sqrt(np.mean((r_va - k * f_va) ** 2)))
                if rms < best_bl[0]:
                    best_bl = (rms, w, k)

            val_rms, w_bl, k_bl = best_bl
            self.params[j] = _JointParams(
                inertia=M_hat, viscous=b_hat, coulomb=tc_hat,
                k_backlash=k_bl, backlash_width=w_bl, val_rms=val_rms)
        return self

    # ---------------------------------------------------------------- 추론
    def reset(self) -> None:
        self._rt_last_sign[:] = 0.0
        self._rt_q_rev[:] = 0.0
        self._rt_has_rev[:] = False

    def predict(self, q, qd, q_des, qd_des) -> np.ndarray:
        """마찰+백래시 보정 토크. 관성항(M*qdd)은 보정하지 않는다."""
        cb = np.array([p.coulomb for p in self.params])
        bv = np.array([p.viscous for p in self.params])
        delta = cb * smooth_sign(qd_des) + bv * np.asarray(qd_des, dtype=float)
        for j, p in enumerate(self.params):
            if p.k_backlash != 0.0 and p.backlash_width > 0:
                v = qd_des[j]
                s = 1.0 if v > 0.01 else (-1.0 if v < -0.01 else 0.0)
                if s != 0.0:
                    if self._rt_last_sign[j] != 0 and s != self._rt_last_sign[j]:
                        self._rt_q_rev[j] = q[j]
                        self._rt_has_rev[j] = True
                    self._rt_last_sign[j] = s
                    # 첫 반전을 실제로 관측한 뒤에만 킥 — 시작 위치가 우연히
                    # q_rev 초기값 근처라는 이유로 발동하면 안 된다
                    if self._rt_has_rev[j] and abs(q[j] - self._rt_q_rev[j]) < p.backlash_width:
                        delta[j] += p.k_backlash * s
        return delta

    def _predict_rows(self, data: CalibrationData, j: int, idx: np.ndarray) -> np.ndarray:
        """벤치마크용: 정렬된 행에서의 마찰 예측 (관성항 제외)."""
        p = self.params[j]
        _, _, qd_prev, _, _ = _aligned_features(data, j, idx)
        out = p.coulomb * smooth_sign(qd_prev) + p.viscous * qd_prev
        if p.k_backlash != 0.0 and p.backlash_width > 0:
            bl = backlash_feature(data.q[:, j], data.qd_des[:, j], p.backlash_width)
            out = out + p.k_backlash * bl[idx - 1]
        return out

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        arr = np.array([[p.inertia, p.viscous, p.coulomb, p.k_backlash,
                         p.backlash_width, p.val_rms] for p in self.params])
        np.savez(path, model_type=np.str_(self.model_type), params=arr,
                 afc_state=np.str_(self.afc_state))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsDeltaModel":
        z = np.load(Path(path), allow_pickle=False)
        assert str(z["model_type"]) == cls.model_type
        arr = z["params"]
        m = cls(arr.shape[0])
        for j, row in enumerate(arr):
            m.params[j] = _JointParams(*[float(x) for x in row])
        m.afc_state = str(z["afc_state"]) if "afc_state" in z.files else "unknown"
        return m

    def sanity_warnings(self) -> list[str]:
        """피팅 결과가 물리적으로 말이 되는지 — 현장 3단계 판정 자동화.

        AFC(액티브 마찰제거)가 켜진 로봇에서는 남은 마찰이 거의 없으므로
        tau_c가 0 근처(살짝 음수 포함)인 것이 **정상**이다. AFC off/unknown일
        때만 '음수 마찰 = 이상' 판정을 적용한다.
        """
        warns = []
        # AFC on: 잔여 마찰이 0 근처가 정상 → 크게 음수일 때만 이상
        tc_floor = -0.5 if self.afc_state == "on" else -0.02
        b_floor = -0.1 if self.afc_state == "on" else -0.01
        for j, p in enumerate(self.params):
            if p.coulomb < tc_floor:
                warns.append(f"joint {j}: tau_c={p.coulomb:.3f} < {tc_floor} — "
                             f"데이터 이상 의심 (AFC={self.afc_state})")
            if p.viscous < b_floor:
                warns.append(f"joint {j}: b={p.viscous:.3f} < {b_floor} — "
                             f"데이터 이상 의심 (AFC={self.afc_state})")
            if p.inertia <= 0:
                warns.append(f"joint {j}: M={p.inertia:.4f} <= 0 — 정렬/여기 부족 의심")
        if self.afc_state == "on" and any(p.coulomb > 0.5 for p in self.params):
            warns.append("AFC on인데 tau_c가 큼 — AFC 설정 확인 또는 afc_state 오기록 의심")
        return warns

    def info(self) -> str:
        lines = [f"[PHYSICS MODEL] tau_delta = tau_c*sign(qd) + b*qd + backlash"
                 f"  (AFC={self.afc_state})"]
        for j, p in enumerate(self.params):
            lines.append(
                f"  joint {j}: M={p.inertia:.4f}  b={p.viscous:.4f}  "
                f"tau_c={p.coulomb:.4f}  k_bl={p.k_backlash:.4f}"
                f"@w={p.backlash_width:.3f}  val_rms={p.val_rms:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- MLP
class MLPDeltaModel:
    """관절별 독립 소형 MLP (4 → 32 → 32 → 1, tanh).

    입력: [q, qd, sign(qd), q_des - q] (관절별). 출력: Δτ.
    학습은 PyTorch(CPU), 추론은 전 관절 일괄 einsum (numpy) — 관절 수와 거의
    무관하게 수십 µs.
    """

    model_type = "mlp"
    HIDDEN = 32

    def __init__(self, n_joints: int) -> None:
        self.n_joints = n_joints
        self.weights: list[list[tuple[np.ndarray, np.ndarray]] | None] = [None] * n_joints
        self.x_mean = np.zeros((n_joints, 4))
        self.x_std = np.ones((n_joints, 4))
        self.y_scale = np.ones(n_joints)
        self.val_rms = np.full(n_joints, np.nan)
        self.afc_state = "unknown"   # fit() 시 데이터에서 복사 — AFC 가드용
        self._packed = None   # 일괄 추론용 스택 가중치

    @staticmethod
    def _features(q, qd, q_des) -> np.ndarray:
        return np.stack([q, qd, np.asarray(smooth_sign(qd)), q_des - q], axis=-1)

    # ---------------------------------------------------------------- 학습
    def fit(self, data: CalibrationData, joints: list[int] | None = None, *,
            inertia: np.ndarray | float | None = None, val_frac: float = 0.25,
            epochs: int = 120, batch: int = 4096, lr: float = 3e-3,
            max_samples: int = 60_000, time_budget_s: float = 180.0,
            rest_stride: int = 20, seed: int = 0, on_log=None) -> "MLPDeltaModel":
        """타깃 = tau - M̂*qdd. M̂은 inertia 인자로 (물리 모델의 것을 재사용 권장)."""
        try:
            import torch
            from torch import nn
        except ImportError as e:
            raise ImportError(
                "MLP 학습에는 torch가 필요하다: pip install torch "
                "(물리 모델은 torch 없이 동작)") from e

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        joints = joints if joints is not None else data.joints
        self.afc_state = data.afc_state
        t_start = time.perf_counter()

        for j in joints:
            rows = _moving_rows(data, j, rest_stride=rest_stride)
            if len(rows) < 100:
                raise ValueError(f"joint {j}: 학습 샘플 {len(rows)}개 — 데이터 부족")
            tr, va = _split(rows, val_frac)

            m_hat = 0.0 if inertia is None else float(np.atleast_1d(inertia)[
                j if np.ndim(inertia) else 0])

            def build(idx):
                y, qdd, qd_prev, q_prev, q_des_prev = _aligned_features(data, j, idx)
                X = self._features(q_prev, qd_prev, q_des_prev)
                return X, y - m_hat * qdd

            X_tr, y_tr = build(tr)
            X_va, y_va = build(va)
            if len(X_tr) > max_samples:
                sel = rng.choice(len(X_tr), max_samples, replace=False)
                X_tr, y_tr = X_tr[sel], y_tr[sel]

            mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
            ys = float(np.std(y_tr) + 1e-9)
            self.x_mean[j], self.x_std[j], self.y_scale[j] = mu, sd, ys

            Xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32)
            yt = torch.tensor(y_tr / ys, dtype=torch.float32).unsqueeze(1)
            Xv = torch.tensor((X_va - mu) / sd, dtype=torch.float32)
            yv = torch.tensor(y_va / ys, dtype=torch.float32).unsqueeze(1)

            net = nn.Sequential(
                nn.Linear(4, self.HIDDEN), nn.Tanh(),
                nn.Linear(self.HIDDEN, self.HIDDEN), nn.Tanh(),
                nn.Linear(self.HIDDEN, 1))
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            best_val, best_state, patience = float("inf"), None, 0
            for epoch in range(epochs):
                perm = torch.randperm(len(Xt))
                for k in range(0, len(Xt), batch):
                    sel = perm[k:k + batch]
                    opt.zero_grad()
                    loss = loss_fn(net(Xt[sel]), yt[sel])
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    val = float(loss_fn(net(Xv), yv))
                if val < best_val - 1e-5:
                    best_val, patience = val, 0
                    best_state = [p.detach().clone() for p in net.parameters()]
                else:
                    patience += 1
                    if patience >= 12:
                        break
                if time.perf_counter() - t_start > time_budget_s:
                    if on_log:
                        on_log(f"joint {j}: 시간 예산 도달, epoch {epoch}에서 마감")
                    break

            if best_state is not None:
                with torch.no_grad():
                    for p, s in zip(net.parameters(), best_state):
                        p.copy_(s)
            layers = [m for m in net if isinstance(m, nn.Linear)]
            self.weights[j] = [(l.weight.detach().numpy().astype(float).copy(),
                                l.bias.detach().numpy().astype(float).copy())
                               for l in layers]
            self.val_rms[j] = float(np.sqrt(best_val)) * ys
            if on_log:
                on_log(f"joint {j}: MLP val RMS = {self.val_rms[j]:.4f} Nm")
        self._pack()
        return self

    # ---------------------------------------------------------------- 추론
    def _pack(self) -> None:
        """관절별 망을 블록 행렬 하나로 합친다 — predict가 행렬곱 3번이면 끝.

        (관절별로 따로 forward 하면 numpy 호출 오버헤드가 관절 수만큼 곱해진다.
        블록 대각으로 합치면 관절 수와 거의 무관하게 수십 µs.)
        """
        n, H = self.n_joints, self.HIDDEN
        W1B = np.zeros((n * H, n * 4)); b1f = np.zeros(n * H)
        W2B = np.zeros((n * H, n * H)); b2f = np.zeros(n * H)
        W3B = np.zeros((n, n * H)); b3f = np.zeros(n)
        scale = np.zeros(n)  # 미학습 관절은 출력 0
        for j, w in enumerate(self.weights):
            if w is None:
                continue
            (W1, b1), (W2, b2), (W3, b3) = w
            W1B[j * H:(j + 1) * H, j * 4:(j + 1) * 4] = W1
            b1f[j * H:(j + 1) * H] = b1
            W2B[j * H:(j + 1) * H, j * H:(j + 1) * H] = W2
            b2f[j * H:(j + 1) * H] = b2
            W3B[j, j * H:(j + 1) * H] = W3[0]
            b3f[j] = b3[0]
            scale[j] = self.y_scale[j]
        self._packed = (W1B, b1f, W2B, b2f, W3B, b3f, scale)

    def reset(self) -> None:
        pass

    def predict(self, q, qd, q_des, qd_des) -> np.ndarray:
        if self._packed is None:
            self._pack()
        W1B, b1f, W2B, b2f, W3B, b3f, scale = self._packed
        x = self._features(np.asarray(q, dtype=float), np.asarray(qd, dtype=float),
                           np.asarray(q_des, dtype=float))          # (n, 4)
        flat = ((x - self.x_mean) / self.x_std).ravel()              # (4n,)
        h = np.tanh(W1B @ flat + b1f)
        h = np.tanh(W2B @ h + b2f)
        return (W3B @ h + b3f) * scale

    def _predict_rows(self, data: CalibrationData, j: int, idx: np.ndarray) -> np.ndarray:
        """벤치마크용: 정렬된 행에서의 예측."""
        w = self.weights[j]
        if w is None:
            return np.zeros(len(idx))
        _, _, qd_prev, q_prev, q_des_prev = _aligned_features(data, j, idx)
        X = (self._features(q_prev, qd_prev, q_des_prev) - self.x_mean[j]) / self.x_std[j]
        (W1, b1), (W2, b2), (W3, b3) = w
        h = np.tanh(X @ W1.T + b1)
        h = np.tanh(h @ W2.T + b2)
        return (h @ W3.T + b3)[:, 0] * self.y_scale[j]

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        payload = {"model_type": np.str_(self.model_type),
                   "n_joints": np.int64(self.n_joints),
                   "x_mean": self.x_mean, "x_std": self.x_std,
                   "y_scale": self.y_scale, "val_rms": self.val_rms,
                   "afc_state": np.str_(self.afc_state)}
        for j, w in enumerate(self.weights):
            if w is None:
                continue
            for li, (W, b) in enumerate(w):
                payload[f"j{j}_W{li}"] = W
                payload[f"j{j}_b{li}"] = b
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLPDeltaModel":
        z = np.load(Path(path), allow_pickle=False)
        assert str(z["model_type"]) == cls.model_type
        m = cls(int(z["n_joints"]))
        m.x_mean, m.x_std = z["x_mean"].copy(), z["x_std"].copy()
        m.y_scale, m.val_rms = z["y_scale"].copy(), z["val_rms"].copy()
        m.afc_state = str(z["afc_state"]) if "afc_state" in z.files else "unknown"
        for j in range(m.n_joints):
            if f"j{j}_W0" in z:
                m.weights[j] = [(z[f"j{j}_W{li}"].copy(), z[f"j{j}_b{li}"].copy())
                                for li in range(3)]
        m._pack()
        return m

    def info(self) -> str:
        lines = [f"[MLP MODEL] per-joint 4-{self.HIDDEN}-{self.HIDDEN}-1 (numpy inference)"]
        for j in range(self.n_joints):
            state = "trained" if self.weights[j] is not None else "untrained"
            lines.append(f"  joint {j}: {state}, val_rms={self.val_rms[j]:.4f}")
        return "\n".join(lines)


# ------------------------------------------------------------------ 벤치마크
def compare_models(data: CalibrationData, physics: PhysicsDeltaModel,
                   mlp: MLPDeltaModel | None, joints: list[int] | None = None,
                   val_frac: float = 0.25) -> tuple[str, dict]:
    """관절별 val 행 잔차 RMS 3열 비교표: 무보정 vs 물리 vs MLP.

    공통 타깃 = tau - M̂*qdd (물리 모델의 M̂). '무보정'은 타깃 자체의 RMS —
    보정 안 하면 PD가 그만큼의 토크를 오차로 감당해야 한다는 뜻.
    val 행은 학습과 동일한 규칙(관절별 움직이는 행의 뒤 25%)이다.
    """
    joints = joints if joints is not None else data.joints
    rows_out = {}
    lines = ["[MODEL COMPARISON] val 구간 잔차 RMS [Nm] (낮을수록 좋음)",
             "  jnt |  무보정  |  물리모델 |   MLP    | 권장"]
    lines.append("  " + "-" * 52)
    for j in joints:
        rows = _moving_rows(data, j)
        _, va = _split(rows, val_frac)
        y, qdd, _, _, _ = _aligned_features(data, j, va)
        target = y - physics.params[j].inertia * qdd

        rms0 = float(np.sqrt(np.mean(target ** 2)))
        rms_ph = float(np.sqrt(np.mean((target - physics._predict_rows(data, j, va)) ** 2)))
        rms_ml = (float(np.sqrt(np.mean((target - mlp._predict_rows(data, j, va)) ** 2)))
                  if mlp is not None else float("nan"))
        # MLP가 물리 대비 15% 이상 나을 때만 MLP 권장 (단순한 쪽 우선)
        pick = "mlp" if (mlp is not None and rms_ml < rms_ph * 0.85) else "physics"
        rows_out[j] = {"uncorrected": rms0, "physics": rms_ph, "mlp": rms_ml, "pick": pick}
        lines.append(f"  {j:3d} | {rms0:8.4f} | {rms_ph:8.4f} | {rms_ml:8.4f} | {pick}")

    picks = [r["pick"] for r in rows_out.values()]
    overall = "mlp" if picks and all(p == "mlp" for p in picks) else "physics"
    lines.append(f"  전체 권장: {overall}  (동률/불명확하면 물리 모델 — 디버깅 가능한 쪽)")
    return "\n".join(lines), {"per_joint": rows_out, "overall": overall}
