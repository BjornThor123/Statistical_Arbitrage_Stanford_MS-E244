from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


ArrayLike = np.ndarray


def _phi_power_law(theta: ArrayLike, eta: float, gamma: float) -> ArrayLike:
    safe_theta = np.maximum(np.asarray(theta, dtype=np.float64), 1e-12)
    return eta * np.power(safe_theta, -gamma)


def ssvi_total_variance(
    k: ArrayLike,
    theta: ArrayLike,
    rho: float,
    eta: float,
    gamma: float,
) -> ArrayLike:
    k = np.asarray(k, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = _phi_power_law(theta, eta=eta, gamma=gamma)
    z = phi * k
    rad = np.sqrt((z + rho) ** 2 + (1.0 - rho * rho))
    return 0.5 * theta * (1.0 + rho * z + rad)


def ssvi_dw_dk(
    k: ArrayLike,
    theta: ArrayLike,
    rho: float,
    eta: float,
    gamma: float,
) -> ArrayLike:
    k = np.asarray(k, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = _phi_power_law(theta, eta=eta, gamma=gamma)
    z = phi * k
    rad = np.sqrt((z + rho) ** 2 + (1.0 - rho * rho))
    return 0.5 * theta * phi * (rho + (z + rho) / np.maximum(rad, 1e-12))


@dataclass
class SSVIParameters:
    rho: float
    eta: float
    gamma: float


@dataclass
class SSVIDiagnostics:
    rmse_total_variance: float
    rmse_implied_volatility: float
    no_butterfly_constraint_score: float
    n_points: int
    n_maturities: int


@dataclass
class CalibratedSSVISurface:
    params: SSVIParameters
    t_nodes: ArrayLike
    theta_nodes: ArrayLike
    diagnostics: SSVIDiagnostics

    def theta_at(self, t: ArrayLike) -> ArrayLike:
        t = np.asarray(t, dtype=np.float64)
        return np.interp(
            np.maximum(t, self.t_nodes[0]),
            self.t_nodes,
            self.theta_nodes,
            left=self.theta_nodes[0],
            right=self.theta_nodes[-1],
        )

    def total_variance(self, k: ArrayLike, t: ArrayLike) -> ArrayLike:
        theta = self.theta_at(t)
        return ssvi_total_variance(
            k=k,
            theta=theta,
            rho=self.params.rho,
            eta=self.params.eta,
            gamma=self.params.gamma,
        )

    def implied_volatility(self, k: ArrayLike, t: ArrayLike) -> ArrayLike:
        t = np.asarray(t, dtype=np.float64)
        w = self.total_variance(k, t)
        return np.sqrt(np.maximum(w, 1e-12) / np.maximum(t, 1e-12))

    def sigma_skew(self, t: float, k0: float = 0.0) -> float:
        theta = float(self.theta_at(np.asarray([t]))[0])
        w = float(
            ssvi_total_variance(
                np.asarray([k0]),
                np.asarray([theta]),
                rho=self.params.rho,
                eta=self.params.eta,
                gamma=self.params.gamma,
            )[0]
        )
        dwdk = float(
            ssvi_dw_dk(
                np.asarray([k0]),
                np.asarray([theta]),
                rho=self.params.rho,
                eta=self.params.eta,
                gamma=self.params.gamma,
            )[0]
        )
        denom = 2.0 * np.sqrt(np.maximum(w, 1e-12)) * np.sqrt(np.maximum(t, 1e-12))
        return float(dwdk / np.maximum(denom, 1e-12))

    def to_dict(self) -> Dict[str, object]:
        return {
            "params": asdict(self.params),
            "t_nodes": self.t_nodes.tolist(),
            "theta_nodes": self.theta_nodes.tolist(),
            "diagnostics": asdict(self.diagnostics),
        }


def _validate_inputs(k: ArrayLike, t: ArrayLike, sigma: ArrayLike) -> None:
    if not (k.shape == t.shape == sigma.shape):
        raise ValueError(f"Input shapes must match. Got {k.shape}, {t.shape}, {sigma.shape}")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(t)) or np.any(~np.isfinite(sigma)):
        raise ValueError("Inputs contain NaN/Inf values.")
    if np.any(t <= 0.0):
        raise ValueError("All maturities t must be strictly positive.")
    if np.any(sigma <= 0.0):
        raise ValueError("All implied volatilities must be strictly positive.")


def _maturity_keys(t: ArrayLike, decimals: int = 8) -> Tuple[ArrayLike, ArrayLike]:
    t_key = np.round(np.asarray(t, dtype=np.float64), decimals=decimals)
    uniq = np.unique(t_key)
    return t_key, uniq


def _initial_theta_from_atm(k: ArrayLike, t_key: ArrayLike, w: ArrayLike, uniq_t: ArrayLike) -> ArrayLike:
    theta = np.zeros(len(uniq_t), dtype=np.float64)
    for i, u in enumerate(uniq_t):
        mask = t_key == u
        k_u = k[mask]
        w_u = w[mask]
        if len(k_u) == 0:
            theta[i] = 0.05
            continue
        cutoff = np.quantile(np.abs(k_u), 0.25)
        atm = w_u[np.abs(k_u) <= max(cutoff, 1e-4)]
        if len(atm) == 0:
            atm = w_u[np.argsort(np.abs(k_u))[: min(3, len(k_u))]]
        theta[i] = float(np.maximum(np.median(atm), 1e-8))
    theta = np.maximum.accumulate(theta)
    return np.maximum(theta, 1e-8)


def _constraint_penalty(rho: float, eta: float, gamma: float) -> float:
    p = 0.0
    if abs(rho) >= 0.999:
        p += 1e3 * (abs(rho) - 0.999) ** 2
    if eta <= 0.0:
        p += 1e3 + 1e3 * eta * eta
    if not (0.0 < gamma < 1.0):
        d = min(abs(gamma - 1e-6), abs(gamma - (1.0 - 1e-6)))
        p += 1e3 + 1e3 * d * d
    # Conservative sufficient condition often used in practice for SSVI power-law.
    p += 10.0 * max(0.0, eta * (1.0 + abs(rho)) - 2.0) ** 2
    return p


def _loss_for_params(
    rho: float,
    eta: float,
    gamma: float,
    k: ArrayLike,
    theta_for_obs: ArrayLike,
    w_true: ArrayLike,
    weights: ArrayLike,
) -> float:
    w_pred = ssvi_total_variance(k, theta_for_obs, rho=rho, eta=eta, gamma=gamma)
    mse = np.average((w_pred - w_true) ** 2, weights=weights)
    return float(mse + _constraint_penalty(rho=rho, eta=eta, gamma=gamma))


def _optimize_theta_nodes(
    rho: float,
    eta: float,
    gamma: float,
    uniq_t: ArrayLike,
    theta_nodes: ArrayLike,
    by_t_index: Iterable[ArrayLike],
    k: ArrayLike,
    w: ArrayLike,
) -> ArrayLike:
    new_theta = theta_nodes.copy()
    for i, idx in enumerate(by_t_index):
        theta0 = max(new_theta[i], 1e-8)
        grid = theta0 * np.geomspace(0.5, 1.5, 31)
        if theta0 < 1e-6:
            grid = np.geomspace(1e-8, 1e-2, 31)
        losses = []
        for th in grid:
            w_pred = ssvi_total_variance(k[idx], np.full(len(idx), th), rho=rho, eta=eta, gamma=gamma)
            losses.append(float(np.mean((w_pred - w[idx]) ** 2)))
        best = float(grid[int(np.argmin(losses))])
        new_theta[i] = max(best, 1e-8)

    # Enforce non-decreasing total variance term structure.
    new_theta = np.maximum.accumulate(new_theta)
    return new_theta


def _resolve_calibration_backend(calibration_backend: str) -> str:
    b = str(calibration_backend).lower()
    if b == "auto":
        if torch is not None and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if b not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported calibration backend '{calibration_backend}'. Use one of: auto, cpu, mps, cuda.")
    return b


def _calibrate_ssvi_surface_torch(
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
    t_key: ArrayLike,
    uniq_t: ArrayLike,
    by_t_index: List[ArrayLike],
    theta_nodes_base: ArrayLike,
    random_seed: int,
    n_param_steps: int,
    n_theta_steps: int,
    n_restarts: int,
    initial_params: Tuple[float, float, float] | None,
    theta_smoothness_lambda: float,
    device_name: str,
) -> Tuple[Tuple[float, float, float], ArrayLike]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch to use GPU/MPS calibration.")

    if device_name == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend requested but not available in this environment.")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but not available in this environment.")

    device = torch.device(device_name)
    # MPS commonly lacks full float64 kernel support; use float32 there.
    dtype = torch.float32 if device_name == "mps" else torch.float64

    k_torch = torch.as_tensor(k, dtype=dtype, device=device)
    t_torch = torch.as_tensor(t, dtype=dtype, device=device)
    sigma_torch = torch.as_tensor(sigma, dtype=dtype, device=device)
    w_torch = sigma_torch * sigma_torch * t_torch

    abs_k = np.abs(k)
    mny_w = 1.0 / (1.0 + (abs_k / 0.5) ** 2)
    weights = (1.0 / np.sqrt(np.maximum(t, 1e-4))) * mny_w
    weights = weights / np.mean(weights)
    weights_torch = torch.as_tensor(weights, dtype=dtype, device=device)
    weight_sum = torch.clamp(torch.sum(weights_torch), min=1e-12)

    t_to_i = {u: i for i, u in enumerate(uniq_t)}
    theta_obs_index = np.array([t_to_i[x] for x in t_key], dtype=np.int64)
    theta_obs_index_torch = torch.as_tensor(theta_obs_index, dtype=torch.long, device=device)

    rng = np.random.default_rng(random_seed)
    seeds: List[Tuple[float, float, float]] = [(-0.4, 0.5, 0.5), (-0.2, 0.8, 0.4), (-0.6, 0.35, 0.65)]
    if initial_params is not None:
        seeds = [tuple(initial_params)] + seeds
    while len(seeds) < max(1, n_restarts):
        seeds.append(
            (
                float(rng.uniform(-0.9, 0.1)),
                float(np.exp(rng.uniform(np.log(0.08), np.log(1.8)))),
                float(rng.uniform(0.08, 0.92)),
            )
        )

    def _loss_batch(rho_vec: np.ndarray, eta_vec: np.ndarray, gamma_vec: np.ndarray, theta_nodes_v: np.ndarray) -> np.ndarray:
        rho_t = torch.as_tensor(rho_vec, dtype=dtype, device=device).reshape(-1, 1)
        eta_t = torch.as_tensor(eta_vec, dtype=dtype, device=device).reshape(-1, 1)
        gamma_t = torch.as_tensor(gamma_vec, dtype=dtype, device=device).reshape(-1, 1)

        theta_nodes_t = torch.as_tensor(theta_nodes_v, dtype=dtype, device=device)
        theta_for_obs = torch.clamp(theta_nodes_t[theta_obs_index_torch], min=1e-12).reshape(1, -1)

        phi = eta_t * torch.pow(theta_for_obs, -gamma_t)
        z = phi * k_torch.reshape(1, -1)
        rad = torch.sqrt((z + rho_t) * (z + rho_t) + (1.0 - rho_t * rho_t))
        w_pred = 0.5 * theta_for_obs * (1.0 + rho_t * z + rad)

        diff2 = (w_pred - w_torch.reshape(1, -1)) ** 2
        mse = torch.sum(diff2 * weights_torch.reshape(1, -1), dim=1) / weight_sum

        penalty = 10.0 * torch.clamp(eta_t.reshape(-1) * (1.0 + torch.abs(rho_t.reshape(-1))) - 2.0, min=0.0) ** 2
        total = mse + penalty

        d = np.diff(theta_nodes_v)
        smooth_pen = theta_smoothness_lambda * float(np.mean(d * d)) if len(d) else 0.0
        return total.detach().cpu().numpy() + smooth_pen

    def _loss_scalar(rho_v: float, eta_v: float, gamma_v: float, theta_nodes_v: np.ndarray) -> float:
        return float(_loss_batch(np.asarray([rho_v]), np.asarray([eta_v]), np.asarray([gamma_v]), theta_nodes_v)[0])

    best_global_loss = np.inf
    best_global = (-0.4, 0.5, 0.5)
    best_theta = theta_nodes_base.copy()

    for r_i, seed in enumerate(seeds[: max(1, n_restarts)]):
        theta_nodes = theta_nodes_base.copy()
        best = seed
        best_loss = _loss_scalar(best[0], best[1], best[2], theta_nodes)

        step_rho = max(0.03, 0.08 / (1.0 + 0.5 * r_i))
        step_eta = max(0.08, 0.18 / (1.0 + 0.5 * r_i))
        step_gamma = max(0.03, 0.08 / (1.0 + 0.5 * r_i))

        for _ in range(n_theta_steps):
            rho_prop = np.clip(best[0] + rng.normal(0.0, step_rho, size=n_param_steps), -0.995, 0.995)
            eta_prop = np.clip(best[1] * np.exp(rng.normal(0.0, step_eta, size=n_param_steps)), 1e-4, 5.0)
            gamma_prop = np.clip(best[2] + rng.normal(0.0, step_gamma, size=n_param_steps), 0.01, 0.99)
            batch_losses = _loss_batch(rho_prop, eta_prop, gamma_prop, theta_nodes)
            best_i = int(np.argmin(batch_losses))
            if float(batch_losses[best_i]) < best_loss:
                best_loss = float(batch_losses[best_i])
                best = (float(rho_prop[best_i]), float(eta_prop[best_i]), float(gamma_prop[best_i]))

            for dr in [(-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, -0.05, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, -0.02), (0.0, 0.0, 0.02)]:
                rho_p = float(np.clip(best[0] + dr[0], -0.995, 0.995))
                eta_p = float(np.clip(best[1] * np.exp(dr[1]), 1e-4, 5.0))
                gamma_p = float(np.clip(best[2] + dr[2], 0.01, 0.99))
                loss = _loss_scalar(rho_p, eta_p, gamma_p, theta_nodes)
                if loss < best_loss:
                    best_loss = loss
                    best = (rho_p, eta_p, gamma_p)

            rho, eta, gamma = best
            theta_new = theta_nodes.copy()
            for i, idx in enumerate(by_t_index):
                theta0 = max(theta_new[i], 1e-8)
                grid = theta0 * np.geomspace(0.5, 1.5, 31)
                if theta0 < 1e-6:
                    grid = np.geomspace(1e-8, 1e-2, 31)

                idx_t = torch.as_tensor(np.asarray(idx, dtype=np.int64), dtype=torch.long, device=device)
                k_idx = k_torch[idx_t].reshape(1, -1)
                w_idx = w_torch[idx_t].reshape(1, -1)
                grid_t = torch.as_tensor(grid, dtype=dtype, device=device).reshape(-1, 1)

                phi = eta * torch.pow(torch.clamp(grid_t, min=1e-12), -gamma)
                z = phi * k_idx
                rad = torch.sqrt((z + rho) * (z + rho) + (1.0 - rho * rho))
                w_pred = 0.5 * grid_t * (1.0 + rho * z + rad)
                losses = torch.mean((w_pred - w_idx) ** 2, dim=1)
                theta_new[i] = float(grid[int(torch.argmin(losses).item())])

            theta_nodes = np.maximum.accumulate(np.maximum(theta_new, 1e-8))
            best_loss = _loss_scalar(rho, eta, gamma, theta_nodes)
            step_rho *= 0.85
            step_eta *= 0.88
            step_gamma *= 0.85

        if best_loss < best_global_loss:
            best_global_loss = best_loss
            best_global = best
            best_theta = theta_nodes.copy()

    return best_global, best_theta


def calibrate_ssvi_surface(
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
    random_seed: int = 42,
    n_param_steps: int = 400,
    n_theta_steps: int = 3,
    n_restarts: int = 4,
    initial_params: Tuple[float, float, float] | None = None,
    initial_theta_nodes: ArrayLike | None = None,
    theta_smoothness_lambda: float = 1e-3,
    calibration_backend: str = "mps",
) -> CalibratedSSVISurface:
    k = np.asarray(k, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    _validate_inputs(k, t, sigma)

    w = (sigma * sigma) * t
    t_key, uniq_t = _maturity_keys(t)
    if len(uniq_t) < 2:
        raise ValueError("Need at least 2 maturities for SSVI calibration.")

    by_t_index = [np.where(t_key == u)[0] for u in uniq_t]
    if initial_theta_nodes is None:
        theta_nodes_base = _initial_theta_from_atm(k=k, t_key=t_key, w=w, uniq_t=uniq_t)
    else:
        arr = np.asarray(initial_theta_nodes, dtype=np.float64).reshape(-1)
        if len(arr) == len(uniq_t):
            theta_nodes_base = np.maximum.accumulate(np.maximum(arr, 1e-8))
        else:
            theta_nodes_base = _initial_theta_from_atm(k=k, t_key=t_key, w=w, uniq_t=uniq_t)

    t_to_i = {u: i for i, u in enumerate(uniq_t)}
    resolved_backend = _resolve_calibration_backend(calibration_backend)

    if resolved_backend in {"mps", "cuda"}:
        best_global, best_theta = _calibrate_ssvi_surface_torch(
            k=k,
            t=t,
            sigma=sigma,
            t_key=t_key,
            uniq_t=uniq_t,
            by_t_index=by_t_index,
            theta_nodes_base=theta_nodes_base,
            random_seed=random_seed,
            n_param_steps=n_param_steps,
            n_theta_steps=n_theta_steps,
            n_restarts=n_restarts,
            initial_params=initial_params,
            theta_smoothness_lambda=theta_smoothness_lambda,
            device_name=resolved_backend,
        )
    else:
        # robust-ish weighting: prefer liquid short maturities but avoid extreme domination
        abs_k = np.abs(k)
        mny_w = 1.0 / (1.0 + (abs_k / 0.5) ** 2)
        weights = (1.0 / np.sqrt(np.maximum(t, 1e-4))) * mny_w
        weights = weights / np.mean(weights)

        rng = np.random.default_rng(random_seed)
        seeds: List[Tuple[float, float, float]] = [(-0.4, 0.5, 0.5), (-0.2, 0.8, 0.4), (-0.6, 0.35, 0.65)]
        if initial_params is not None:
            seeds = [tuple(initial_params)] + seeds
        while len(seeds) < max(1, n_restarts):
            seeds.append(
                (
                    float(rng.uniform(-0.9, 0.1)),
                    float(np.exp(rng.uniform(np.log(0.08), np.log(1.8)))),
                    float(rng.uniform(0.08, 0.92)),
                )
            )

        best_global_loss = np.inf
        best_global = (-0.4, 0.5, 0.5)
        best_theta = theta_nodes_base.copy()

        def loss_with_theta_smooth(rho_v: float, eta_v: float, gamma_v: float, theta_nodes_v: np.ndarray) -> float:
            theta_for_obs_v = np.array([theta_nodes_v[t_to_i[x]] for x in t_key], dtype=np.float64)
            base = _loss_for_params(rho_v, eta_v, gamma_v, k, theta_for_obs_v, w, weights)
            d = np.diff(theta_nodes_v)
            smooth_pen = theta_smoothness_lambda * float(np.mean(d * d)) if len(d) else 0.0
            return base + smooth_pen

        for r_i, seed in enumerate(seeds[: max(1, n_restarts)]):
            rho, eta, gamma = seed
            theta_nodes = theta_nodes_base.copy()
            best_loss = loss_with_theta_smooth(rho, eta, gamma, theta_nodes)
            best = (rho, eta, gamma)

            step_rho = max(0.03, 0.08 / (1.0 + 0.5 * r_i))
            step_eta = max(0.08, 0.18 / (1.0 + 0.5 * r_i))
            step_gamma = max(0.03, 0.08 / (1.0 + 0.5 * r_i))

            for _ in range(n_theta_steps):
                # Multi-try random local search
                for _ in range(n_param_steps):
                    rho_p = float(np.clip(best[0] + rng.normal(0.0, step_rho), -0.995, 0.995))
                    eta_p = float(np.clip(best[1] * np.exp(rng.normal(0.0, step_eta)), 1e-4, 5.0))
                    gamma_p = float(np.clip(best[2] + rng.normal(0.0, step_gamma), 0.01, 0.99))
                    loss = loss_with_theta_smooth(rho_p, eta_p, gamma_p, theta_nodes)
                    if loss < best_loss:
                        best_loss = loss
                        best = (rho_p, eta_p, gamma_p)

                # Coordinate-local refinement around incumbent
                for dr in [(-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, -0.05, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, -0.02), (0.0, 0.0, 0.02)]:
                    rho_p = float(np.clip(best[0] + dr[0], -0.995, 0.995))
                    eta_p = float(np.clip(best[1] * np.exp(dr[1]), 1e-4, 5.0))
                    gamma_p = float(np.clip(best[2] + dr[2], 0.01, 0.99))
                    loss = loss_with_theta_smooth(rho_p, eta_p, gamma_p, theta_nodes)
                    if loss < best_loss:
                        best_loss = loss
                        best = (rho_p, eta_p, gamma_p)

                rho, eta, gamma = best
                theta_nodes = _optimize_theta_nodes(
                    rho=rho,
                    eta=eta,
                    gamma=gamma,
                    uniq_t=uniq_t,
                    theta_nodes=theta_nodes,
                    by_t_index=by_t_index,
                    k=k,
                    w=w,
                )
                best_loss = loss_with_theta_smooth(rho, eta, gamma, theta_nodes)
                step_rho *= 0.85
                step_eta *= 0.88
                step_gamma *= 0.85

            if best_loss < best_global_loss:
                best_global_loss = best_loss
                best_global = (rho, eta, gamma)
                best_theta = theta_nodes.copy()

    theta_for_obs = np.array([best_theta[t_to_i[x]] for x in t_key], dtype=np.float64)
    w_fit = ssvi_total_variance(k, theta_for_obs, rho=best_global[0], eta=best_global[1], gamma=best_global[2])
    sigma_fit = np.sqrt(np.maximum(w_fit, 1e-12) / np.maximum(t, 1e-12))
    rmse_w = float(np.sqrt(np.mean((w_fit - w) ** 2)))
    rmse_sigma = float(np.sqrt(np.mean((sigma_fit - sigma) ** 2)))
    no_bfly_score = float(max(0.0, 2.0 - best_global[1] * (1.0 + abs(best_global[0]))))

    return CalibratedSSVISurface(
        params=SSVIParameters(rho=float(best_global[0]), eta=float(best_global[1]), gamma=float(best_global[2])),
        t_nodes=uniq_t.astype(np.float64),
        theta_nodes=best_theta.astype(np.float64),
        diagnostics=SSVIDiagnostics(
            rmse_total_variance=rmse_w,
            rmse_implied_volatility=rmse_sigma,
            no_butterfly_constraint_score=no_bfly_score,
            n_points=int(len(k)),
            n_maturities=int(len(uniq_t)),
        ),
    )
