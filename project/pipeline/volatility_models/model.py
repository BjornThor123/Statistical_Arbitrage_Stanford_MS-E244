from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

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


# Deterministic starting points covering the admissible parameter space.
_PARAM_SEEDS: List[Tuple[float, float, float]] = [
    (-0.4, 0.50, 0.50),
    (-0.2, 0.80, 0.40),
    (-0.6, 0.35, 0.65),
    (-0.1, 1.00, 0.30),
    (-0.7, 0.25, 0.70),
    ( 0.0, 0.60, 0.50),
    (-0.5, 0.70, 0.45),
    (-0.3, 0.45, 0.55),
]


def _loss_and_grad(
    x: np.ndarray,
    k: ArrayLike,
    theta_for_obs: ArrayLike,
    w: ArrayLike,
    weights: ArrayLike,
    weight_sum: float,
) -> Tuple[float, np.ndarray]:
    """MSE loss + butterfly penalty, with exact gradient w.r.t. x = [rho, log_eta, gamma]."""
    rho, log_eta, gamma = float(x[0]), float(x[1]), float(x[2])
    eta = np.exp(log_eta)

    safe_theta = np.maximum(theta_for_obs, 1e-12)
    phi = eta * np.power(safe_theta, -gamma)
    z = phi * k
    zr = z + rho
    r = np.maximum(np.sqrt(zr * zr + (1.0 - rho * rho)), 1e-12)

    w_pred = 0.5 * safe_theta * (1.0 + rho * z + r)
    residual = w_pred - w
    wr = weights * residual
    mse = float(np.dot(wr, residual) / weight_sum)

    # Shared sub-expression: ρ + (z+ρ)/r
    rho_plus_zrr = rho + zr / r

    # dw/dρ = ½θ·z·(1 + 1/r)
    dw_drho = 0.5 * safe_theta * z * (1.0 + 1.0 / r)
    # dw/d(log η) = ½θ·z·(ρ + (z+ρ)/r)   [chain rule: d/d(log η) = η·d/dη]
    dw_dlogeta = 0.5 * safe_theta * z * rho_plus_zrr
    # dw/dγ = −½θ·z·ln(θ)·(ρ + (z+ρ)/r)
    dw_dgamma = -0.5 * safe_theta * z * np.log(safe_theta) * rho_plus_zrr

    scale = 2.0 / weight_sum
    grad_mse = np.array([
        scale * float(np.dot(wr, dw_drho)),
        scale * float(np.dot(wr, dw_dlogeta)),
        scale * float(np.dot(wr, dw_dgamma)),
    ])

    # Butterfly penalty and its gradient.
    v = eta * (1.0 + abs(rho)) - 2.0
    if v > 0.0:
        penalty = 10.0 * v * v
        grad_penalty = np.array([
            20.0 * v * eta * np.sign(rho),
            20.0 * v * eta * (1.0 + abs(rho)),  # d/d(log η) = η · d/dη
            0.0,
        ])
    else:
        penalty = 0.0
        grad_penalty = np.zeros(3)

    return mse + penalty, grad_mse + grad_penalty


def _optimize_global_params(
    k: ArrayLike,
    theta_for_obs: ArrayLike,
    w: ArrayLike,
    weights: ArrayLike,
    seeds: List[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Fit (rho, eta, gamma) via L-BFGS-B with analytical gradient.

    eta is log-transformed so the optimiser works with simple box bounds.
    """
    weight_sum = float(np.sum(weights))
    args = (k, theta_for_obs, w, weights, weight_sum)
    bounds = [(-0.995, 0.995), (np.log(1e-4), np.log(5.0)), (0.01, 0.99)]

    best_loss = np.inf
    best_params: Tuple[float, float, float] = seeds[0]

    for rho0, eta0, gamma0 in seeds:
        x0 = np.array([rho0, np.log(eta0), gamma0])
        result = minimize(_loss_and_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds, args=args, options={"maxiter": 2000, "ftol": 1e-14})
        if result.fun < best_loss:
            best_loss = result.fun
            rho, log_eta, gamma = result.x
            best_params = (float(rho), float(np.exp(log_eta)), float(gamma))

    return best_params


def _optimize_theta_nodes(
    rho: float,
    eta: float,
    gamma: float,
    theta_nodes: ArrayLike,
    by_t_index: Iterable[ArrayLike],
    k: ArrayLike,
    w: ArrayLike,
) -> ArrayLike:
    """Fit each theta_i independently via bounded scalar minimisation, then enforce monotonicity."""
    new_theta = theta_nodes.copy()
    for i, idx in enumerate(by_t_index):
        k_i, w_i = k[idx], w[idx]
        theta0 = max(new_theta[i], 1e-8)

        def obj(th: float, k_i: ArrayLike = k_i, w_i: ArrayLike = w_i) -> float:
            w_pred = ssvi_total_variance(k_i, np.full(len(k_i), th), rho=rho, eta=eta, gamma=gamma)
            return float(np.mean((w_pred - w_i) ** 2))

        result = minimize_scalar(obj, bounds=(max(theta0 * 0.05, 1e-8), theta0 * 20.0), method="bounded")
        new_theta[i] = max(float(result.x), 1e-8)

    return np.maximum.accumulate(new_theta)


def _make_seeds(
    n_restarts: int,
    initial_params: Tuple[float, float, float] | None,
    rng: np.random.Generator,
) -> List[Tuple[float, float, float]]:
    """Generate random starting points for the optimization restarts."""
    seeds: List[Tuple[float, float, float]] = [(-0.4, 0.5, 0.5), (-0.2, 0.8, 0.4), (-0.6, 0.35, 0.65)]
    if initial_params is not None:
        seeds = [tuple(initial_params)] + seeds
    while len(seeds) < max(1, n_restarts):
        seeds.append((
            float(rng.uniform(-0.9, 0.1)),
            float(np.exp(rng.uniform(np.log(0.08), np.log(1.8)))),
            float(rng.uniform(0.08, 0.92)),
        ))
    return seeds


def _compute_calibration_diagnostics(
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
    rho: float,
    eta: float,
    gamma: float,
    theta_nodes: ArrayLike,
    t_to_i: dict,
    t_key: ArrayLike,
) -> SSVIDiagnostics:
    """Compute fit quality diagnostics after calibration."""
    theta_for_obs = np.array([theta_nodes[t_to_i[x]] for x in t_key], dtype=np.float64)
    w = sigma * sigma * t
    w_fit = ssvi_total_variance(k, theta_for_obs, rho=rho, eta=eta, gamma=gamma)
    sigma_fit = np.sqrt(np.maximum(w_fit, 1e-12) / np.maximum(t, 1e-12))
    rmse_w = float(np.sqrt(np.mean((w_fit - w) ** 2)))
    rmse_sigma = float(np.sqrt(np.mean((sigma_fit - sigma) ** 2)))
    no_bfly_score = float(max(0.0, 2.0 - eta * (1.0 + abs(rho))))
    return SSVIDiagnostics(
        rmse_total_variance=rmse_w,
        rmse_implied_volatility=rmse_sigma,
        no_butterfly_constraint_score=no_bfly_score,
        n_points=int(len(k)),
        n_maturities=int(len(np.unique(t_key))),
    )


def calibrate_ssvi_surface(
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
    n_theta_steps: int = 3,
    n_restarts: int = 4,
    initial_params: Tuple[float, float, float] | None = None,
    initial_theta_nodes: ArrayLike | None = None,
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
    t_to_i = {u: i for i, u in enumerate(uniq_t)}

    if initial_theta_nodes is None:
        theta_nodes_base = _initial_theta_from_atm(k=k, t_key=t_key, w=w, uniq_t=uniq_t)
    else:
        arr = np.asarray(initial_theta_nodes, dtype=np.float64).reshape(-1)
        if len(arr) == len(uniq_t):
            theta_nodes_base = np.maximum.accumulate(np.maximum(arr, 1e-8))
        else:
            theta_nodes_base = _initial_theta_from_atm(k=k, t_key=t_key, w=w, uniq_t=uniq_t)

    abs_k = np.abs(k)
    mny_w = 1.0 / (1.0 + (abs_k / 0.5) ** 2)
    weights = (1.0 / np.sqrt(np.maximum(t, 1e-4))) * mny_w
    weights = weights / np.mean(weights)

    rng = np.random.default_rng(seed=42)
    outer_seeds = _make_seeds(n_restarts, initial_params, rng)

    best_global_loss = np.inf
    best_global: Tuple[float, float, float] = outer_seeds[0]
    best_theta = theta_nodes_base.copy()

    for seed in outer_seeds[: max(1, n_restarts)]:
        params = seed
        theta_nodes = theta_nodes_base.copy()

        for _ in range(n_theta_steps):
            theta_for_obs = np.array([theta_nodes[t_to_i[x]] for x in t_key], dtype=np.float64)
            # Single L-BFGS-B run from current iterate (BCD global-param step).
            params = _optimize_global_params(
                k=k,
                theta_for_obs=theta_for_obs,
                w=w,
                weights=weights,
                seeds=[params],
            )
            theta_nodes = _optimize_theta_nodes(
                rho=params[0],
                eta=params[1],
                gamma=params[2],
                theta_nodes=theta_nodes,
                by_t_index=by_t_index,
                k=k,
                w=w,
            )

        theta_for_obs = np.array([theta_nodes[t_to_i[x]] for x in t_key], dtype=np.float64)
        w_pred = ssvi_total_variance(k, theta_for_obs, rho=params[0], eta=params[1], gamma=params[2])
        loss = float(np.mean((w_pred - w) ** 2))
        if loss < best_global_loss:
            best_global_loss = loss
            best_global = params
            best_theta = theta_nodes.copy()

    diagnostics = _compute_calibration_diagnostics(
        k=k,
        t=t,
        sigma=sigma,
        rho=float(best_global[0]),
        eta=float(best_global[1]),
        gamma=float(best_global[2]),
        theta_nodes=best_theta,
        t_to_i=t_to_i,
        t_key=t_key,
    )

    return CalibratedSSVISurface(
        params=SSVIParameters(rho=float(best_global[0]), eta=float(best_global[1]), gamma=float(best_global[2])),
        t_nodes=uniq_t.astype(np.float64),
        theta_nodes=best_theta.astype(np.float64),
        diagnostics=diagnostics,
    )
