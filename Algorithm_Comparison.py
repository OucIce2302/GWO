# -*- coding: utf-8 -*-
# Multi-UAV spraying simulator with vanilla GWO (baseline) and improved C-GWO (OBL + cooperative)
# Author: Your teammate :)
#
# Vanilla GWO = unchanged control (as in your original).
# Improved C-GWO = opposition-based initialization + cooperative round-robin update (+ optional stagnation reset).
#
# Requirements: numpy, matplotlib, pandas
# Optional: scipy (for fast maximum_filter). If absent, a numpy fallback is used automatically.

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

# -------------------- RNG (fixed seed for reproducibility) --------------------
rng = np.random.default_rng(0)

# -------------------- Domain & Masks --------------------

def _binary_dilate_numpy(B: np.ndarray, radius_cells: int) -> np.ndarray:
    """
    SciPy-free fallback for binary dilation with a square (2r+1)x(2r+1) kernel.
    B: bool array (H, W)
    """
    if radius_cells <= 0:
        return B.copy()
    k = 2 * radius_cells + 1
    pad = radius_cells
    P = np.pad(B.astype(np.uint8), pad, mode='edge')
    H, W = B.shape
    out = np.zeros_like(B, dtype=np.uint8)
    # naive but fine for ~120x180
    for i in range(H):
        ii = i
        for j in range(W):
            jj = j
            block = P[ii:ii + k, jj:jj + k]
            out[i, j] = 1 if block.max() > 0 else 0
    return out.astype(bool)

def make_domain(nx=180, ny=120, dx=5.0):
    """
    Create a rectangular domain grid.
    Returns: X, Y, dx, masks with 'Z' (spray area), 'B' (no-spray), 'Bbuf' (buffered no-spray)
    """
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y, indexing='xy')

    # spray area Z init as all True, then carve out B
    Z = np.ones((ny, nx), dtype=bool)

    # No-spray rectangles (schools/hospitals analogs)
    def rect_mask(x0, y0, w, h):
        return (X >= x0) & (X <= x0 + w) & (Y >= y0) & (Y <= y0 + h)

    B = np.zeros_like(Z)
    B |= rect_mask(350, 250, 120, 120)
    B |= rect_mask(700, 150, 120, 120)

    # buffer band d (meters)
    d = 30.0
    buf_cells = int(math.ceil(d / dx))

    # Try SciPy maximum_filter, else fallback
    try:
        from scipy.ndimage import maximum_filter
        Bbuf = maximum_filter(B.astype(np.uint8), size=(2 * buf_cells + 1, 2 * buf_cells + 1)) > 0
    except Exception:
        Bbuf = _binary_dilate_numpy(B, buf_cells)

    # spray area excludes only hard B (buffer is used for overspray metric)
    Z = (~B)

    masks = {'Z': Z, 'B': B, 'Bbuf': Bbuf}
    return X, Y, dx, masks


# -------------------- Wind & Kernel --------------------

def kernel_patch(sx, sy, angle_rad, radius_mult=3.0):
    """
    Rotated anisotropic Gaussian kernel with stds sx, sy along rotated axes.
    Returns K (2D) and radii rx, ry in grid cells.
    """
    rx = max(3, int(radius_mult * sx))
    ry = max(3, int(radius_mult * sy))
    xs = np.arange(-rx, rx + 1)
    ys = np.arange(-ry, ry + 1)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')

    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    Xr = ca * XX + sa * YY
    Yr = -sa * XX + ca * YY

    K = np.exp(-0.5 * ((Xr / sx) ** 2 + (Yr / sy) ** 2))
    K = K / (K.sum() + 1e-9)
    return K, rx, ry


def sigma_params(w_speed, h, sigma0=3.0, alpha_w=2.0, alpha_h=0.05, beta_h=0.02):
    """
    sigma_parallel grows with wind speed and altitude; sigma_perp grows mildly with altitude.
    """
    sigma_parallel = sigma0 + alpha_w * abs(w_speed) + alpha_h * h
    sigma_perp = sigma0 + beta_h * h
    return sigma_parallel, sigma_perp


# -------------------- Paths & Deposition --------------------

def clip_to_domain(pt, nx, ny, dx):
    x = min(max(pt[0], 0.0), (nx - 1) * dx)
    y = min(max(pt[1], 0.0), (ny - 1) * dx)
    return np.array([x, y], dtype=float)

def raster_index(pt, dx):
    return int(round(pt[0] / dx)), int(round(pt[1] / dx))

def deposit_paths(X, Y, dx, masks, waypoints_per_uav: List[np.ndarray],
                  wind_speed=2.0, wind_dir_deg=45.0, tau_half_min=40.0,
                  dt_s=10.0, step_m=10.0, altitude_m=40.0,
                  spray_on: List[np.ndarray]=None, dose_scale=0.5):
    """
    Simulate deposition for multiple UAV paths described by waypoints.
    Each UAV moves along polyline; deposition kernel follows wind-oriented anisotropic Gaussian + time decay.
    """
    ny, nx = X.shape
    dep = np.zeros((ny, nx), dtype=float)

    angle = math.radians(wind_dir_deg)
    sigma_par, sigma_perp = sigma_params(wind_speed, altitude_m)
    K, rx, ry = kernel_patch(sigma_par / dx, sigma_perp / dx, angle)

    lam = math.log(2.0) / (tau_half_min * 60.0)  # per-second decay
    if spray_on is None:
        spray_on = [np.ones(len(wp) - 1, dtype=bool) for wp in waypoints_per_uav]

    for idx_u, wps in enumerate(waypoints_per_uav):
        if len(wps) < 2:
            continue
        t_u = 0.0
        for s in range(len(wps) - 1):
            P = wps[s].astype(float).copy()
            Q = wps[s + 1].astype(float).copy()
            seg = Q - P
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue
            nsteps = max(1, int(seg_len / step_m))
            direction = seg / seg_len
            for k in range(nsteps + 1):
                pos = P + direction * (k * step_m)
                pos = clip_to_domain(pos, nx, ny, dx)
                xi, yi = raster_index(pos, dx)
                xi = min(max(xi, 0), nx - 1)
                yi = min(max(yi, 0), ny - 1)

                if not masks['B'][yi, xi]:  # forbid spray in hard B
                    if spray_on[idx_u][s]:
                        decay = math.exp(-lam * t_u)
                        y0, y1 = yi - ry, yi + ry + 1
                        x0, x1 = xi - rx, xi + rx + 1
                        ky0, ky1 = 0, K.shape[0]
                        kx0, kx1 = 0, K.shape[1]
                        if y0 < 0:   ky0 += -y0; y0 = 0
                        if x0 < 0:   kx0 += -x0; x0 = 0
                        if y1 > ny: ky1 -= (y1 - ny); y1 = ny
                        if x1 > nx: kx1 -= (x1 - nx); x1 = nx
                        dep[y0:y1, x0:x1] += dose_scale * decay * K[ky0:ky1, kx0:kx1]
                t_u += dt_s
    return dep, None


# -------------------- Metrics --------------------

def compute_metrics(dep, masks, theta=0.5):
    Z = masks['Z']
    Bbuf = masks['Bbuf']
    Z_area = max(1, Z.sum())
    cov = (dep[Z] >= theta).sum() / Z_area
    vals = dep[Z]
    mean = vals.mean()
    std = vals.std()
    cv = std / (mean + 1e-12)
    uni = 1.0 / (1.0 + cv)             # (0,1], higher is more uniform
    over = dep[Bbuf & (~Z)].sum() / (dep.sum() + 1e-12)  # fraction of mass fallen on buffer-only
    return cov, uni, over

def auto_theta(dep, Z_mask, q=0.60):
    vals = dep[Z_mask]
    nz = vals[vals > 0]
    return float(np.quantile(nz, q)) if nz.size else 0.0


# -------------------- Baseline: Lawnmower --------------------

def lawnmower_paths(masks, dx, lane_spacing=30.0, start_side='left'):
    """
    Generate serpentine (lawnmower) lanes covering bbox(Z). Split lanes across 3 UAVs round-robin.
    """
    Z = masks['Z']
    ny, nx = Z.shape
    ys, xs = np.where(Z)
    if len(xs) == 0:
        return [], []
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    xmin_m, xmax_m = xmin * dx, xmax * dx
    ymin_m, ymax_m = ymin * dx, ymax * dx

    spacing = max(dx, lane_spacing)
    lanes = []
    y = ymin_m
    ltr = (start_side == 'left')
    while y <= ymax_m:
        if ltr:
            lanes.append(np.array([[xmin_m, y], [xmax_m, y]], dtype=float))
        else:
            lanes.append(np.array([[xmax_m, y], [xmin_m, y]], dtype=float))
        y += spacing
        ltr = not ltr

    m = 3
    paths = [[lanes[i][0]] for i in range(min(m, len(lanes)))]
    for idx, seg in enumerate(lanes):
        u = idx % m
        if len(paths) <= u:
            paths.append([seg[0]])
        paths[u].append(seg[1])
    waypoints_per_uav = [np.array(wps) for wps in paths]
    spray = [np.ones(len(wps) - 1, dtype=bool) for wps in waypoints_per_uav]
    return waypoints_per_uav, spray


# -------------------- Utility --------------------

def path_energy(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

def population_energy(pop) -> np.ndarray:
    n_pop, m_uav, n_wp, _ = pop.shape
    en = np.zeros(n_pop, dtype=float)
    for i in range(n_pop):
        e = 0.0
        for u in range(m_uav):
            e += path_energy(pop[i, u, :, :])
        en[i] = e
    return en


# -------------------- Vanilla GWO (UNCHANGED CONTROL) --------------------

def init_population(n_pop, m_uav=3, n_wp=6, bounds=((0, 900), (0, 600))):
    (xmin, xmax), (ymin, ymax) = bounds
    pop = rng.uniform([xmin, ymin], [xmax, ymax], size=(n_pop, m_uav, n_wp, 2))
    return pop

def repair(pop, masks, dx):
    ny, nx = masks['Z'].shape
    pop[..., 0] = np.clip(pop[..., 0], 0, (nx - 1) * dx)
    pop[..., 1] = np.clip(pop[..., 1], 0, (ny - 1) * dx)
    return pop

def objective_from_pop(pop, X, Y, dx, masks, wind_speed, wind_dir_deg,
                       tau_half_min, dt_s, step_m, altitude, theta,
                       energy_weight=0.001):
    n_pop = pop.shape[0]
    scores = np.zeros(n_pop)
    cov_uni_over_e = []
    for i in range(n_pop):
        waypoints = [pop[i, u, :, :] for u in range(pop.shape[1])]
        dep, _ = deposit_paths(X, Y, dx, masks, waypoints, wind_speed, wind_dir_deg,
                               tau_half_min, dt_s, step_m, altitude)
        cov, uni, over = compute_metrics(dep, masks, theta)
        # energy proxy across UAVs
        energy = 0.0
        for u in range(pop.shape[1]):
            energy += path_energy(pop[i, u, :, :])
        score = cov + 0.5 * uni - (over + energy_weight * energy / 10000.0)
        scores[i] = score
        cov_uni_over_e.append((cov, uni, over, energy))
    return scores, cov_uni_over_e

def gwo_optimize(X, Y, dx, masks, n_iter=25, n_pop=20, m_uav=3, n_wp=6,
                 wind_speed=2.0, wind_dir_deg=45.0, tau_half_min=40.0,
                 dt_s=10.0, step_m=10.0, altitude=40.0, theta=0.5,
                 cooperative=False, bounds=((0, 900), (0, 600))):
    (xmin, xmax), (ymin, ymax) = bounds
    pop = init_population(n_pop, m_uav, n_wp, bounds)
    pop = repair(pop, masks, dx)

    best_hist = []
    best_solution = None
    best_score = -1e9

    for it in range(n_iter):
        a = 2 - 2 * it / (n_iter - 1 if n_iter > 1 else 1)  # linear decrease
        scores, _ = objective_from_pop(pop, X, Y, dx, masks, wind_speed, wind_dir_deg,
                                       tau_half_min, dt_s, step_m, altitude, theta)
        idx_sorted = np.argsort(-scores)
        alpha = pop[idx_sorted[0]].copy()
        beta  = pop[idx_sorted[1]].copy() if n_pop > 1 else alpha.copy()
        delta = pop[idx_sorted[2]].copy() if n_pop > 2 else alpha.copy()
        alpha_score = scores[idx_sorted[0]]
        best_hist.append(alpha_score)
        if alpha_score > best_score:
            best_score = alpha_score
            best_solution = alpha.copy()

        # Update (standard GWO); if cooperative=False, all UAVs update together
        if cooperative:
            group = it % m_uav  # only one UAV group per iteration
        for i in range(n_pop):
            for u in range(m_uav):
                if cooperative and u != group:
                    continue
                for j in range(n_wp):
                    Xij, Yij = pop[i, u, j, 0], pop[i, u, j, 1]
                    for target in [alpha[u, j], beta[u, j], delta[u, j]]:
                        A = 2 * a * rng.random(2) - a
                        C = 2 * rng.random(2)
                        D = np.abs(C * target - np.array([Xij, Yij]))
                        Xij, Yij = np.array([Xij, Yij]) + (-A * D)
                    pop[i, u, j, 0] = Xij
                    pop[i, u, j, 1] = Yij
        pop = repair(pop, masks, dx)

    # Final metrics for best solution
    dep, _ = deposit_paths(X, Y, dx, masks, [best_solution[u, :, :] for u in range(m_uav)],
                           wind_speed, wind_dir_deg, tau_half_min, dt_s, step_m, altitude)
    cov, uni, over = compute_metrics(dep, masks, theta)
    energy = 0.0
    for u in range(m_uav):
        energy += path_energy(best_solution[u, :, :])
    return {
        'best_score': best_score,
        'cov': cov, 'uni': uni, 'over': over, 'energy': energy,
        'dep': dep, 'solution': best_solution, 'hist': best_hist
    }


# -------------------- Improved C-GWO (OBL + cooperative + optional reset) --------------------

def opposition_points(pop, bounds):
    """Compute opposite positions x' = L + U - x for all coordinates."""
    (xmin, xmax), (ymin, ymax) = bounds
    opp = np.empty_like(pop)
    opp[..., 0] = xmin + xmax - pop[..., 0]
    opp[..., 1] = ymin + ymax - pop[..., 1]
    return opp

def init_population_OBL(n_pop, m_uav, n_wp, bounds,
                        X, Y, dx, masks,
                        wind_speed, wind_dir_deg, tau_half_min, dt_s, step_m, altitude, theta):
    """
    OBL initialization:
    - Random n_pop
    - Generate their opposites
    - Evaluate 2n_pop candidates, keep the top n_pop
    """
    pop_rand = rng.uniform([bounds[0][0], bounds[1][0]],
                           [bounds[0][1], bounds[1][1]],
                           size=(n_pop, m_uav, n_wp, 2))
    pop_opp = opposition_points(pop_rand, bounds)
    cand = np.concatenate([pop_rand, pop_opp], axis=0)
    cand = repair(cand, masks, dx)
    scores, _ = objective_from_pop(cand, X, Y, dx, masks, wind_speed, wind_dir_deg,
                                   tau_half_min, dt_s, step_m, altitude, theta)
    idx = np.argsort(-scores)[:n_pop]
    return cand[idx]

def cgwo_optimize_OBL(X, Y, dx, masks, n_iter=120, n_pop=20, m_uav=3, n_wp=6,
                      wind_speed=2.0, wind_dir_deg=45.0, tau_half_min=40.0,
                      dt_s=10.0, step_m=10.0, altitude=40.0, theta=0.5,
                      bounds=((0, 900), (0, 600)),
                      energy_weight=0.001,
                      stagnation_patience=40, reset_frac=0.30):
    """
    Cooperative GWO with OBL initialization and round-robin UAV updates.
    Optional stagnation reset: if no improvement for `stagnation_patience` iterations,
    reinitialize a fraction of worst individuals around the current alpha/opposites.
    """
    (xmin, xmax), (ymin, ymax) = bounds

    # OBL initialization
    pop = init_population_OBL(n_pop, m_uav, n_wp, bounds,
                              X, Y, dx, masks,
                              wind_speed, wind_dir_deg, tau_half_min, dt_s, step_m, altitude, theta)
    pop = repair(pop, masks, dx)

    best_hist, best_solution, best_score = [], None, -1e9
    no_improve = 0

    for it in range(n_iter):
        a = 2 - 2 * it / (n_iter - 1 if n_iter > 1 else 1)
        scores, _ = objective_from_pop(pop, X, Y, dx, masks, wind_speed, wind_dir_deg,
                                       tau_half_min, dt_s, step_m, altitude, theta,
                                       energy_weight=energy_weight)
        idx_sorted = np.argsort(-scores)
        alpha = pop[idx_sorted[0]].copy()
        beta  = pop[idx_sorted[1]].copy() if n_pop > 1 else alpha.copy()
        delta = pop[idx_sorted[2]].copy() if n_pop > 2 else alpha.copy()
        alpha_score = scores[idx_sorted[0]]
        best_hist.append(alpha_score)

        if alpha_score > best_score + 1e-12:
            best_score = alpha_score
            best_solution = alpha.copy()
            no_improve = 0
        else:
            no_improve += 1

        # Cooperative round-robin: update one UAV group per iteration
        group = it % m_uav
        for i in range(n_pop):
            for u in range(m_uav):
                if u != group:
                    continue
                for j in range(n_wp):
                    Xij, Yij = pop[i, u, j, 0], pop[i, u, j, 1]
                    # Pull towards alpha, beta, delta
                    for target in [alpha[u, j], beta[u, j], delta[u, j]]:
                        A = 2 * a * rng.random(2) - a
                        C = 2 * rng.random(2)
                        D = np.abs(C * target - np.array([Xij, Yij]))
                        Xij, Yij = np.array([Xij, Yij]) + (-A * D)
                    pop[i, u, j, 0], pop[i, u, j, 1] = Xij, Yij

        pop = repair(pop, masks, dx)

        # Optional: stagnation reset to escape local minima
        if stagnation_patience and no_improve >= stagnation_patience:
            no_improve = 0
            num_reset = max(1, int(n_pop * reset_frac))
            worst_idx = idx_sorted[-num_reset:]
            # re-seed these around opposite of alpha with small Gaussian noise
            opp_alpha = opposition_points(alpha[np.newaxis, ...], bounds)[0]
            for wi in worst_idx:
                base = opp_alpha.copy() if rng.random() < 0.5 else alpha.copy()
                noise = rng.normal(loc=0.0, scale=0.05, size=base.shape)  # 5% domain magnitude (relative)
                # scale noise by domain range in meters
                xrange = (xmax - xmin)
                yrange = (ymax - ymin)
                base[..., 0] = np.clip(base[..., 0] + noise[..., 0] * xrange, xmin, xmax)
                base[..., 1] = np.clip(base[..., 1] + noise[..., 1] * yrange, ymin, ymax)
                pop[wi] = base
            pop = repair(pop, masks, dx)

    # Final metrics for best solution
    dep, _ = deposit_paths(X, Y, dx, masks, [best_solution[u, :, :] for u in range(m_uav)],
                           wind_speed, wind_dir_deg, tau_half_min, dt_s, step_m, altitude)
    cov, uni, over = compute_metrics(dep, masks, theta)
    energy = 0.0
    for u in range(m_uav):
        energy += path_energy(best_solution[u, :, :])
    return {
        'best_score': best_score,
        'cov': cov, 'uni': uni, 'over': over, 'energy': energy,
        'dep': dep, 'solution': best_solution, 'hist': best_hist
    }


# -------------------- Demo / Experiment --------------------

if __name__ == "__main__":
    # Build domain/masks
    X, Y, dx, masks = make_domain(nx=180, ny=120, dx=5.0)

    # Scene wind
    wind_speed = 2.0
    wind_dir_deg = 45.0
    tau_half = 40.0

    # Lawn mower baseline
    lm_paths, lm_spray = lawnmower_paths(masks, dx, lane_spacing=30.0, start_side='left')
    dep_lm, _ = deposit_paths(
        X, Y, dx, masks, lm_paths,
        wind_speed, wind_dir_deg, tau_half,
        dt_s=10, step_m=5, altitude_m=40.0,
        spray_on=lm_spray, dose_scale=0.5
    )

    # Use 50% quantile of non-zero lawnmower deposition over Z as threshold
    theta = auto_theta(dep_lm, masks['Z'], q=0.50)
    cov_lm, uni_lm, over_lm = compute_metrics(dep_lm, masks, theta)
    energy_lm = 0.0
    for u in range(len(lm_paths)):
        energy_lm += path_energy(lm_paths[u])

    # Vanilla GWO (unchanged baseline)
    res_gwo = gwo_optimize(X, Y, dx, masks,
                           n_iter=60, n_pop=18, m_uav=3, n_wp=6,
                           wind_speed=wind_speed, wind_dir_deg=wind_dir_deg,
                           tau_half_min=tau_half, dt_s=10.0, step_m=5,
                           altitude=40.0, theta=theta,
                           cooperative=False,
                           bounds=((0, (X.shape[1] - 1) * dx), (0, (Y.shape[0] - 1) * dx)))

    # Improved C-GWO (OBL + cooperative + reset)
    res_cgwo = cgwo_optimize_OBL(X, Y, dx, masks,
                                 n_iter=180, n_pop=18, m_uav=3, n_wp=6,
                                 wind_speed=wind_speed, wind_dir_deg=wind_dir_deg,
                                 tau_half_min=tau_half, dt_s=10.0, step_m=5,
                                 altitude=40.0, theta=theta,
                                 bounds=((0, (X.shape[1] - 1) * dx), (0, (Y.shape[0] - 1) * dx)),
                                 energy_weight=0.001,
                                 stagnation_patience=45, reset_frac=0.30)

    # Collect metrics
    rows = [
        ['Lawnmower', cov_lm,       uni_lm,       over_lm,       energy_lm],
        ['GWO',       res_gwo['cov'], res_gwo['uni'], res_gwo['over'], res_gwo['energy']],
        ['C-GWO+',    res_cgwo['cov'], res_cgwo['uni'], res_cgwo['over'], res_cgwo['energy']]
    ]
    df = pd.DataFrame(rows, columns=['Algorithm', 'Coverage(Cov)', 'Uniformity(Uni)', 'Overspray(Over)', 'Energy proxy'])
    print(df.to_string(index=False))

    # ---- Plot helpers ----
    def plot_heatmap(dep, title):
        plt.figure(figsize=(6, 4))
        plt.title(title)
        plt.imshow(dep, origin='lower')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()

    # Heatmaps
    plot_heatmap(dep_lm,          "Deposition heatmap - Lawnmower")
    plot_heatmap(res_gwo['dep'],  "Deposition heatmap - GWO (vanilla)")
    plot_heatmap(res_cgwo['dep'], "Deposition heatmap - C-GWO+ (OBL)")

    # Convergence histories
    plt.figure(figsize=(6, 4))
    plt.title("Convergence (best score)")
    plt.plot(res_gwo['hist'],  label='GWO (vanilla)')
    plt.plot(res_cgwo['hist'], label='C-GWO+ (OBL)')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
