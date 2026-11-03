import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.constants import DATASETS, DOWNTIME, FEATURES, FIGURES, SEED, TABLES

COLS = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]


def select_design():
    """Pick 30 batches: 8 corners, the center, and Sobol points on the 7-level grid."""
    df = pd.read_csv(DATASETS / "initial_conditions.csv")
    levels = [np.sort(df[c].unique()) for c in COLS]
    chosen = list(itertools.product([0, 6], repeat=3))
    chosen.append((3, 3, 3))
    for p in qmc.Sobol(d=3, scramble=True, seed=SEED).random(64):
        idx = tuple(int(i) for i in np.round(p * 6))
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) == 30:
            break
    rng = np.random.default_rng(SEED)
    rows = []
    for gi, qi, vi in chosen:
        match = df[(df[COLS[0]] == levels[0][gi]) & (df[COLS[1]] == levels[1][qi])
                   & (df[COLS[2]] == levels[2][vi])]
        rows.append(match.iloc[rng.integers(len(match))])
    design = pd.DataFrame(rows).reset_index(drop=True)
    design["File"] = design["Batch ID"].apply(lambda b: f"batch_{int(b):04d}.csv")
    design.to_csv(TABLES / "design.csv", index=False)
    return design


def load_data(design):
    frames = []
    for _, row in design.iterrows():
        b = pd.read_csv(DATASETS / row["File"])
        b["Batch ID"] = int(row["Batch ID"])
        b["G0"], b["Q0"], b["VCD0"] = row[COLS[0]], row[COLS[1]], row[COLS[2]]
        b["Time [d]"] = b["Time [h]"] / 24
        frames.append(b)
    data = pd.concat(frames, ignore_index=True)
    return data[data["Time [h]"] % 16 == 0].reset_index(drop=True)


def new_gp():
    kernel = C(1.0) * Matern(length_scale=[1.0] * 4, nu=2.5) + WhiteKernel(0.01)
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED))


def compare_models(X, y, groups):
    models = {
        "Gaussian process": new_gp(),
        "Cubic polynomial ridge": make_pipeline(
            StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1.0)),
        "Gradient boosting": GradientBoostingRegressor(random_state=SEED),
    }
    results, gp_pred = [], None
    for name, model in models.items():
        pred, std = np.zeros_like(y), np.full_like(y, np.nan)
        for train, test in GroupKFold(n_splits=5).split(X, y, groups):
            model.fit(X[train], y[train])
            if name == "Gaussian process":
                pred[test], std[test] = model.predict(X[test], return_std=True)
            else:
                pred[test] = model.predict(X[test])
        if name == "Gaussian process":
            gp_pred = pred.copy()
        results.append({
            "Model": name, "R2": r2_score(y, pred),
            "RMSE [mg/L]": np.sqrt(mean_squared_error(y, pred)),
            "95% coverage": np.mean(np.abs(y - pred) <= 1.96 * std)
            if name == "Gaussian process" else np.nan})
    table = pd.DataFrame(results)
    table.to_csv(TABLES / "model_comparison.csv", index=False)
    return table, gp_pred


def predict_chunks(model, grid, size=20000):
    out = [model.predict(grid[i:i + size], return_std=True) for i in range(0, len(grid), size)]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def economics(grid, mean):
    cost = 10 * grid[:, 0] + 20 * grid[:, 1] + 0.0005 * grid[:, 2]
    den = grid[:, 3] + DOWNTIME
    return cost, (1000 * mean - cost) / den, mean / den


def search(model, G, Q, V, T):
    grid = np.array(np.meshgrid(G, Q, V, T, indexing="ij")).reshape(4, -1).T
    mean, std = predict_chunks(model, grid)
    cost, stp, sty = economics(grid, mean)
    i = np.argmax(stp)
    return grid[i], mean[i], std[i], cost[i], stp[i], sty[i]


def optimize(model):
    """Coarse search over the full sampled range, then a fine search around the winner."""
    g, q, v, t = search(model, np.linspace(5, 35, 13), np.linspace(1, 7, 13),
                        np.linspace(2e5, 1.4e6, 13), np.linspace(1, 28, 28))[0]
    g, q, v, t = search(model,
                        np.arange(max(5, g - 5), min(35, g + 5) + 0.1, 1.0),
                        np.arange(max(1, q - 1), min(7, q + 1) + 0.01, 0.25),
                        np.arange(max(2e5, v - 2e5), min(1.4e6, v + 2e5) + 1, 5e4),
                        np.arange(max(0.5, t - 1.5), t + 1.51, 0.1))[0]
    point, mean, std, cost, stp, sty = search(model, [g], [q], [v], [t])
    g, q, v, t = point
    den = t + DOWNTIME
    lo, hi = mean - 1.96 * std, mean + 1.96 * std
    table = pd.DataFrame({
        "Quantity": ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]", "Harvest time [d]",
                     "Cost [$/L]", "Value [$/L]", "Space-time profit [$/(L d)]",
                     "Space-time yield [mg/(L d)]"],
        "Estimate": [g, q, v, t, cost, 1000 * mean, stp, sty],
        "95% PI lower": [np.nan] * 5 + [1000 * lo, (1000 * lo - cost) / den, lo / den],
        "95% PI upper": [np.nan] * 5 + [1000 * hi, (1000 * hi - cost) / den, hi / den]})
    table.to_csv(TABLES / "recommendation.csv", index=False)
    return table, (g, q, v, t)


def plot_design(design):
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(design[COLS[a]], design[COLS[b]], s=40)
        ax.set_xlabel(COLS[a])
        ax.set_ylabel(COLS[b])
    fig.suptitle("Experimental design: 30 selected batches")
    fig.tight_layout()
    fig.savefig(FIGURES / "design.png", dpi=200)
    plt.close(fig)


def plot_predicted_vs_actual(y, pred):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, pred, s=6, alpha=0.5)
    lim = [0, max(y.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "k--")
    ax.set_xlabel("Actual mAb [mg/L]")
    ax.set_ylabel("Predicted mAb [mg/L] (held-out batches)")
    fig.tight_layout()
    fig.savefig(FIGURES / "predicted_vs_actual.png", dpi=200)
    plt.close(fig)


def plot_stp_vs_time(model, best):
    g, q, v, t = best
    tt = np.arange(0.5, 28.01, 0.1)
    curve = np.column_stack([np.full_like(tt, g), np.full_like(tt, q), np.full_like(tt, v), tt])
    m, s = predict_chunks(model, curve)
    c, d = 10 * g + 20 * q + 0.0005 * v, tt + DOWNTIME
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tt, (1000 * m - c) / d, label="Predicted mean")
    ax.fill_between(tt, (1000 * (m - 1.96 * s) - c) / d, (1000 * (m + 1.96 * s) - c) / d,
                    alpha=0.25, label="95% PI")
    ax.axvline(t, color="k", linestyle="--", label=f"Optimum: {t:.1f} d")
    ax.set_xlabel("Harvest time [d]")
    ax.set_ylabel("Space-time profit [$/(L·d)]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "stp_vs_time.png", dpi=200)
    plt.close(fig)


def plot_effects(model, best):
    g, q, v, _ = best
    base = dict(G0=g, Q0=q, VCD0=v)
    ranges = {"G0": (5, 35, "Glucose [mM]"), "Q0": (1, 7, "Glutamine [mM]"),
              "VCD0": (2e5, 1.4e6, "VCD [cells/mL]")}
    T = np.arange(1, 14.01, 0.25)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    rows = []
    for j, (key, (lo, hi, label)) in enumerate(ranges.items()):
        vals = np.linspace(lo, hi, 25)
        best_stp, best_sty = [], []
        for val in vals:
            p = dict(base)
            p[key] = val
            grid = np.column_stack([np.full_like(T, p["G0"]), np.full_like(T, p["Q0"]),
                                    np.full_like(T, p["VCD0"]), T])
            mu = model.predict(grid)
            cost = 10 * p["G0"] + 20 * p["Q0"] + 0.0005 * p["VCD0"]
            best_stp.append(((1000 * mu - cost) / (T + DOWNTIME)).max())
            best_sty.append((mu / (T + DOWNTIME)).max())
        axes[0, j].plot(vals, best_stp)
        axes[0, j].set_ylabel("Max space-time profit [$/(L·d)]")
        axes[1, j].plot(vals, best_sty, color="tab:orange")
        axes[1, j].set_ylabel("Max space-time yield [mg/(L·d)]")
        axes[1, j].set_xlabel(label)
        rows += [{"Variable": label, "Value": a, "Max STP": b, "Max STY": c}
                 for a, b, c in zip(vals, best_stp, best_sty)]
    fig.suptitle("Effect of each initial condition (others at the optimum, harvest time optimized)")
    fig.tight_layout()
    fig.savefig(FIGURES / "initial_condition_effects.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TABLES / "initial_condition_effects.csv", index=False)
