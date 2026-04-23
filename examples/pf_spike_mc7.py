#PF algorithm w/ discovery spike
"""
Monte Carlo Simulation of the Cooper (2020) Potential Field guidance algorithm
vs baselines in SAREnv.

This script runs the evaluation N times, randomizing the clustered victim
locations on each run, and outputs statistical graphs of the performance metrics.

ENHANCEMENTS:
  - Checkpoint / resume: results are saved to CSV after every MC iteration.
    If the script is restarted it will automatically skip already-completed runs.
  - Interactive CLI: Prompts the user to resume or clear existing checkpoints.
  - Reward-time tracking: per-drone reward, mean-drone reward, and mean GRBF
    reward are recorded at every RECORD_EVERY step inside generate_pf_path.
  - Bounded Math: Integrates decoupled baseline coverage [0, 1] and spike arrays.
  - Concurrency & Identification: Outputs are isolated by dataset name to allow parallel runs.
"""

from __future__ import annotations
from functools import wraps
from pathlib import Path

import os
import json
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

import sarenv
from sarenv.core.cluster_lost_person import ClusterLostPersonLocationGenerator
from sarenv.analytics.evaluator import (
    ComparativeEvaluator,
    PathGenerator,
    PathGeneratorConfig,
)
from sarenv.analytics import metrics
from sarenv.utils import plot

log = sarenv.get_logger()

#User config
NUM_MC_RUNS = 500                   # Number of Monte Carlo iterations
DATA_DIR = "sarenv_dataset/58"    # Path to a single SAREnv dataset directory
EVALUATION_SIZES = ["medium"]            # SAREnv environment sizes to evaluate
NUM_DRONES = 3                     # Number of UAVs
NUM_LOST_PERSONS = 100                   # Synthetic victims to generate per run
BUDGET_M = 60000                 # Path budget per evaluation in metres

USE_PROBABILITY_MAP_AS_PRIOR = True
NUM_CLUSTERS = 20

#Extract the dataset name
DATASET_NAME = Path(DATA_DIR).name

#Output / checkpoint locations
OUTPUT_DIR = Path(f"mc_graphs_{DATASET_NAME}")
CHECKPOINT_CSV= OUTPUT_DIR / f"mc_checkpoint_results_{DATASET_NAME}.csv"
REWARD_CKPT_DIR= OUTPUT_DIR / f"reward_checkpoints_{DATASET_NAME}"

#Tuned PF Params

PF_W_S = 16.13108754983279
PF_W_C = 96.5493573808659
PF_DELTA_DEFAULT_M = None
PF_K = 184.1544534653549
PF_V_MAX = 10.0
PF_K_R_PLUS  = 0.012267695779649248
PF_K_R_MINUS = 1.9273804294374508
PF_DT = 0.5
PF_MU_DEFAULT_M = 66.0

USE_DISCOVERY_SPIKE = False
SPIKE_AMPLITUDE = 0
SPIKE_RADIUS_M = 0
SPIKE_COOLDOWN_S = 0.0

#Reward tracking
_last_reward_history: dict = {
    "drone_rewards": [],
    "mean_grbf_reward": [],
}

def compute_mu(fov_deg: float | None, altitude: float | None) -> float:
    if fov_deg is not None and altitude is not None:
        return float(altitude * np.tan(np.radians(fov_deg / 2.0)))
    return PF_MU_DEFAULT_M

def build_grbf_grid(center_x: float, center_y: float, max_radius: float, grid_res: int) -> np.ndarray:
    xs = np.linspace(center_x - max_radius, center_x + max_radius, grid_res)
    ys = np.linspace(center_y - max_radius, center_y + max_radius, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])
    dists = np.linalg.norm(all_pts - np.array([center_x, center_y]), axis=1)
    return all_pts[dists <= max_radius]

def initialise_rewards(p: np.ndarray, probability_map: np.ndarray | None, bounds: tuple[float, float, float, float] | None, use_prior: bool) -> np.ndarray:
    if not use_prior or probability_map is None or bounds is None:
        return np.ones(len(p))

    minx, miny, maxx, maxy = bounds
    h, w = probability_map.shape

    col_frac = np.clip((p[:, 0] - minx) / (maxx - minx) * (w - 1), 0, w - 1)
    row_frac = np.clip((p[:, 1] - miny) / (maxy - miny) * (h - 1), 0, h - 1)

    col0, row0 = np.floor(col_frac).astype(int), np.floor(row_frac).astype(int)
    col1, row1 = np.clip(col0 + 1, 0, w - 1), np.clip(row0 + 1, 0, h - 1)
    dc, dr = col_frac - col0, row_frac - row0

    rewards = (
        probability_map[row0, col0] * (1 - dc) * (1 - dr)
        + probability_map[row0, col1] * dc       * (1 - dr)
        + probability_map[row1, col0] * (1 - dc) * dr
        + probability_map[row1, col1] * dc       * dr
    )

    rmax = rewards.max()
    if rmax > 0:
        rewards /= rmax
    else:
        rewards = np.ones(len(p))

    return rewards.astype(float)

def pf_velocity(ksi: np.ndarray, r_total: np.ndarray, p: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    N = ksi.shape[0]
    diffs = ksi[:, np.newaxis, :] - p[np.newaxis, :, :]
    d_sq = np.sum(diffs ** 2, axis=2)
    weights = r_total * np.exp(-d_sq / delta)
    grad_s = np.sum(weights[:, :, np.newaxis] * diffs, axis=1)

    drone_diffs = ksi[:, np.newaxis, :] - ksi[np.newaxis, :, :]
    drone_dist_sq = np.sum(drone_diffs ** 2, axis=2) + 1e-6
    force_mag = 1.0 / (drone_dist_sq ** 2)
    np.fill_diagonal(force_mag, 0.0)
    grad_c = np.sum(force_mag[:, :, np.newaxis] * drone_diffs, axis=1)

    dV = (PF_W_S / delta) * grad_s - (PF_W_C * grad_c)
    v_cmds = -PF_K * dV

    speeds = np.linalg.norm(v_cmds, axis=1)
    overspeed_mask = speeds > PF_V_MAX
    if np.any(overspeed_mask):
        v_cmds[overspeed_mask] = (v_cmds[overspeed_mask] / speeds[overspeed_mask, np.newaxis]) * PF_V_MAX

    return v_cmds, d_sq

def update_r_base(r_base: np.ndarray, d_sq: np.ndarray, mu_sq: float, dt: float) -> np.ndarray:
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    r_dot = PF_K_R_PLUS * (1.0 - alpha) - (PF_K_R_MINUS * alpha * r_base)
    return np.clip(r_base + r_dot * dt, 0.0, 1.0)

def update_r_spike(r_spike: np.ndarray, d_sq: np.ndarray, mu_sq: float, dt: float) -> np.ndarray:
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    r_dot = - (PF_K_R_MINUS * alpha * r_spike)
    return np.maximum(0.0, r_spike + r_dot * dt)

def apply_discovery_spike(r_spike: np.ndarray, p: np.ndarray, discovery_xy: np.ndarray, spike_radius_sq: float, spike_amplitude: float) -> np.ndarray:
    diff = p - discovery_xy[np.newaxis, :]
    dist_sq = np.sum(diff ** 2, axis=1)
    pulse = spike_amplitude * np.exp(-dist_sq / spike_radius_sq)
    r_spike += pulse
    return r_spike


def drone_reward_scalar(ksi_i: np.ndarray, r_total: np.ndarray, p: np.ndarray, delta: float) -> float:
    diff = ksi_i[np.newaxis, :] - p
    d_sq = np.sum(diff ** 2, axis=1)
    w = np.exp(-d_sq / delta)
    return float(np.dot(w, r_total) / max(len(r_total), 1))

def generate_pf_path(
    center_x: float,
    center_y: float,
    max_radius: float,
    probability_map: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> list[LineString]:
    
    global _last_reward_history

    num_drones = int(kwargs.get('num_drones', 3))
    budget_m   = float(kwargs.get('budget', BUDGET_M))
    fov_deg    = kwargs.get('fov_deg')
    altitude   = kwargs.get('altitude')
    victim_locations_raw = kwargs.get('victim_locations', None)

    mu = compute_mu(fov_deg, altitude)
    target_spacing = mu * 1.5
    dynamic_grid_res = int(np.ceil((2 * max_radius) / target_spacing)) + 1
    p = build_grbf_grid(center_x, center_y, max_radius, dynamic_grid_res)
    grid_spacing = (2 * max_radius) / (dynamic_grid_res - 1)

    if PF_DELTA_DEFAULT_M is None:
        delta = (grid_spacing * 1.0) ** 2
    else:
        delta = PF_DELTA_DEFAULT_M ** 2

    r_base = initialise_rewards(p, probability_map, bounds, USE_PROBABILITY_MAP_AS_PRIOR)
    r_spike = np.zeros_like(r_base)

    do_spike = USE_DISCOVERY_SPIKE and victim_locations_raw is not None

    if do_spike:
        victims_arr = np.array(victim_locations_raw, dtype=float)
        if victims_arr.ndim == 1:
            victims_arr = victims_arr.reshape(-1, 2)
        victim_discovered = np.zeros(len(victims_arr), dtype=bool)
        spike_r = (SPIKE_RADIUS_M if SPIKE_RADIUS_M is not None else 3.0 * mu)
        spike_radius_sq = spike_r ** 2
    else:
        victims_arr = victim_discovered = spike_radius_sq = None

    init_radius = 5.0
    ksi = np.array(
        [
            [
                center_x + init_radius * np.cos(2 * np.pi * i / num_drones),
                center_y + init_radius * np.sin(2 * np.pi * i / num_drones),
            ]
            for i in range(num_drones)
        ],
        dtype=float,
    )

    budget_per_drone = budget_m / num_drones
    max_steps = int(np.ceil(budget_per_drone / (PF_V_MAX * PF_DT)))
    RECORD_EVERY = max(1, int(np.ceil(mu / (PF_V_MAX * PF_DT))))
    
    trajectories = [[ksi[i].copy()] for i in range(num_drones)]
    center_arr = np.array([center_x, center_y])
    mu_sq = mu ** 2

    _drone_reward_buf: list[list[float]] = []
    _grbf_mean_buf:   list[float]        = []

    for step in range(max_steps):
        r_total = r_base + r_spike
        velocities, d_sq = pf_velocity(ksi, r_total, p, delta)
        ksi += velocities * PF_DT

        offsets = ksi - center_arr
        dists = np.linalg.norm(offsets, axis=1)
        out_of_bounds = dists > max_radius
        if np.any(out_of_bounds):
            ksi[out_of_bounds] = (
                center_arr
                + offsets[out_of_bounds]
                * (max_radius / dists[out_of_bounds, np.newaxis])
            )

        if do_spike:
            victim_diff = ksi[:, np.newaxis, :] - victims_arr[np.newaxis, :, :]
            drone_to_victim_sq = np.sum(victim_diff ** 2, axis=2)
            min_drone_dist_sq = np.min(drone_to_victim_sq, axis=0)
            newly_found = (~victim_discovered) & (min_drone_dist_sq <= mu_sq)

            if np.any(newly_found):
                for vi in np.where(newly_found)[0]:
                    r_spike = apply_discovery_spike(
                        r_spike, p, victims_arr[vi], spike_radius_sq, SPIKE_AMPLITUDE
                    )
                    victim_discovered[vi] = True
                
                if np.all(victim_discovered):
                    do_spike = False

        r_base = update_r_base(r_base, d_sq, mu_sq, PF_DT)
        r_spike = update_r_spike(r_spike, d_sq, mu_sq, PF_DT)

        if (step + 1) % RECORD_EVERY == 0:
            for i in range(num_drones):
                trajectories[i].append(ksi[i].copy())

            drone_scalars = [
                drone_reward_scalar(ksi[i], r_total, p, delta) for i in range(num_drones)
            ]
            _drone_reward_buf.append(drone_scalars)
            _grbf_mean_buf.append(float(np.mean(r_total)))

    for i in range(num_drones):
        trajectories[i].append(ksi[i].copy())

    _last_reward_history["drone_rewards"] = _drone_reward_buf
    _last_reward_history["mean_grbf_reward"] = _grbf_mean_buf

    paths = []
    for i, traj in enumerate(trajectories):
        if len(traj) >= 2:
            paths.append(LineString(traj))
        else:
            paths.append(LineString())

    return paths

#Checkpoint system
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REWARD_CKPT_DIR.mkdir(parents=True, exist_ok=True)

def handle_existing_checkpoints() -> None:
    has_csv = CHECKPOINT_CSV.exists()
    has_jsons = REWARD_CKPT_DIR.exists() and any(REWARD_CKPT_DIR.iterdir())

    if has_csv or has_jsons:
        print("\n" + "!"*60)
        print(f"WARNING: Existing checkpoint data found for dataset '{DATASET_NAME}'!")
        print("!"*60)
        while True:
            choice = input("Resume, or clear? (r/c): ").strip().lower()
            if choice == 'r':
                log.info("Keeping existing checkpoints.")
                break
            elif choice == 'c':
                log.info("Deleting old data...")
                if CHECKPOINT_CSV.exists():
                    CHECKPOINT_CSV.unlink()
                if REWARD_CKPT_DIR.exists():
                    shutil.rmtree(REWARD_CKPT_DIR)
                ensure_dirs()
                log.info("Old data cleared")
                break
            else:
                print("Invalid input. Please enter 'r' to Resume or 'c' to Clear.")

def _load_completed_runs() -> tuple[set[int], pd.DataFrame]:
    if CHECKPOINT_CSV.exists():
        df = pd.read_csv(CHECKPOINT_CSV)
        if "MC_Run_ID" in df.columns:
            completed = set(df["MC_Run_ID"].unique().tolist())
            log.info(f"Checkpoint found - skipping already-completed runs: {sorted(completed)}")
            return completed, df
    return set(), pd.DataFrame()


def _save_metrics_checkpoint(df: pd.DataFrame) -> None:
    df.to_csv(CHECKPOINT_CSV, index=False)
    log.info(f"Checkpoint saved -> {CHECKPOINT_CSV}")


def _reward_ckpt_path(mc_run: int) -> Path:
    return REWARD_CKPT_DIR / f"run_{mc_run:04d}_rewards_{DATASET_NAME}.json"


def _save_reward_checkpoint(mc_run: int, reward_data: dict) -> None:
    path = _reward_ckpt_path(mc_run)
    with open(path, "w") as f:
        json.dump(reward_data, f)
    log.info(f"Reward checkpoint saved -> {path}")


def _load_all_reward_checkpoints(completed_runs: set[int]) -> list[dict]:
    histories = []
    for run_id in sorted(completed_runs):
        path = _reward_ckpt_path(run_id)
        if path.exists():
            with open(path) as f:
                histories.append(json.load(f))
        else:
            log.warning(f"Reward checkpoint missing for run {run_id}: {path}")
    return histories


#Plotting reward plot

def pad_to_common_length(arrays: list[list]) -> np.ndarray:
    max_len = max(len(a) for a in arrays)
    out = np.zeros((len(arrays), max_len))
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def generate_reward_plots(reward_histories: list[dict], output_dir: Path, num_drones: int, dataset_name: str) -> None:
    if not reward_histories:
        log.warning("No reward history data available — skipping reward plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    drone_runs = []
    grbf_runs  = []

    for h in reward_histories:
        dr = h.get("drone_rewards", [])
        mg = h.get("mean_grbf_reward", [])
        if dr:
            drone_runs.append(np.array(dr))
        if mg:
            grbf_runs.append(np.array(mg))

    def mc_mean_2d(arrays: list[np.ndarray]) -> np.ndarray:
        if not arrays:
            return np.zeros((1, num_drones))
        T_max = max(a.shape[0] for a in arrays)
        D     = arrays[0].shape[1]
        stack = np.zeros((len(arrays), T_max, D))
        for i, a in enumerate(arrays):
            stack[i, : a.shape[0], :] = a
        return np.mean(stack, axis=0)

    def mc_mean_1d(arrays: list[np.ndarray]) -> np.ndarray:
        if not arrays:
            return np.zeros(1)
        T_max = max(len(a) for a in arrays)
        stack = np.zeros((len(arrays), T_max))
        for i, a in enumerate(arrays):
            stack[i, : len(a)] = a
        return np.mean(stack, axis=0)

    avg_drone = mc_mean_2d(drone_runs)
    avg_grbf  = mc_mean_1d(grbf_runs)

    steps = np.arange(avg_drone.shape[0])
    palette = sns.color_palette("tab10", n_colors=num_drones)

    fig, ax = plt.subplots(figsize=(10, 5))
    for d in range(avg_drone.shape[1]):
        ax.plot(steps, avg_drone[:, d], label=f"Drone {d + 1}", color=palette[d], linewidth=1.6)
    ax.set_xlabel("Recorded Step")
    ax.set_ylabel("Experienced Reward (weighted GRBF)")
    ax.set_title(f"Per-Drone Reward vs Time — Dataset: {dataset_name} (N={len(reward_histories)})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = output_dir / f"mc_reward_per_drone_{dataset_name}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

    mean_across_drones = avg_drone.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mean_across_drones, color="steelblue", linewidth=2)
    ax.set_xlabel("Recorded Step")
    ax.set_ylabel("Mean Drone Reward (weighted GRBF)")
    ax.set_title(f"Mean Reward of All Drones vs Time — Dataset: {dataset_name} (N={len(reward_histories)})")
    plt.tight_layout()
    out = output_dir / f"mc_reward_mean_drones_{dataset_name}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

    grbf_steps = np.arange(len(avg_grbf))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grbf_steps, avg_grbf, color="darkorange", linewidth=2)
    ax.set_xlabel("Recorded Step")
    ax.set_ylabel("Mean GRBF Node Reward (Base + Spike)")
    ax.set_title(f"Mean GRBF Reward vs Time — Dataset: {dataset_name} (N={len(reward_histories)})")
    plt.tight_layout()
    out = output_dir / f"mc_reward_mean_grbf_{dataset_name}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


#Monte Carlo Plots
def generate_mc_plots(master_df: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = [
        "Likelihood Score", "Time-Discounted Score", "Victims Found (%)",
        "Area Covered (km^2)", "Total Path Length (km)",
    ]
    sns.set_theme(style="whitegrid")

    for metric in metrics_to_plot:
        if metric not in master_df.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=master_df, x="Algorithm", y=metric,
            hue="Dataset" if "Dataset" in master_df.columns else None, palette="viridis",
        )
        plt.title(f"Monte Carlo Distribution ({dataset_name}) (N={NUM_MC_RUNS}): {metric}")
        plt.xticks(rotation=15)
        plt.tight_layout()

        safe = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace("²", "sq")
        plot_file = output_dir / f"mc_{safe}_{dataset_name}.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()

#Monte carlo
if __name__ == "__main__":
    
    print(f"=== SCRIPT INITIATED: Running Dataset '{DATASET_NAME}' ===")

    ensure_dirs()
    handle_existing_checkpoints()

    completed_runs, existing_df = _load_completed_runs()
    all_results: list[pd.DataFrame] = [existing_df] if not existing_df.empty else []

    log.info(f"=== Starting Monte Carlo Evaluation (N={NUM_MC_RUNS}) for Dataset: {DATASET_NAME} ===")
    if completed_runs:
        remaining = NUM_MC_RUNS - len(completed_runs)
        log.info(f"Resuming — {len(completed_runs)} run(s) already done, {remaining} remaining.")

    config = PathGeneratorConfig(
        num_drones = NUM_DRONES,
        budget = BUDGET_M,
        fov_degrees= 45.0,
        altitude_meters= 80.0,
        overlap_ratio= 0.0,
        path_point_spacing_m= 10.0,
        transition_distance_m = 50.0,
        pizza_border_gap_m  = 15.0,
    )

    for mc_run in range(1, NUM_MC_RUNS + 1):
        if mc_run in completed_runs:
            log.info(f"  Skipping run {mc_run}/{NUM_MC_RUNS} (already completed).")
            continue

        log.info(f"\n--- Monte Carlo Iteration {mc_run}/{NUM_MC_RUNS} ---")

        path_cache = {}
        global_victim_locations = {}

        run_drone_rewards:  list[list[float]] = []
        run_grbf_mean:      list[float]       = []

        def get_cached_func(name, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                center_x   = args[0] if len(args) > 0 else kwargs.get('center_x')
                max_radius = args[2] if len(args) > 2 else kwargs.get('max_radius')

                if name == "PotentialField" and max_radius is not None:
                    radius_key = round(max_radius)
                    if radius_key in global_victim_locations:
                        kwargs['victim_locations'] = global_victim_locations[radius_key]

                key = (name, center_x)
                if key not in path_cache:
                    path_cache[key] = func(*args, **kwargs)

                    if name == "PotentialField":
                        run_drone_rewards.extend(_last_reward_history["drone_rewards"])
                        run_grbf_mean.extend(_last_reward_history["mean_grbf_reward"])

                return path_cache[key]
            return wrapper

        path_generators = {
            "PotentialField": PathGenerator(
                name="PotentialField", func=get_cached_func("PotentialField", generate_pf_path),
                path_generator_config=config,
            ),
            "Greedy": PathGenerator(
                name="Greedy", func=get_cached_func("Greedy", sarenv.analytics.paths.generate_greedy_path),
                path_generator_config=config,
            ),
            "Concentric": PathGenerator(
                name="Concentric", func=get_cached_func("Concentric", sarenv.analytics.paths.generate_concentric_circles_path),
                path_generator_config=config,
            ),
        }

        evaluator = ComparativeEvaluator(
            dataset_directory= DATA_DIR,
            evaluation_sizes= EVALUATION_SIZES,
            num_drones  = NUM_DRONES,
            num_lost_persons= NUM_LOST_PERSONS,
            budget= BUDGET_M,
            path_generators = path_generators,
            path_generator_config = config,
        )

        person_per_cluster = NUM_LOST_PERSONS // NUM_CLUSTERS

        for size, env_data in evaluator.environments.items():
            item = env_data["item"]
            cluster_generator = ClusterLostPersonLocationGenerator(item)
            locations = cluster_generator.generate_locations(NUM_CLUSTERS, 0)
            lp_locs = cluster_generator.generate_cluster_LPs(locations, NUM_CLUSTERS, person_per_cluster)

            if lp_locs:
                env_data["victims"] = gpd.GeoDataFrame(geometry=lp_locs, crs=item.features.crs)
                radius_key = round(item.radius_km * 1000)
                global_victim_locations[radius_key] = [(p.x, p.y) for p in lp_locs]

        results_df, _ = evaluator.run_baseline_evaluations()

        if not results_df.empty:
            results_df["MC_Run_ID"] = mc_run
            all_results.append(results_df)

            combined_so_far = pd.concat(all_results, ignore_index=True)
            _save_metrics_checkpoint(combined_so_far)

        _save_reward_checkpoint(
            mc_run,
            {
                "drone_rewards":    run_drone_rewards,
                "mean_grbf_reward": run_grbf_mean,
            },
        )

        completed_runs.add(mc_run)

    if all_results:
        master_results_df = pd.concat(all_results, ignore_index=True)

        log.info("\n=== MONTE CARLO AGGREGATE SUMMARY ===")
        summary_stats = master_results_df.groupby(["Algorithm", "Dataset"]).agg(
            {
                "Likelihood Score":     ["mean", "std"],
                "Time-Discounted Score": ["mean", "std"],
                "Victims Found (%)":    ["mean", "std"],
            }
        ).round(3)
        print(summary_stats.to_string())

        log.info("\n--- Generating Monte Carlo Metric Plots ---")
        generate_mc_plots(master_results_df, output_dir=OUTPUT_DIR, dataset_name=DATASET_NAME)

        log.info("\n--- Generating Reward-over-Time Plots ---")
        reward_histories = _load_all_reward_checkpoints(completed_runs)
        generate_reward_plots(reward_histories, output_dir=OUTPUT_DIR, num_drones=NUM_DRONES, dataset_name=DATASET_NAME)

        log.info(f"=== Monte Carlo Evaluation Complete for Dataset: {DATASET_NAME} ===")
        print("=" * 60)
        print(f"=== SCRIPT COMPLETED: Finished Dataset '{DATASET_NAME}' ===")
        print("=" * 60)
    else:
        log.error("No results were generated across any runs. Check dataset parameters.")
