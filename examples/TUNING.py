
#Hyperparameter Optimization of the Potential Field Algorithm for SAREnv.
#Optimizing for total victims found across multiple datasets.


import optuna
import numpy as np
import pandas as pd
import geopandas as gpd
from functools import wraps
from pathlib import Path
from shapely.geometry import LineString, Point

import sarenv
from sarenv.core.cluster_lost_person import ClusterLostPersonLocationGenerator
from sarenv.analytics.evaluator import ComparativeEvaluator, PathGenerator, PathGeneratorConfig

log = sarenv.get_logger()
# Suppress excessive logging
log.setLevel("WARNING") 

DATASETS = ["4", "7", "12", "16", "23", "29", "33", "37", "45", "49", "51", "58", "tryfan", "dartmoor"]
EVALUATION_SIZES = ["medium"]
NUM_DRONES = 3
NUM_LOST_PERSONS = 100
NUM_CLUSTERS = 20
BUDGET_M = 60000
#Use SAREnv heatmap to bias GRBF rewards
USE_PROBABILITY_MAP = True

N_TRIALS = 50 
NUM_EVALS_PER_DATASET = 1

PF_V_MAX = 10.0 #Max drone speed
PF_DT = 0.5 #Time step

def compute_mu(fov_deg, altitude):
    if fov_deg is not None and altitude is not None:
        return float(altitude * np.tan(np.radians(fov_deg / 2.0)))
    return 66.0

def pf_velocity(ksi, r_total, p, delta, params):
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

    dV = (params['pf_w_s'] / delta) * grad_s - (params['pf_w_c'] * grad_c)
    v_cmds = -params['pf_k'] * dV

    speeds = np.linalg.norm(v_cmds, axis=1)
    overspeed_mask = speeds > PF_V_MAX
    if np.any(overspeed_mask):
        v_cmds[overspeed_mask] = (v_cmds[overspeed_mask] / speeds[overspeed_mask, np.newaxis]) * PF_V_MAX

    return v_cmds, d_sq

def update_r_base(r_base, d_sq, mu_sq, dt, params):
    #Updates bounded baseline coverage field
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    r_dot = params['pf_k_r_plus'] * (1.0 - alpha) - (params['pf_k_r_minus'] * alpha * r_base)
    return np.clip(r_base + r_dot * dt, 0.0, 1.0)

def update_r_spike(r_spike, d_sq, mu_sq, dt, params):
    #Updates spike field for decay only
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    r_dot = - (params['pf_k_r_minus'] * alpha * r_spike)
    return np.maximum(0.0, r_spike + r_dot * dt)

def generate_pf_path(center_x, center_y, max_radius, probability_map=None, bounds=None, **kwargs):
    params = kwargs.get('opt_params')
    num_drones = int(kwargs.get('num_drones', 3))
    budget_m   = float(kwargs.get('budget', BUDGET_M))
    mu = compute_mu(kwargs.get('fov_deg'), kwargs.get('altitude'))
    
    target_spacing = mu * 1.5
    dynamic_grid_res = int(np.ceil((2 * max_radius) / target_spacing)) + 1
    
    xs = np.linspace(center_x - max_radius, center_x + max_radius, dynamic_grid_res)
    ys = np.linspace(center_y - max_radius, center_y + max_radius, dynamic_grid_res)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])
    dists = np.linalg.norm(all_pts - np.array([center_x, center_y]), axis=1)
    p = all_pts[dists <= max_radius]
    
    grid_spacing = (2 * max_radius) / (dynamic_grid_res - 1)
    delta = (grid_spacing * 1.0) ** 2
    
    #Initialising rewards
    r_base = np.ones(len(p)) 
    r_spike = np.zeros(len(p))
    
    victims_arr = np.array(kwargs.get('victim_locations', []), dtype=float)
    do_spike = len(victims_arr) > 0
    if do_spike:
        if victims_arr.ndim == 1: victims_arr = victims_arr.reshape(-1, 2)
        victim_discovered = np.zeros(len(victims_arr), dtype=bool)
        spike_radius_sq = (params['spike_radius_mult'] * mu) ** 2

    init_radius = 5.0  #Provides a push at start of simulation
    ksi = np.array([[center_x + init_radius * np.cos(2 * np.pi * i / num_drones), 
                     center_y + init_radius * np.sin(2 * np.pi * i / num_drones)] for i in range(num_drones)], dtype=float)

    max_steps = int(np.ceil((budget_m / num_drones) / (PF_V_MAX * PF_DT)))
    record_every = max(1, int(np.ceil(mu / (PF_V_MAX * PF_DT))))
    trajectories = [[list(ksi[i])] for i in range(num_drones)]
    mu_sq = mu ** 2

    for step in range(max_steps):
        r_total = r_base + r_spike
        velocities, d_sq = pf_velocity(ksi, r_total, p, delta, params)
        ksi += velocities * PF_DT

        offsets = ksi - np.array([center_x, center_y])
        dists = np.linalg.norm(offsets, axis=1)
        out = dists > max_radius
        if np.any(out):
            ksi[out] = np.array([center_x, center_y]) + offsets[out] * (max_radius / dists[out, np.newaxis])

        if do_spike:
            victim_diff = ksi[:, np.newaxis, :] - victims_arr[np.newaxis, :, :]
            drone_to_victim_sq = np.sum(victim_diff ** 2, axis=2)
            min_drone_dist_sq = np.min(drone_to_victim_sq, axis=0)
            newly_found = (~victim_discovered) & (min_drone_dist_sq <= mu_sq)

            if np.any(newly_found):
                for vi in np.where(newly_found)[0]:
                    diff = p - victims_arr[vi][np.newaxis, :]
                    dist_sq_spike = np.sum(diff ** 2, axis=1)
                    # Apply directly to spike array
                    r_spike += params['spike_amplitude'] * np.exp(-dist_sq_spike / spike_radius_sq)
                    victim_discovered[vi] = True

        if (step + 1) % record_every == 0:
            for i in range(num_drones): trajectories[i].append(list(ksi[i]))
            
        r_base = update_r_base(r_base, d_sq, mu_sq, PF_DT, params)
        r_spike = update_r_spike(r_spike, d_sq, mu_sq, PF_DT, params)

    for i in range(num_drones): trajectories[i].append(list(ksi[i]))
    return [LineString(traj) if len(traj) >= 2 else LineString() for traj in trajectories]

#Optimisation function
def objective(trial):
    
    #Parameter limits
    params = {
        "pf_w_s": trial.suggest_float("pf_w_s", 1.0, 30.0),             
        "pf_w_c": trial.suggest_float("pf_w_c", 10.0, 100.0),           
        "pf_k": trial.suggest_float("pf_k", 50.0, 500.0),               
        "pf_k_r_plus": trial.suggest_float("pf_k_r_plus", 0.001, 0.05), 
        "pf_k_r_minus": trial.suggest_float("pf_k_r_minus", 1.0, 5.0), 
    }

    config = PathGeneratorConfig(
        num_drones=NUM_DRONES, budget=BUDGET_M, fov_degrees=45.0, 
        altitude_meters=80.0, overlap_ratio=0.0
    )

    all_scores = []

    #Loop through all datasets
    for dataset_id in DATASETS:
        dataset_path = f"sarenv_dataset/{dataset_id}"
        
        if not Path(dataset_path).exists():
            log.warning(f"Dataset missing, skipping: {dataset_path}")
            continue

        #MC evaluations for a specific dataset
        for run in range(NUM_EVALS_PER_DATASET):
            path_cache = {}
            global_victim_locations = {}

            def pf_wrapper(*args, **kwargs):
                kwargs['opt_params'] = params 
                center_x = args[0] if len(args) > 0 else kwargs.get('center_x')
                max_radius = args[2] if len(args) > 2 else kwargs.get('max_radius')
                if max_radius is not None:
                    radius_key = round(max_radius)
                    if radius_key in global_victim_locations:
                        kwargs['victim_locations'] = global_victim_locations[radius_key]
                
                key = ("PotentialField", center_x)
                if key not in path_cache:
                    path_cache[key] = generate_pf_path(*args, **kwargs)
                return path_cache[key]

            evaluator = ComparativeEvaluator(
                dataset_directory=dataset_path,
                evaluation_sizes=EVALUATION_SIZES,
                num_drones=NUM_DRONES,
                num_lost_persons=NUM_LOST_PERSONS,
                budget=BUDGET_M,
                path_generators={
                    "PotentialField": PathGenerator(
                        name="PotentialField", func=pf_wrapper, path_generator_config=config
                    )
                },
                path_generator_config=config,
            )

            person_per_cluster = NUM_LOST_PERSONS // NUM_CLUSTERS
            
            for size, env_data in evaluator.environments.items():
                item = env_data["item"]
                cluster_generator = ClusterLostPersonLocationGenerator(item)
                locations = cluster_generator.generate_locations(NUM_CLUSTERS, 0)
                lp_locs = cluster_generator.generate_cluster_LPs(locations, NUM_CLUSTERS, person_per_cluster)
                if lp_locs:
                    env_data["victims"] = gpd.GeoDataFrame(geometry=lp_locs, crs=item.features.crs)
                    global_victim_locations[round(item.radius_km * 1000)] = [(p.x, p.y) for p in lp_locs]

            results_df, _ = evaluator.run_baseline_evaluations()
                    
            if not results_df.empty:
                score = results_df[results_df["Algorithm"] == "PotentialField"]["Victims Found (%)"].mean()
                all_scores.append(score)

    #Mean of all datasets and runs
    if not all_scores:
        return 0.0
    return np.mean(all_scores)

if __name__ == "__main__":
    print(f"Optimising across {len(DATASETS)} datasets...")
    print(f"{NUM_EVALS_PER_DATASET} MC iterations per dataset per trial")
    print(f"Total simulations per trial: {len(DATASETS) * NUM_EVALS_PER_DATASET}")
    
#Create study
    study = optuna.create_study(
        direction="maximize", 
        study_name="PF_Generalization_Opt_Victims",
        storage="sqlite:///pf_tuning_victims.db", 
        load_if_exists=True                 
    )
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "="*50)
    print("Optimization Complete!")
    print(f"Best Generalized Victims Found (%): {study.best_value:.2f}%")
