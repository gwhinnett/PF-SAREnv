
#Visualizes the Potential Field algorithm's path overlaid on a heatmap

from __future__ import annotations
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

import sarenv
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.analytics.evaluator import (
    ComparativeEvaluator,
    PathGenerator,
    PathGeneratorConfig,
)
from sarenv.utils import plot

log = sarenv.get_logger()

#User config

DATA_DIR = "sarenv_dataset/1"    
EVALUATION_SIZES = ["medium"]            
NUM_DRONES = 3                     
NUM_LOST_PERSONS = 100                   
BUDGET_M = 100000                 

#Place tuned params here

PF_W_S = 23.28396146912085
PF_W_C = 33.22088093347115
PF_DELTA_DEFAULT_M = None
PF_K = 203.88548299439373
PF_V_MAX = 10.0
PF_K_R_PLUS  = 0.004086627304453559
PF_K_R_MINUS = 2.4430790740931654
PF_DT = 0.5
PF_MU_DEFAULT_M = 66.0

USE_PROBABILITY_MAP = True
USE_DISCOVERY_SPIKE = True
SPIKE_AMPLITUDE = 40
SPIKE_RADIUS_M = 10

def _compute_mu(fov_deg: float | None, altitude: float | None) -> float:
    if fov_deg is not None and altitude is not None:
        return float(altitude * np.tan(np.radians(fov_deg / 2.0)))
    return PF_MU_DEFAULT_M

def _build_grbf_grid(center_x: float, center_y: float, max_radius: float, grid_res: int) -> np.ndarray:
    xs = np.linspace(center_x - max_radius, center_x + max_radius, grid_res)
    ys = np.linspace(center_y - max_radius, center_y + max_radius, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])
    dists = np.linalg.norm(all_pts - np.array([center_x, center_y]), axis=1)
    return all_pts[dists <= max_radius]

def _initialise_rewards(p: np.ndarray, probability_map: np.ndarray | None, bounds: tuple[float, float, float, float] | None, use_prior: bool) -> np.ndarray:
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

def _pf_velocity(ksi: np.ndarray, r_total: np.ndarray, p: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
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

def _update_r_base(r_base: np.ndarray, d_sq: np.ndarray, mu_sq: float, dt: float) -> np.ndarray:
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    r_dot = PF_K_R_PLUS * (1.0 - alpha) - (PF_K_R_MINUS * alpha * r_base)
    return np.clip(r_base + r_dot * dt, 0.0, 1.0)

def _update_r_spike(r_spike: np.ndarray, d_sq: np.ndarray, mu_sq: float, dt: float) -> np.ndarray:
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    r_dot = - (PF_K_R_MINUS * alpha * r_spike)
    return np.maximum(0.0, r_spike + r_dot * dt)

def _apply_discovery_spike(r_spike: np.ndarray, p: np.ndarray, discovery_xy: np.ndarray, spike_radius_sq: float, spike_amplitude: float) -> np.ndarray:
    diff = p - discovery_xy[np.newaxis, :]
    dist_sq = np.sum(diff ** 2, axis=1)
    pulse = spike_amplitude * np.exp(-dist_sq / spike_radius_sq)
    r_spike += pulse
    return r_spike

def generate_pf_path(
    center_x: float,
    center_y: float,
    max_radius: float,
    probability_map: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> list[LineString]:
    num_drones = int(kwargs.get('num_drones', 3))
    budget_m   = float(kwargs.get('budget', BUDGET_M))
    fov_deg    = kwargs.get('fov_deg')
    altitude   = kwargs.get('altitude')
    victim_locations_raw = kwargs.get('victim_locations', None)

    mu = _compute_mu(fov_deg, altitude)
    target_spacing = mu * 1.5
    dynamic_grid_res = int(np.ceil((2 * max_radius) / target_spacing)) + 1
    p = _build_grbf_grid(center_x, center_y, max_radius, dynamic_grid_res)
    grid_spacing = (2 * max_radius) / (dynamic_grid_res - 1)

    if PF_DELTA_DEFAULT_M is None:
        delta = (grid_spacing * 1.0) ** 2
    else:
        delta = PF_DELTA_DEFAULT_M ** 2

    r_base = _initialise_rewards(p, probability_map, bounds, USE_PROBABILITY_MAP)
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
    record_every = max(1, int(np.ceil(mu / (PF_V_MAX * PF_DT))))
    
    trajectories = [[ksi[i].copy()] for i in range(num_drones)]
    center_arr = np.array([center_x, center_y])
    mu_sq = mu ** 2

    for step in range(max_steps):
        r_total = r_base + r_spike
        velocities, d_sq = _pf_velocity(ksi, r_total, p, delta)
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
                    r_spike = _apply_discovery_spike(
                        r_spike, p, victims_arr[vi], spike_radius_sq, SPIKE_AMPLITUDE
                    )
                    victim_discovered[vi] = True
                if np.all(victim_discovered):
                    do_spike = False

        r_base = _update_r_base(r_base, d_sq, mu_sq, PF_DT)
        r_spike = _update_r_spike(r_spike, d_sq, mu_sq, PF_DT)

        if (step + 1) % record_every == 0:
            for i in range(num_drones):
                trajectories[i].append(ksi[i].copy())

    for i in range(num_drones):
        trajectories[i].append(ksi[i].copy())

    paths = []
    for i, traj in enumerate(trajectories):
        if len(traj) >= 2:
            paths.append(LineString(traj))
        else:
            paths.append(LineString())

    return paths


#Main script
if __name__ == "__main__":
    log.info("Starting Heatmap Generation")

    config = PathGeneratorConfig(
        num_drones = NUM_DRONES,
        budget = BUDGET_M,
        fov_degrees  = 45.0,
        altitude_meters = 80.0,
        overlap_ratio = 0.0,
        path_point_spacing_m = 10.0,
        transition_distance_m = 50.0,
        pizza_border_gap_m = 15.0,
    )

    path_cache = {}
    global_victim_locations = {}

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
        dataset_directory = DATA_DIR,
        evaluation_sizes = EVALUATION_SIZES,
        num_drones = NUM_DRONES,
        num_lost_persons = NUM_LOST_PERSONS,
        budget = BUDGET_M,
        path_generators = path_generators,
        path_generator_config = config,
    )

    # Setup the environments and victims
    for size, env_data in evaluator.environments.items():
        item = env_data["item"]
        generator = LostPersonLocationGenerator(item)
        lp_locs = generator.generate_locations(NUM_LOST_PERSONS, 0)

        if lp_locs:
            env_data["victims"] = gpd.GeoDataFrame(geometry=lp_locs, crs=item.features.crs)
            radius_key = round(item.radius_km * 1000)
            global_victim_locations[radius_key] = [(p.x, p.y) for p in lp_locs]

    # Heatmap Generation
    log.info("Generating Heatmaps...")
    heatmap_dir = Path("graphs/heatmaps")
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for size, env_data in evaluator.environments.items():
        item = env_data["item"]
        
        # Get center point in projected coordinates (UTM)
        center_proj = (
            gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
            .to_crs(env_data["crs"])
            .geometry.iloc[0]
        )

        for name, generator in evaluator.path_generators.items():
            log.info(f"Generating heatmap plot for {name}...")
            
            # Use the cached path generation
            generated_paths = generator(
                center_x=center_proj.x,
                center_y=center_proj.y,
                max_radius=item.radius_km * 1000,
                probability_map=item.heatmap,
                bounds=item.bounds,
            )

            output_file = heatmap_dir / f"{name}_{size}_heatmap.pdf"

            plot.plot_heatmap(
                item=item,
                generated_paths=generated_paths,
                name=name,
                x_min=item.bounds[0],
                x_max=item.bounds[2],
                y_min=item.bounds[1],
                y_max=item.bounds[3],
                output_file=str(output_file)
            )

            log.info(f"Saved heatmap -> {output_file}")
            
    log.info("Process Complete.")
