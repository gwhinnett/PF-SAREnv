#Integrates the Cooper (2020) PF algorithm into SAREnv and compares it against the baseline planners.


#The PF algorithm is described in:
#Cooper, J. R. (2020). Optimal Multi-Agent Search and Rescue Using Potential Field Theory. AIAA SciTech 2020 Forum. DOI: 10.2514/6.2020-0879

#Optional mechanism added: when any drone comes within mu metres of a known victim location, a Gaussian reward pulse of amplitude SPIKE_AMPLITUDE
#is added to every GRBF centre within SPIKE_RADIUS metres of the discovery site.
#The pulse decays naturally via the existing K_R_MINUS drain term. Can be enabled / disabled using flag USE_DISCOVERY_SPIKE.

from __future__ import annotations
from functools import wraps
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from sarenv.core.cluster_lost_person import ClusterLostPersonLocationGenerator

import sarenv
from sarenv.analytics.evaluator import (
    ComparativeEvaluator,
    PathGenerator,
    PathGeneratorConfig,
)
from sarenv.analytics import metrics
from sarenv.utils import plot

log = sarenv.get_logger()

#User config

DATA_DIR = "sarenv_dataset/1"   
EVALUATION_SIZES = ["medium"]         
NUM_DRONES = 3         
NUM_LOST_PERSONS = 100    
BUDGET_M = 100000

USE_DISCOVERY_SPIKE = True
SPIKE_AMPLITUDE = 10.0
#None is calculate dynamically: 3 * mu.
SPIKE_RADIUS_M = None
#0: each lost person can only ever spike the field once.
SPIKE_COOLDOWN_S = 0.0

USE_PROBABILITY_MAP = False

NUM_CLUSTERS = 10

#Fallback value; likely far too coarse
PF_GRID_RES = 15

PF_W_S = 15    # Search weight
PF_W_C = 30     # Collision avoidance weight

#GRBF width delta in metres. Use none to calculate dynamically.
PF_DELTA_DEFAULT_M = None

#Gradient descent gain K
PF_K = 500

PF_V_MAX = 10.0  # m/s

#Reward rate gains
PF_K_R_PLUS  = 0.01
PF_K_R_MINUS = 2.0

#Timestep in seconds
PF_DT = 0.5

#Measurement radius in metres. Fallback.
PF_MU_DEFAULT_M = 66.0


def _compute_mu(fov_deg: float | None, altitude: float | None) -> float:

    if fov_deg is not None and altitude is not None:
        return float(altitude * np.tan(np.radians(fov_deg / 2.0)))
    return PF_MU_DEFAULT_M


def _build_grbf_grid(
    center_x: float,
    center_y: float,
    max_radius: float,
    grid_res: int,
) -> np.ndarray:
    #Places GRBF centres on a grid. Returns an (M, 2) array of (x, y) positions in metres.
    #M is the number of centres whose distance from the IPP is ≤ max_radius.

    xs = np.linspace(center_x - max_radius, center_x + max_radius, grid_res)
    ys = np.linspace(center_y - max_radius, center_y + max_radius, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    # Keep only points inside the search circle
    dists = np.linalg.norm(all_pts - np.array([center_x, center_y]), axis=1)
    return all_pts[dists <= max_radius]


def _initialise_rewards(
    p: np.ndarray,
    probability_map: np.ndarray | None,
    bounds: tuple[float, float, float, float] | None,
    use_prior: bool,
) -> np.ndarray:

#    Initialises the GRBF reward vector r (shape: (M,)).
#
#   If use_prior is True and a probability map is available, each GRBF centre's initial reward is set proportional to the probability map value at that
#   location. Otherwise: all rewards are 1.0.

    if not use_prior or probability_map is None or bounds is None:
        return np.ones(len(p))

    minx, miny, maxx, maxy = bounds
    h, w = probability_map.shape

    # Map each GRBF centre's world coordinate to a fractional grid index
    col_frac = (p[:, 0] - minx) / (maxx - minx) * (w - 1)
    row_frac = (p[:, 1] - miny) / (maxy - miny) * (h - 1)

    # Clamp to valid range
    col_frac = np.clip(col_frac, 0, w - 1)
    row_frac = np.clip(row_frac, 0, h - 1)

    # Bilinear interpolation into the probability map
    col0 = np.floor(col_frac).astype(int)
    row0 = np.floor(row_frac).astype(int)
    col1 = np.clip(col0 + 1, 0, w - 1)
    row1 = np.clip(row0 + 1, 0, h - 1)

    dc = col_frac - col0
    dr = row_frac - row0

    rewards = (
        probability_map[row0, col0] * (1 - dc) * (1 - dr)
        + probability_map[row0, col1] * dc       * (1 - dr)
        + probability_map[row1, col0] * (1 - dc) * dr
        + probability_map[row1, col1] * dc       * dr
    )

    # Normalise to [0, 1] to keep the same scale as the paper
    rmax = rewards.max()
    if rmax > 0:
        rewards /= rmax
    else:
        rewards = np.ones(len(p))

    return rewards.astype(float)


def _pf_velocity(
    ksi: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        ksi: (N, 2) current drone positions in metres
        r: (M,)  current GRBF reward values
        p: (M, 2) GRBF centre positions in metres
        delta: float  GRBF width parameter (m^2)

    Returns:
        v_cmds: (N, 2) velocity command for each drone [m/s]
        d_sq: (N, M) squared distance from each drone to each GRBF centre
    """
    N = ksi.shape[0]

    #Search Gradient
    
    #Shape: (N, M, 2) -> Difference vector from every drone to every GRBF center
    diffs = ksi[:, np.newaxis, :] - p[np.newaxis, :, :]

    #Shape: (N, M) -> Squared distance from every drone to every GRBF center
    d_sq = np.sum(diffs ** 2, axis=2)

    #Shape: (N, M) -> Weight matrix
    weights = r * np.exp(-d_sq / delta)

    #Shape: (N, 2) -> Summed search gradient for each drone
    grad_s = np.sum(weights[:, :, np.newaxis] * diffs, axis=1)

    #Collision Avoidance Gradient
    
    #Shape: (N, N, 2) -> Difference vector between every pair of drones
    drone_diffs = ksi[:, np.newaxis, :] - ksi[np.newaxis, :, :]

    #Shape: (N, N) -> Squared distance between every pair of drones
    drone_dist_sq = np.sum(drone_diffs ** 2, axis=2) + 1e-6

    #Shape: (N, N) -> Force magnitude (1 / ||diff||^4)
    force_mag = 1.0 / (drone_dist_sq ** 2)
    #So drone does not repel itself
    np.fill_diagonal(force_mag, 0.0)

    #Shape: (N, 2) -> Summed collision gradient for each drone
    grad_c = np.sum(force_mag[:, :, np.newaxis] * drone_diffs, axis=1)
    
    #Calculate magnitudes
    mag_s = np.linalg.norm((PF_W_S / delta) * grad_s, axis=1)
    mag_c = np.linalg.norm(PF_W_C * grad_c, axis=1)
    
    #Only see the initial state
    if not hasattr(_pf_velocity, "has_printed"):
        print("\n--- PF DIAGNOSTICS (STEP 0) ---")
        print(f"Search Force Magnitudes: {mag_s}")
        print(f"Collision Force Magnitudes: {mag_c}")
        print(f"Max Reward on Map: {np.max(r)}")
        print("-------------------------------\n")
        _pf_velocity.has_printed = True

    #Combined potential gradient
        
    dV = (PF_W_S / delta) * grad_s - (PF_W_C * grad_c)
    v_cmds = -PF_K * dV

    #Speed saturation
    speeds = np.linalg.norm(v_cmds, axis=1)
    overspeed_mask = speeds > PF_V_MAX
    if np.any(overspeed_mask):
        v_cmds[overspeed_mask] = (
            v_cmds[overspeed_mask] / speeds[overspeed_mask, np.newaxis]
        ) * PF_V_MAX

    return v_cmds, d_sq


def _update_r_base(r_base: np.ndarray, d_sq: np.ndarray, mu_sq: float, dt: float) -> np.ndarray:
    #bounded between 0.0 and 1.0
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    
    #Normal growth and drain
    r_dot = PF_K_R_PLUS * (1.0 - alpha) - (PF_K_R_MINUS * alpha * r_base)
    
    #Clip to prevent infinite growth
    return np.clip(r_base + r_dot * dt, 0.0, 1.0)

def _update_r_spike(r_spike: np.ndarray, d_sq: np.ndarray, mu_sq: float, dt: float) -> np.ndarray:
    min_d_sq = np.min(d_sq, axis=0)
    alpha = (min_d_sq <= mu_sq).astype(float)
    
    #Only drain K_R_MINUS
    r_dot = - (PF_K_R_MINUS * alpha * r_spike)
    
    #Bound to 0 at the bottom, no upper limit
    return np.maximum(0.0, r_spike + r_dot * dt)


def _apply_discovery_spike(
    r_spike: np.ndarray,
    p: np.ndarray,
    discovery_xy: np.ndarray,
    spike_radius_sq: float,
    spike_amplitude: float,
) -> np.ndarray:
    """
    Adds a Gaussian reward pulse to all GRBF centres near a discovery site. The boost decays naturally via the K_R_MINUS drain in _update_rewards:

    Args:
        r: (M,) current reward vector — MODIFIED IN PLACE and returned.
        p: (M, 2) GRBF centre positions in metres.
        discovery_xy: (2,) position of the discovered victim in metres.
        spike_radius_sq: Gaussian width for the spike (metres^2). Spike strength tapers towards radius.
        spike_amplitude: peak amplitude of the reward boost
    Returns:
        r : (M,) reward vector with the spike added.
    """
    #Squared distance from every GRBF centre to the discovery point
    diff = p - discovery_xy[np.newaxis, :]        
    dist_sq = np.sum(diff ** 2, axis=1)             

    #Pulse, peaks at 1 directly on top of the discovery site, then tapers.
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
    """
    Args:
        center_x: IPP easting in projected metres (UTM).
        center_y: IPP northing in projected metres (UTM).
        max_radius: Search circle radius in metres.
        probability_map: (H, W) float array of prior lost-person probabilities.
        bounds: (minx, miny, maxx, maxy) of the probability_map in m.
        **kwargs

    Returns:
        A list of num_drones Shapely LineString objects per drone
    """
    num_drones = int(kwargs.get('num_drones', 3))
    budget_m = float(kwargs.get('budget', BUDGET_M))
    fov_deg= kwargs.get('fov_deg')
    altitude = kwargs.get('altitude')

    victim_locations_raw = kwargs.get('victim_locations', None)

    mu = _compute_mu(fov_deg, altitude)

#   1.5 * mu so sensor footprints overlap smoothly
    target_spacing = mu * 1.5

    #How many grid points span diameter at that spacing
    dynamic_grid_res = int(np.ceil((2 * max_radius) / target_spacing)) + 1

    #Build the grid of GRBFs
    p = _build_grbf_grid(center_x, center_y, max_radius, dynamic_grid_res)

    grid_spacing = (2 * max_radius) / (dynamic_grid_res - 1)

    if PF_DELTA_DEFAULT_M is None:
        delta = (grid_spacing * 1.0) ** 2
    else:
        delta = PF_DELTA_DEFAULT_M ** 2

    log.info(
        f"PF generator: {len(p)} GRBF centres, "
        f"grid_spacing={grid_spacing:.1f} m, delta={delta:.1f} m², mu={mu:.1f} m"
    )

    r_base = _initialise_rewards(p, probability_map, bounds, USE_PROBABILITY_MAP)
    r_spike = np.zeros_like(r_base)

    do_spike = USE_DISCOVERY_SPIKE and victim_locations_raw is not None

    if do_spike:
        #Convert to (V, 2) array
        victims_arr = np.array(victim_locations_raw, dtype=float)
        if victims_arr.ndim == 1:
            victims_arr = victims_arr.reshape(-1, 2)

        #One bool flag per victim
        victim_discovered = np.zeros(len(victims_arr), dtype=bool)

        #Width of spike
        spike_r = (SPIKE_RADIUS_M if SPIKE_RADIUS_M is not None else 3.0 * mu)
        spike_radius_sq = spike_r ** 2

        log.info(
            f"PF discovery spike ENABLED: {len(victims_arr)} victim locations, "
            f"amplitude={SPIKE_AMPLITUDE}, spike_radius={spike_r:.1f} m"
        )
    else:
        victims_arr = None
        victim_discovered = None
        spike_radius_sq  = None
        if USE_DISCOVERY_SPIKE and victim_locations_raw is None:
            log.warning(
                "USE_DISCOVERY_SPIKE=True but no 'victim_locations' were passed "
            )

    init_radius = 5.0
    ksi = np.array([
        [
            center_x + init_radius * np.cos(2 * np.pi * i / num_drones),
            center_y + init_radius * np.sin(2 * np.pi * i / num_drones),
        ]
        for i in range(num_drones)
    ], dtype=float)

    #Determine simulation length from budget
    budget_per_drone = budget_m / num_drones
    max_steps = int(np.ceil(budget_per_drone / (PF_V_MAX * PF_DT)))

    log.info(
        f"PF generator: team_budget={budget_m:.0f} m, per_drone={budget_per_drone:.0f} m, "
        f"v_max={PF_V_MAX} m/s, dt={PF_DT} s -> max_steps={max_steps}"
    )

    #Less trajectory recording to manage linestring size
    record_every = max(1, int(np.ceil(mu / (PF_V_MAX * PF_DT))))
    trajectories = [[list(ksi[i])] for i in range(num_drones)]

    #Pre-compute constants
    center_arr = np.array([center_x, center_y])
    mu_sq      = mu ** 2

    
    for step in range(max_steps):
            r_total = r_base + r_spike
            
            #Calculate velocities using combined reward
            velocities, d_sq = _pf_velocity(ksi, r_total, p, delta)
            ksi += velocities * PF_DT

            #Discovery spike check
            if do_spike:
                victim_diff = ksi[:, np.newaxis, :] - victims_arr[np.newaxis, :, :]
                drone_to_victim_sq = np.sum(victim_diff ** 2, axis=2)
                min_drone_dist_sq = np.min(drone_to_victim_sq, axis=0)
                newly_found = (~victim_discovered) & (min_drone_dist_sq <= mu_sq)

                if np.any(newly_found):
                    for vi in np.where(newly_found)[0]:
                        # Apply spike to r_spike only
                        r_spike = _apply_discovery_spike(
                            r_spike, p, victims_arr[vi], spike_radius_sq, SPIKE_AMPLITUDE
                        )
                        victim_discovered[vi] = True

            r_base = _update_r_base(r_base, d_sq, mu_sq, PF_DT)
            r_spike = _update_r_spike(r_spike, d_sq, mu_sq, PF_DT)

            #Record keeping
            if (step + 1) % record_every == 0:
                for i in range(num_drones):
                    trajectories[i].append(list(ksi[i]))

    paths = []
    for i, traj in enumerate(trajectories):
        if len(traj) >= 2:
            paths.append(LineString(traj))
        else:
            log.warning(f"PF generator: drone {i} produced a degenerate path.")
            paths.append(LineString())

    return paths


if __name__ == "__main__":
    log.info("=== Potential Field Algorithm Evaluation in SAREnv ===")

    config = PathGeneratorConfig(
        num_drones = NUM_DRONES,
        budget = BUDGET_M,
        fov_degrees = 45.0,
        altitude_meters = 80.0,
        overlap_ratio = 0.0,
        path_point_spacing_m = 10.0,
        transition_distance_m = 50.0,
        pizza_border_gap_m = 15.0,
    )

    path_cache = {}

    global_victim_locations = {}

    def get_cached_func(name, func):
        #Wraps generator func to cache output
        @wraps(func)
        def wrapper(*args, **kwargs):
            #Extract arguments
            center_x = args[0] if len(args) > 0 else kwargs.get('center_x')
            max_radius = args[2] if len(args) > 2 else kwargs.get('max_radius')
            
            #Inject victim locations into kwargs
            if name == "PotentialField" and max_radius is not None:
                radius_key = round(max_radius)
                if radius_key in global_victim_locations:
                    kwargs['victim_locations'] = global_victim_locations[radius_key]

            key = (name, center_x)
            if key not in path_cache:
                log.info(f"Computing {name} paths (cached)...")
                path_cache[key] = func(*args, **kwargs)
            return path_cache[key]
        return wrapper

#Paths used in evaluation; can add others
    path_generators = {
        "PotentialField": PathGenerator(
            name="PotentialField",
            func=get_cached_func("PotentialField", generate_pf_path),
            path_generator_config=config,
            description="Cooper (2020) Potential Field guidance with discovery spike"
        ),
        "Greedy": PathGenerator(
            name="Greedy",
            func=get_cached_func("Greedy", sarenv.analytics.paths.generate_greedy_path),
            path_generator_config=config,
            description="SAREnv greedy"
        ),
        "Concentric": PathGenerator(
            name="Concentric",
            func=get_cached_func("Concentric", sarenv.analytics.paths.generate_concentric_circles_path),
            path_generator_config=config,
            description="SAREnv concentric circles"
        ),
    }

#Initialise evaluator
    evaluator = ComparativeEvaluator(
        dataset_directory = DATA_DIR,
        evaluation_sizes = EVALUATION_SIZES,
        num_drones = NUM_DRONES,
        num_lost_persons = NUM_LOST_PERSONS,
        budget = BUDGET_M,
        path_generators = path_generators,
        path_generator_config = config,
    )

    #Injectlost persons into the evaluator.

    log.info(f"Injecting Clustered Lost Persons ({NUM_CLUSTERS} clusters)")
    person_per_cluster = NUM_LOST_PERSONS // NUM_CLUSTERS

    for size, env_data in evaluator.environments.items():
        item = env_data["item"]

        cluster_generator = ClusterLostPersonLocationGenerator(item)

        #Generate cluster centres
        locations = cluster_generator.generate_locations(NUM_CLUSTERS, 0)

        #Generate lost-person coordinates around centres
        lp_locs = cluster_generator.generate_cluster_LPs(
            locations,
            NUM_CLUSTERS,
            person_per_cluster,
        )

        if lp_locs:
            env_data["victims"] = gpd.GeoDataFrame(
                geometry=lp_locs, crs=item.features.crs
            )
            
            #Save locations to global dict
            radius_key = round(item.radius_km * 1000)
            global_victim_locations[radius_key] = [(p.x, p.y) for p in lp_locs]

            log.info(
                f"Injected {len(lp_locs)} clustered victims "
            )
        else:
            log.warning(
                f"Failed to generate clustered victims for '{size}'. "
                f"Falling back to defaults."
            )
            
    log.info("Running evaluations")
    results_df, time_series_data = evaluator.run_baseline_evaluations()

#Print summary
    if not results_df.empty:
        log.info("\n=== RESULTS SUMMARY ===")
        cols = [
            "Algorithm",
            "Dataset",
            "Likelihood Score",
            "Time-Discounted Score",
            "Victims Found (%)",
            "Area Covered (km²)",
            "Total Path Length (km)",
        ]
        print(results_df[cols].to_string(index=False))
    else:
        log.warning("No results produced. Check dataset path")

#Comparison plots
    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)
    evaluator.plot_results(results_df, output_dir=str(output_dir))
    log.info(f"Comparison plots saved to: {output_dir}/")

#Path plotting
    log.info("--- Generating Path Heatmap Visualizations ---")
    heatmap_dir = Path("graphs/heatmaps")
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for size, env_data in evaluator.environments.items():
        item        = env_data["item"]
        victims_gdf = env_data["victims"]

        # Convert to the projected CRS
        center_proj = (
            gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
            .to_crs(env_data["crs"])
            .geometry.iloc[0]
        )

        for name, generator in evaluator.path_generators.items():
            log.info(f"Generating heatmap plot for {name} on '{size}' dataset...")

            generated_paths = generator(
                center_x = center_proj.x,
                center_y = center_proj.y,
                max_radius = item.radius_km * 1000,
                probability_map = item.heatmap,
                bounds = item.bounds,
            )

            output_file = heatmap_dir / f"{name}_{size}_heatmap.pdf"

            plot.plot_heatmap(
                item = item,
                generated_paths = generated_paths,
                name = name,
                x_min = item.bounds[0],
                x_max = item.bounds[2],
                y_min = item.bounds[1],
                y_max = item.bounds[3],
                output_file = str(output_file),
            )

            log.info(f"Saved heatmap plot: {output_file}")

    log.info("--- Path Heatmap Visualization Complete ---")
    log.info("=== Evaluation complete ===")
