#George's clustering mk1 -  adapted from plot.py

import os
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import stats
from shapely.geometry import Point

from sarenv.utils.logging_setup import get_logger
from sarenv.core.loading import SARDatasetItem
from sarenv.core.cluster_lost_person import ClusterLostPersonLocationGenerator
from sarenv.utils.geo import get_utm_epsg
from sarenv.utils.lost_person_behavior import get_environment_radius

log = get_logger()

FEATURE_COLOR_MAP = {
    # --- Infrastructure / Man-made ---
    # Greys and browns for concrete, metal, and wood structures.
    "structure": '#636363',  # Dark Grey (e.g., buildings)
    "road": '#bdbdbd',       # Light Grey (e.g., roads, paths)
    "linear": '#8B4513',     # Saddle Brown (e.g., fences, railways, pipelines)

    # --- Water Features ---
    # Blues for all water-related elements.
    "water": '#3182bd',      # Strong Blue (e.g., lakes, rivers)
    "drainage": '#9ecae1',   # Light Blue (e.g., ditches, canals)

    # --- Vegetation ---
    # Greens and yellows for different types of plant life.
    "woodland": '#31a354',   # Forest Green (e.g., forests)
    "scrub": '#78c679',      # Muted Green (e.g., scrubland)
    "brush": '#c2e699',      # Very Light Green (e.g., grass)
    "field": '#fee08b',      # Golden Yellow (e.g., farmland, meadows)
    
    # --- Natural Terrain ---
    # Earth tones for rock and soil.
    "rock": '#969696',       # Stony Grey (e.g., cliffs, bare rock)
}
DEFAULT_COLOR = '#f0f0f0' # A very light, neutral default color.

# Color palette for plotting multiple paths
COLORS_BLUE = [
    '#08519c',  # Dark blue
    '#3182bd',  # Medium blue
    '#6baed6',  # Light blue
    '#9ecae1',  # Very light blue
    '#c6dbef',  # Pale blue
]

def visualize_clusters(item: SARDatasetItem, plot_basemap: bool = False, plot_inset: bool = False, num_clusters: int = 0, plot_show=True, num_lost_persons: int = 0):
    """
    Creates a plot with a circular, magnified callout of the "medium" radius.

    Args:
        item (SARDatasetItem): The loaded dataset item to visualize.
        plot_basemap (bool): Whether to plot the basemap.
        plot_inset (bool): Whether to plot the inset.
        sample_lost_persons (bool): Whether to sample and plot lost person locations.
        num_clusters (int): Number of clusters to generate.
    """
    personPerCluster = num_lost_persons // num_clusters if num_clusters > 0 else 0
    LP_locs = []
    
    if not item:
        log.warning("No dataset items provided to visualize.")
        return

    radii = get_environment_radius(item.environment_type, item.environment_climate)

    log.info(f"Generating nested visualization for '{item.size}' with circular magnification...")
    center_point_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    )
    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    fig, ax = plt.subplots(figsize=(18, 15))

    feature_legend_handles = []
    # Set zorder for features to 1
    for feature_type, data in item.features.groupby("feature_type"):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
        data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.7, zorder=1)
        feature_legend_handles.append(Patch(color=color, label=feature_type.capitalize()))

    if plot_basemap:
        # The basemap will have a zorder of 0 by default
        cx.add_basemap(ax, crs=item.features.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
    
    # Sample and plot lost person locations if requested
    radii_legend_handles = []
    lost_person_gdf = None
    cluster_gdf = None
    if num_clusters > 0:
        log.info(f"Generating {num_clusters} cluster locations...")
        cluster_generator = ClusterLostPersonLocationGenerator(item)
        locations = cluster_generator.generate_locations(num_clusters, 0) # 0% random samples
        
        # --- Bounding Circle Logic ---
        # Define the bounding circle using the max radius (Extra Large)
        max_radius_m = radii[-1] * 1000
        bounding_circle_geom = center_point_proj.buffer(max_radius_m).iloc[0]
        
        LPLocs = []
        if locations:
            for loc in locations:
                valid_cluster_lps = []
                
                # Continuously generate points until we meet the required quota for this cluster
                while len(valid_cluster_lps) < personPerCluster:
                    needed = personPerCluster - len(valid_cluster_lps)
                    candidates = cluster_generator.generate_cluster_LPs([loc], 1, needed)
                    
                    # Convert to projected CRS for accurate geometric boundary checking
                    candidates_gdf = gpd.GeoDataFrame(geometry=candidates, crs=item.features.crs)
                    candidates_proj = candidates_gdf.to_crs(data_crs)
                    
                    for orig_pt, proj_pt in zip(candidates, candidates_proj.geometry):
                        # Only keep the point if it falls within the bounding circle
                        if bounding_circle_geom.contains(proj_pt):
                            valid_cluster_lps.append(orig_pt)
                            if len(valid_cluster_lps) == personPerCluster:
                                break
                                
                LPLocs.extend(valid_cluster_lps)

            cluster_gdf = gpd.GeoDataFrame(geometry=locations, crs=item.features.crs)
            LP_gdf = gpd.GeoDataFrame(geometry=LPLocs, crs=item.features.crs)
            LP_gdf.plot(ax=ax, marker='*', color='red', markersize=400, zorder=2, label="Lost Person")
            cluster_gdf.plot(ax=ax, marker='X', color='blue', markersize=400, zorder=1, label="Cluster")
            radii_legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Lost Person'))
            radii_legend_handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=15, label='Incident Cluster'))

    colors = ["blue", "orange", "red", "green"]
    labels = ["Small", "Medium", "Large", "Extra Large"]
    # Set zorder for rings to 2 to be on top of features
    for idx, r in enumerate(radii):
        circle = center_point_proj.buffer(r * 1000).iloc[0]
        color = colors[idx % len(colors)]
        gpd.GeoSeries([circle], crs=data_crs).boundary.plot(
            ax=ax, edgecolor=color, linestyle="--", linewidth=2, alpha=1, zorder=2)
        label = f"{labels[idx]} ({r} km)"
        radii_legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5, linestyle="--", label=label)
        )

    # Create two separate legends
    # Features legend (upper left)
    features_legend = ax.legend(handles=feature_legend_handles, title="Features", 
                               loc="upper left", fontsize=16, title_fontsize=18)
    
    # Radii and lost person legend (upper right)
    radii_legend = ax.legend(handles=radii_legend_handles, title="RoIs", 
                            loc="upper right", fontsize=16, title_fontsize=18)
    
    # Add the features legend back since the second legend call removes the first
    ax.add_artist(features_legend)
    ax.add_artist(radii_legend)

    if plot_inset:
        medium_radius_idx = 1
        medium_radius_m = radii[medium_radius_idx] * 1000

        inset_width = 0.60
        inset_height = 0.60
        ax_inset = ax.inset_axes([1.05, 0.5 - (inset_height / 2), inset_width, inset_height])

        # Create clipping circle for the inset
        center_x = center_point_proj.geometry.x.iloc[0]
        center_y = center_point_proj.geometry.y.iloc[0]
        clipping_circle = Point(center_x, center_y).buffer(medium_radius_m)
        clipping_gdf = gpd.GeoDataFrame([1], geometry=[clipping_circle], crs=data_crs)

        # Ensure features are in the same CRS as the clipping circle for proper clipping
        features_proj = item.features.to_crs(data_crs)
        clipped_features = gpd.clip(features_proj, clipping_gdf)

        # Plot the CLIPPED features on the inset axes
        if plot_basemap:
             cx.add_basemap(ax_inset, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik)
        
        if not clipped_features.empty:
            for feature_type, data in clipped_features.groupby("feature_type"):
                data.plot(ax=ax_inset, color=FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR), alpha=0.7)
        
        for idx, r in enumerate(radii):
            circle_geom = center_point_proj.buffer(r * 1000).iloc[0]
            gpd.GeoSeries([circle_geom], crs=data_crs).boundary.plot(
                ax=ax_inset, edgecolor=colors[idx % len(colors)], linestyle="--", linewidth=2, alpha=1
            )

        # Add lost person locations to inset if they were generated
        if num_clusters>0 and cluster_gdf is not None:
            # Ensure lost person data is in the same CRS as the clipping circle
            cluster_proj = cluster_gdf.to_crs(data_crs)
            lost_person_proj = LP_gdf.to_crs(data_crs)
            
            clipped_cluster = gpd.clip(cluster_proj, clipping_gdf)
            clipped_lost_person = gpd.clip(lost_person_proj, clipping_gdf)
            
            if not clipped_cluster.empty:
                clipped_cluster.plot(ax=ax_inset, marker='X', color='blue', markersize=250, zorder=3)
                clipped_lost_person.plot(ax=ax_inset, marker='*', color='red', markersize=250, zorder=3)

        ax_inset.set_xlim(center_x - medium_radius_m, center_x + medium_radius_m)
        ax_inset.set_ylim(center_y - medium_radius_m, center_y + medium_radius_m)

        circle = Circle((0.5, 0.5), 0.5, transform=ax_inset.transAxes, facecolor='none', edgecolor='black', linewidth=1)
        ax_inset.set_clip_path(circle)
        ax_inset.patch.set_alpha(0.0)

        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        # Set zorder for connector lines to 1 to be under the main rings
        pp1, c1, c2 = mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp1.set_zorder(0)
        c1.set_zorder(0)
        c2.set_zorder(0)

        pp2, c3, c4 = mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp2.set_zorder(0)
        c3.set_zorder(0)
        c4.set_zorder(0)
        
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels([f"{x/1000:.1f}" for x in x_ticks], fontsize=18)
    ax.set_yticklabels([f"{y/1000:.1f}" for y in y_ticks], fontsize=18)
    ax.set_xlabel("Easting (km)", fontsize=22)
    ax.set_ylabel("Northing (km)", fontsize=22)
    

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"features_{item.size}_circular_magnified_final.pdf", bbox_inches='tight')
    if plot_show:
        plt.show()
