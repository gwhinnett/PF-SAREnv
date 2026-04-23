#Adapted from sarenv/core/lost_person.py



"""
Generates plausible lost_person locations based on geographic features.
"""
import random
import geopandas as gpd
from shapely.geometry import Point, Polygon

from sarenv.core.loading import SARDatasetItem
from sarenv.utils.logging_setup import get_logger

log = get_logger()

class ClusterLostPersonLocationGenerator:
    """
    Generates plausible lost_person locations based on geographic features.
    """
    def __init__(self, dataset_item: SARDatasetItem):
        self.dataset_item = dataset_item
        self.features = dataset_item.features.copy()
        self.type_probabilities = {}
        self._calculate_weights()

    def _calculate_weights(self):
        if self.features.empty or 'area_probability' not in self.features.columns:
            log.warning("Features are empty or missing 'area_probability'. Cannot calculate weights.")
            return

        type_weights = self.features.groupby('feature_type')['area_probability'].sum()
        self.type_probabilities = type_weights.to_dict()
        log.info(f"Calculated feature type probabilities: {self.type_probabilities}")

    def _generate_random_point_in_polygon(self, poly: Polygon) -> Point:
        min_x, min_y, max_x, max_y = poly.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if poly.contains(random_point):
                return random_point

    #Now generates locations for 'cluster centres'.
    def generate_locations(self, n: int = 1, percent_random_samples=0) -> list[Point]:
        """
        Generate multiple plausible lost_person locations.

        Args:
            n (int): Number of locations to generate.

        Returns:
            List of shapely.geometry.Point objects (may be fewer than n if not enough valid locations).
        """
        if not self.type_probabilities:
            log.error("No feature probabilities available. Cannot generate locations.")
            return []

        locations = []

        center_proj = gpd.GeoDataFrame(
            geometry=[Point(self.dataset_item.center_point)],
            crs="EPSG:4326"
        ).to_crs(self.features.crs).geometry.iloc[0]
        main_search_circle = center_proj.buffer(self.dataset_item.radius_km * 1000)

        while len(locations) < n:
            # Randomly choose a feature type based on the probabilities
            if random.random() < percent_random_samples:  # 10% chance to choose a random type
                chosen_feature = self.features.sample(n=1).iloc[0]
                feature_buffer = chosen_feature.geometry.buffer(15)
                final_search_area = feature_buffer.intersection(main_search_circle)
            else:
                chosen_type = random.choices(
                    list(self.type_probabilities.keys()),
                    weights=list(self.type_probabilities.values()),
                    k=1
                )[0]
                type_gdf = self.features[self.features['feature_type'] == chosen_type]
                chosen_feature = type_gdf.sample(n=1, weights='area_probability').iloc[0]
                feature_buffer = chosen_feature.geometry.buffer(15)
                final_search_area = feature_buffer.intersection(main_search_circle)


            point = self._generate_random_point_in_polygon(final_search_area)
            if point:
                locations.append(point)
        if len(locations) < n:
            log.warning(f"Only generated {len(locations)} out of {n} requested locations.")

        return locations


    def generate_cluster_LPs(self, locations, num_clusters, persons_per_cluster):
        print("Cluster locations:", locations)

        LP_locs = []

        for i in range(len(locations)):
            cluster_point = locations[i]
            local_buffer = cluster_point.buffer(1000) #1000m buffer radius
            
            subiter = 1

            while subiter <= persons_per_cluster:
                random_point = self._generate_random_point_in_polygon(local_buffer)
                LP_locs.append(random_point)
                subiter+=1
        print("Generated:", len(LP_locs))
        return LP_locs


