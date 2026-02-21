# SAREnv: UAV Search and Rescue Dataset and Evaluation Framework

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0-green.svg)](https://github.com/namurproject/sarenv)

SAREnv is an open-access dataset and evaluation framework designed to support research in UAV-based search and rescue (SAR) algorithms. This toolkit addresses the critical need for standardized datasets and benchmarks in wilderness SAR operations, enabling systematic evaluation and comparison of algorithmic approaches including coverage path planning, probabilistic search, and information-theoretic exploration.

## 🎯 Project Goals

Unmanned Aerial Vehicles (UAVs) play an increasingly vital role in wilderness search and rescue operations by enhancing situational awareness and extending the reach of human teams. However, the absence of standardized datasets and benchmarks has hindered systematic evaluation and comparison of UAV-based SAR algorithms. SAREnv bridges this gap by providing:

- **Realistic geospatial scenarios** across diverse terrain types
- **Synthetic victim locations** derived from statistical models of lost person behavior
- **Comprehensive evaluation metrics** for search trajectory assessment
- **Baseline planners** for reproducible algorithm comparisons
- **Extensible framework** for custom algorithm development

## 🌟 Key Features

### 📊 Dataset Generation

- **Multi-scale environments**: Small, medium, large, and extra-large search areas
- **Diverse terrain types**: Flat and mountainous environments
- **Climate variations**: Temperate and dry climate conditions
- **Realistic geospatial features**: Roads, water bodies, vegetation, structures, and terrain features extracted from OpenStreetMap

### 🎯 Lost Person Modeling

- Statistical models based on established lost person behavior research
- Probability heatmaps incorporating environmental factors
- Configurable victim location generation with terrain-aware distributions

### 🚁 Path Planning Algorithms

- **Spiral Coverage**: Efficient outward spiral search patterns
- **Concentric Circles**: Systematic circular search patterns
- **Pizza Zigzag**: Sector-based zigzag coverage
- **Greedy Search**: Probability-driven adaptive search
- **Extensible framework** for custom algorithm integration

### 📈 Evaluation Metrics

- **Coverage metrics**: Area coverage and search efficiency
- **Likelihood scores**: Probability-weighted path evaluation
- **Time-discounted scoring**: Temporal effectiveness assessment
- **Victim detection rates**: Success probability and timeliness analysis
- **Multi-drone coordination**: Support for collaborative search strategies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/sarenv.git
cd sarenv

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Download Pre-generated Dataset

The repository includes pre-generated datasets stored using Git LFS (Large File Storage). To download the data files needed to run the examples:

```bash
# Install Git LFS if not already installed
# On Ubuntu/Debian:
sudo apt-get install git-lfs

# On macOS with Homebrew:
brew install git-lfs

# On Windows, download from: https://git-lfs.github.io/

# Initialize Git LFS in the repository
git lfs install

# Download the dataset files
git lfs pull
```

**Note**: The dataset files are stored in the `sarenv_dataset/` directory and are required to run the examples. If you prefer to generate your own dataset, you can skip this step and use the dataset generation functionality described below.

### Basic Usage

#### 1. Generate Dataset (Optional)

If you prefer to generate your own dataset instead of using the pre-generated data:

```python
import sarenv

# Initialize data generator
generator = sarenv.DataGenerator()

# Generate dataset for different locations and sizes
generator.export_dataset()
```

**Note**: This step is optional if you've already downloaded the pre-generated dataset using Git LFS as described above.

#### 2. Load and Visualize Data

```python
import sarenv

# Load a dataset
loader = sarenv.DatasetLoader("sarenv_dataset")
item = loader.load_environment("large")

# Visualize the environment
from examples.02_load_and_visualize import visualize_heatmap, visualize_features
visualize_heatmap(item)
visualize_features(item)
```

#### 3. Generate Lost Person Locations

```python
import sarenv

# Load environment
loader = sarenv.DatasetLoader("sarenv_dataset")
item = loader.load_environment("medium")

# Generate victim locations
victim_generator = sarenv.LostPersonLocationGenerator(item)
locations = victim_generator.generate_locations(num_locations=100)
```

#### 4. Evaluate Search Algorithms

```python
import sarenv

# Initialize comparative evaluator
evaluator = sarenv.ComparativeEvaluator(
    dataset_directory="sarenv_dataset",
    evaluation_sizes=["medium", "large"],
    num_drones=5,
    num_lost_persons=50
)

# Run baseline evaluations
results = evaluator.run_baseline_evaluations()

# Plot comparative results
evaluator.plot_results(results)
```

## 📁 Example Scripts

The `examples/` directory contains comprehensive scripts demonstrating different aspects of the SAREnv framework. Here's how to use each one:

### `01_generate_sar_data.py`

**Purpose**: Generate custom SAR datasets for specific geographic regions  
**Usage**: Demonstrates how to create datasets from custom polygon areas using real geospatial data

```bash
python examples/01_generate_sar_data.py
```

- Creates datasets for custom geographic polygons
- Downloads and processes OpenStreetMap features
- Generates probability heatmaps for lost person locations
- Exports multi-scale environments (small to extra-large)

### `02_load_and_visualize.py`

**Purpose**: Load existing datasets and create visualizations  
**Usage**: Shows how to load pre-generated datasets and create publication-quality plots

```bash
python examples/02_load_and_visualize.py
```

- Loads datasets from the `sarenv_dataset/` directory
- Creates heatmap visualizations with probability distributions
- Generates feature maps showing terrain, roads, water bodies, and vegetation
- Supports both basemap overlays and standalone visualizations

### `03_generate_survivors.py`

**Purpose**: Generate realistic lost person locations  
**Usage**: Demonstrates statistical modeling of victim locations based on terrain features

```bash
python examples/03_generate_survivors.py
```

- Uses research-based behavioral models for lost person distributions
- Generates probabilistic victim locations considering terrain types
- Creates visualizations showing victim locations overlaid on terrain features
- Supports configurable numbers of victims for different scenario sizes

### `04_evaluate_coverage_paths.py`

**Purpose**: Evaluate and compare path planning algorithms  
**Usage**: Run comparative analysis of built-in search algorithms on a single dataset

```bash
python examples/04_evaluate_coverage_paths.py
```

- Compares Spiral, Concentric, Pizza, and Greedy search algorithms
- Calculates comprehensive performance metrics (coverage, likelihood, detection rates)
- Generates comparison plots and performance charts
- Supports multi-drone scenarios with configurable team sizes

### `05_evaluate_all_datasets.py`

**Purpose**: Large-scale evaluation across multiple datasets  
**Usage**: Systematic evaluation framework for algorithm benchmarking across many scenarios

```bash
python examples/05_evaluate_all_datasets.py --budget 300000 --num_drones 5
```

- Evaluates algorithms across multiple geographic regions (datasets 1-60)
- Provides statistical significance testing across diverse scenarios
- Generates comprehensive CSV results for further analysis
- Supports custom algorithm integration for research purposes
- Command-line arguments for budget and drone configuration

### `06_generate_comparative_coverage_video.py`

**Purpose**: Create animated videos showing algorithm performance  
**Usage**: Generate dynamic visualizations comparing multiple algorithms in real-time

```bash
python examples/06_generate_comparative_coverage_video.py
```

- Creates MP4 videos showing 4 algorithms side-by-side in 2×2 grid layout
- Real-time visualization of drone movement and path building
- Dynamic metrics graphs showing performance evolution over time
- Configurable video quality and frame rates for different use cases
- Efficient parallel processing for faster video generation

### Running Examples

Each script can be run independently after installing SAREnv:

```bash
# Make sure you're in the project directory
cd SAREnv

# Run any example script
python examples/[script_name].py
```

**Note**: Scripts 02-06 require existing datasets. Either download the pre-generated datasets using Git LFS or run script 01 to generate custom datasets first.

## 📁 Repository Structure

```text
sarenv/
├── sarenv/                     # Main package
│   ├── analytics/              # Path planning and evaluation
│   │   ├── paths.py           # Coverage path algorithms
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── evaluator.py       # Comparative evaluation framework
│   ├── core/                   # Core functionality
│   │   ├── generation.py      # Dataset generation
│   │   ├── loading.py         # Dataset loading utilities
│   │   └── lost_person.py     # Lost person modeling
│   └── utils/                  # Utility functions
│       ├── geo.py             # Geospatial utilities
│       ├── plot.py            # Visualization tools
│       └── lost_person_behavior.py  # Behavioral models
├── examples/                   # Usage examples
│   ├── 01_generate_sar_data.py
│   ├── 02_load_and_visualize.py
│   ├── 03_generate_survivors.py
│   ├── 04_evaluate_coverage_paths.py
│   ├── 05_evaluate_all_datasets.py
│   └── 06_generate_comparative_coverage_video.py
├── data/                       # Data processing scripts
└── sarenv_dataset/            # Generated datasets (created after running)
```

## 🔬 Research Applications

### Supported Algorithm Types

- **Coverage Path Planning**: Systematic area coverage strategies
- **Probabilistic Search**: Bayesian and heuristic search methods
- **Information-Theoretic Exploration**: Entropy-based search optimization
- **Multi-Agent Coordination**: Collaborative UAV search strategies

### Evaluation Dimensions

- **Spatial Coverage**: Area covered vs. time efficiency
- **Probability Optimization**: Likelihood-weighted search performance
- **Temporal Dynamics**: Time-sensitive victim detection modeling
- **Resource Utilization**: Multi-drone coordination effectiveness


## 🛠️ Custom Algorithm Integration

Add your own path planning algorithm:

```python
def custom_search_algorithm(center_x, center_y, max_radius, num_drones, **kwargs):
    """
    Custom search algorithm implementation.
    
    Args:
        center_x, center_y: Search center coordinates
        max_radius: Maximum search radius in meters
        num_drones: Number of UAVs
        **kwargs: Additional parameters (fov_deg, altitude, etc.)
    
    Returns:
        list[LineString]: Path for each drone
    """
    # Your algorithm implementation
    paths = []
    # ... algorithm logic ...
    return paths

# Register with evaluator
evaluator = sarenv.ComparativeEvaluator()
evaluator.path_generators['custom'] = custom_search_algorithm
```

## 📃 Publications

* Grøntved, K. A. R., Jarabo-Peñas, A., Reid, S., Rolland, E. G. A., Watson, M., Richards, A., Bullock, S., & Christensen, A. L. (2025). SAREnv: An Open-Source Dataset and Benchmark Tool for Informed Wilderness Search and Rescue Using UAVs. Drones, 9(9), 628. https://doi.org/10.3390/drones9090628

## 📝 Citation

If you use SAREnv in your research, please cite:

```bibtex
@article{sarenv2025,
  title={SAREnv: An Open-Source Dataset and Benchmark Tool for Informed Wilderness Search and Rescue using UAVs},
  author={Kasper Andreas Rømer Grøntved, Alejandro Jarabo-Peñas, Sid Reid, Edouard George Alain Rolland, Matthew Watson, Arthur Richards, Steve Bullock, and Anders Lyhne Christensen},
  journal={Drones},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is supported by; the Innovation Fund Denmark for the DIREC project (9142-00001B), the Independent Research Fund Denmark under grant 10.46540/4264-00105B (the NAMUR project), and by the WildDrone MSCA Doctoral Network funded by EU Horizon Europe under grant agreement no. 101071224


