# SAREnv: UAV Search and Rescue Dataset and Evaluation Framework

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0-green.svg)](https://github.com/namurproject/sarenv)

SAREnv is an open-access dataset and evaluation framework designed to support research in UAV-based search and rescue (SAR) algorithms. This toolkit addresses the critical need for standardized datasets and benchmarks in wilderness SAR operations, enabling systematic evaluation and comparison of algorithmic approaches including coverage path planning, probabilistic search, and information-theoretic exploration.

**# G. Whinnett's Potential Field Algorithm in SAREnv. All credit goes to the original authors.**

All relevant files can be found in the examples folder. My additions: Potential Field algorithm based on Cooper et al.'s paper https://arc.aiaa.org/doi/abs/10.2514/6.2020-0879 with the addition of a Gaussian spike at the point of discovery of a lost person. Changed lost person generation to work in clusters.

Quick start:
```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

Running PF Files, example:
```bash
python examples\visualize_pf_heatmap.py
```



## 📃 SAREnv main publication:

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



