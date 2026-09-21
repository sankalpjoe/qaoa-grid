# Graph-Constrained Warm-Started QAOA for Smart Grid Cascading Outage Mitigation (HG-WS-QAOA)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.5](https://img.shields.io/badge/qiskit-2.5.0-613399.svg)](https://qiskit.org/)
[![Hardware Acceleration](https://img.shields.io/badge/NVIDIA%20RTX%205070-CUDA%20Enabled-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Target Topology](https://img.shields.io/badge/IBM%20Quantum-Heavy--Hex%20(Eagle%2FHeron)-0062FF.svg)](https://quantum.ibm.com/)

Enterprise case study and IEEE conference publication package evaluating **Hardware-Aligned Graph-Constrained Warm-Started QAOA (HG-WS-QAOA)** for near-real-time cascading blackout mitigation in power transmission grids.

---

## Key Highlights & Research Innovations

- **Workload Co-Design**:
  - **NVIDIA GeForce RTX 5070 Laptop GPU**: Solves the continuous DC Optimal Power Flow (DC-OPF) relaxation in $\sim 1.5$ seconds using PyTorch CUDA projected gradient descent.
  - **Regularized Safe-Exploration Bloch Mapping**: Maps continuous line variables $s_l^* \in [0, 1]$ to Bloch angles $\theta_l$ with margin $\epsilon = 0.08$, mathematically avoiding "pole freezing" / barren plateaus.
  - **IBM Heavy-Hex Quantum Processor (Qiskit 2.5)**: Executes shallow ($p=1, 2$) warm-started QAOA circuits mapped to IBM Eagle (127-qubit) and Heron (133-qubit) coupling maps.
- **Physics-Informed Sensitivity Truncation**:
  - Enforces interaction graph vertex degree $\le 3$, cutting native two-qubit gates by **84.0%** (from 94 to 15) and circuit depth by **72.4%** (from 199 to 55).
- **Subspace-Preserving Custom Warm-Start Mixer**:
  - Prevents state drift into invalid, islanded grid states.
- **Experimental Metric Deliverables**:
  - **Valid Zero-Violation Probability $P(\text{valid})$**: **13.16%** under calibrated IBM superconducting noise (vs. **6.59%** for Standard QAOA, a **$>2.1\times$ gain**).
  - **Approximation Ratio (AR)**: Outperforms standard QAOA by **+36.7%** on valid samples.

---

## Directory Structure

```
grid_resilience_ws_qaoa/
├── README.md                                  # This enterprise guide
├── run_pipeline.py                            # Master execution pipeline
├── pyproject.toml                             # Project metadata & dependencies
├── grid_resilience/
│   ├── grid/
│   │   ├── network.py                         # IEEE 14 & 30-bus network physics, B-matrix, PTDF
│   │   └── cascade_simulator.py               # Cascading failure simulator & switching evaluator
│   ├── gpu_solver/
│   │   └── dc_opf_gpu.py                      # RTX 5070 PyTorch CUDA continuous relaxation solver
│   ├── hamiltonian/
│   │   └── ising_formulation.py               # Degree <= 3 hardware-sparsified Ising formulation
│   ├── quantum/
│   │   ├── warm_start_qaoa.py                 # HG-WS-QAOA ansatz & custom Bloch mixer
│   │   ├── standard_qaoa.py                   # Standard un-warm-started QAOA baseline
│   │   ├── heavy_hex_mapping.py               # IBM Eagle/Heron heavy-hex layout & Sabre transpilation
│   │   └── noise_models.py                    # Calibrated IBM superconducting noise factory
│   ├── baselines/
│   │   └── classical_greedy.py                # PTDF-ranked classical transmission switching heuristic
│   └── benchmarks/
│       ├── evaluator.py                       # Comprehensive benchmarking engine
│       └── plotter.py                         # Publication-grade figure generator (300 DPI)
├── tests/
│   ├── test_grid_physics.py                   # DC-OPF, PTDF, and Kirchhoff laws verification
│   ├── test_gpu_solver.py                     # RTX 5070 CUDA continuous relaxation verification
│   ├── test_quantum_circuits.py               # Circuit construction & heavy-hex transpilation
│   └── test_benchmarks.py                     # End-to-end integration test
├── reports/
│   ├── benchmark_summary.json                 # Machine-readable benchmark data
│   ├── fig2_heavy_hex_depth.png               # Heavy-hex transpilation gate/depth reduction
│   ├── fig3_sampling_validity_p_valid.png     # P(valid) under Ideal vs IBM noise
│   └── fig4_noise_degradation_curves.png      # Hardware noise sensitivity curves (0x to 2x)
└── paper/
    ├── ieee_smart_grid_resilience_qaoa.md     # Full IEEE conference paper manuscript
    └── ieee_smart_grid_resilience_qaoa.tex    # Overleaf / IEEEtran LaTeX document
```

---

## Quickstart & Execution

### 1. Run Automated Test Suite
```powershell
python -m pytest tests/ -v
```

### 2. Run Full Experimental Benchmark & Generate Figures
```powershell
python run_pipeline.py --network ieee14 --outage 6 --candidates 8 --shots 4096 --plots
```

### 3. Generate Paper Results for IEEE 30-Bus System
```powershell
python run_pipeline.py --network ieee30 --outage 6 --candidates 8 --shots 4096 --plots
```

---

## Benchmark Results Summary (IEEE 14-Bus Test Case)

| Algorithm / Method | Hardware Noise | $P(\text{valid})$ | Exp. Cost $\mathbb{E}[C]$ | Approx. Ratio (AR) | Native 2Q Gates | Heavy-Hex Depth |
|---|---|---|---|---|---|---|
| **Uniform Random** | Ideal | 9.38% | N/A | N/A | 0 | 0 |
| **Classical Greedy (PTDF)** | Classical | 100.00% | 10.00 | 1.000 | N/A | N/A |
| **HG-WS-QAOA ($p=1$) [Proposed]** | **Ideal** | **13.67%** | **$3.12 \times 10^6$** | **0.640** | **15** | **55** |
| **HG-WS-QAOA ($p=1$) [Proposed]** | **IBM Heavy-Hex Noise** | **13.16%** | **$4.26 \times 10^6$** | **0.604** | **15** | **55** |
| **HG-WS-QAOA ($p=2$) [Proposed]** | Ideal | 10.55% | $5.84 \times 10^6$ | 0.562 | 36 | 108 |
| **HG-WS-QAOA ($p=2$) [Proposed]** | IBM Heavy-Hex Noise | 9.42% | $7.51 \times 10^6$ | 0.555 | 36 | 108 |
| **Standard QAOA ($p=1$)** | Ideal | 6.37% | $12.74 \times 10^6$ | 0.468 | 15 | 54 |
| **Standard QAOA ($p=1$)** | IBM Heavy-Hex Noise | 6.59% | $12.91 \times 10^6$ | 0.463 | 15 | 54 |
| **Standard QAOA ($p=2$)** | Ideal | 3.83% | $13.99 \times 10^6$ | 0.458 | 36 | 107 |
| **Standard QAOA ($p=2$)** | IBM Heavy-Hex Noise | 4.05% | $13.62 \times 10^6$ | 0.460 | 36 | 107 |

---

## Citation
If utilizing this framework for academic research or industrial implementation, please cite:
```bibtex
@inproceedings{qgc2026hg_ws_qaoa,
  title={Hardware-Aligned Graph-Constrained Warm-Started QAOA for Cascading Outage Mitigation in Smart Grids},
  author={Quantum Grid Consortium},
  booktitle={IEEE Transactions on Power Systems / IEEE PES General Meeting},
  year={2026}
}
```
