# Robust Allocation and Admission for Simultaneous Cut-Ins

Simulation and analysis code for **Robust Allocation and Admission for Simultaneous Cut-Ins of Multiple Vehicles into Multiple Platoons under Parameter Uncertainty**, by Qiang Wen, Zhengwu Wang, Tao Chen, and Jiazheng Li.

The method combines set-membership parameter identification, certified displacement capacities, joint platoon allocation, and admission checks with future-time plan handover.

## Contents

| Directory | Contents |
| --- | --- |
| `code/sim/` | Vehicle models, robust control, plan execution, metric and initial-scene inputs |
| `code/experiments/` | Registered case configurations, batch runners, audits and statistical analysis |
| `code/baseline_b2/` | Centralized planning baseline and its component tests |
| `code/timing/` | Host timing experiments and timing analysis |
| `code/postprocess/` | Trajectory extraction and lateral-motion diagnostics |
| `code/figures/` | Scripts for Figures 4–10 and the numerical results in Table III |
| `data/baseline_b2/` | Request-state snapshot and acceleration history used to initialize B2 |
| `data/noise_scan/` | Configurations for the supplementary measurement-noise sweep |

The repository contains source code and required inputs. Simulation trajectories, solver logs, generated figures, statistical outputs, and manuscript files are not distributed. The main experiment registries define **1,995 unique runs**; the supplementary noise sweep is provided separately.

## Installation

Use Python 3.11 in a virtual environment. The simulation dependencies match the recorded numerical environment:

```sh
python -m pip install -r requirements.txt
```

For plots, also install:

```sh
python -m pip install -r requirements-figures.txt
```

Plotting requires a TeX Live or MiKTeX installation containing TeX Gyre Termes and `kpsewhich` on `PATH`. The plotting helper converts the installed OpenType fonts to local TrueType files. The fonts retain the GUST Font License and are not bundled in this repository.

For the external B2 planner, install:

```sh
python -m pip install -r requirements-baseline.txt
```

The registered lateral controller uses CVXOPT. The small module in `code/sim/osqp_only/` satisfies an unused import; it raises an error if an OSQP solve is requested.

## Quick start

Run these commands from the repository root:

```sh
python reproduce.py check
python reproduce.py demo
python reproduce.py figures --figures 4
```

The demo executes and audits the nominal main scenario. It saves outputs under `data/recomputed/runs/cb13911c1dadd2de7413/`. Its expected audit label is `C3`, indicating completion without a registered longitudinal violation. Figure 4 is written to `figures/`.

## Registered simulations and analysis

```sh
python reproduce.py simulate --jobs 2
python reproduce.py analyze
python reproduce.py figures
```

`--jobs` controls concurrent cases. Each case retains its registered internal planning-worker setting. Start with a small number of concurrent cases to avoid CPU oversubscription. A full run is substantially longer than the demo; completed cases are reused when the command is repeated.

The analysis stage requires the complete set of run outputs. It recomputes the registered statistics, independently checks trajectory quantities, extracts figure inputs, and evaluates the EX9 lateral-motion diagnostics. The default figure selection is Figures 4–8; Table III is computed alongside Figure 5 and Table IV during analysis. Numerical results depend on the documented model and uncertainty assumptions; adverse cases remain part of the registries.

To run a selected family:

```sh
python code/experiments/run_family.py data/recomputed/runs --family EX4 --jobs 2
python code/experiments/run_family_addA.py data/recomputed/runs --family EX9 --jobs 2
```

For a supplementary noise case:

```sh
python code/experiments/run_case.py data/noise_scan/ell_0002/case.json data/noise_scan/ell_0002
python code/experiments/audit_run.py data/noise_scan/ell_0002
```

Choose one of the supplied `ell_*` directories. Output directories containing an existing `result.json` are protected against accidental replacement.

## B2 planning baseline

```sh
python code/baseline_b2/test_b2_nlp_v6.py
python code/baseline_b2/run_b2v6.py all
```

The test command exercises model components and derivatives without running the optimizer. The full planner uses IPOPT and has a three-hour budget per job. Nine-vehicle and 25-vehicle studies are separate; the full study is conditional on an accepted nine-vehicle plan. Results are written under `code/baseline_b2/runs/`.

The input JSON stores the state at 5 s and the preceding one-second acceleration history. These values were extracted from the main-scenario trajectory; its original SHA-256 is retained in the JSON. Keeping the required history in the input makes B2 independent of the full trajectory archive.

To plot a selected accepted B2 solution, supply its actual path:

```sh
python reproduce.py figures --figures 9 --b2-solution path/to/solution.npz
```

## Timing experiments

The published timing protocol uses four configurations, ten repetitions with 16 planning workers and three repetitions with one worker:

```sh
python code/timing/timing_host.py --src code/sim/sources --out data/timing/host_runs --workers 16 --repeats 10
python code/timing/timing_host.py --src code/sim/sources --out data/timing/host_runs --workers 1 --repeats 3
python code/experiments/ex10t.py data/timing/ex10t --workers 1
python code/timing/recompute_timing.py
python reproduce.py figures --figures 10
```

Timing varies with hardware and system load. Run these experiments without concurrent simulation batches. The raw timing records and regenerated timing tables remain local.

## Integrity and reproducibility

`SHA256SUMS.txt` lists the distributed files. The single-case runners additionally verify the frozen simulation sources and input files against their recorded checksums. Git is configured to preserve file bytes, including line endings, because the checksums are byte-sensitive.

The simulation kernel and case registries retain the numerical implementation used for the paper. The public entry point connects simulation, analysis and plotting without requiring the full output archive. The B2 input snapshot includes the exact acceleration-history values required by its initial jerk constraints. Small differences between platforms or solver versions may affect numerical tolerances, and execution timings are not portable.

Correspondence: Zhengwu Wang, zhengwu.wang@csust.edu.cn.
