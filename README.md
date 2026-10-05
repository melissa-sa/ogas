# Learning Where to Simulate

### Generative Active Sampling for Online PDE Surrogate Training (OGAS)

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-7b2cbf.svg)](#citation) 
[![arXiv](https://img.shields.io/badge/arXiv-2606.09949-b31b1b.svg)](https://arxiv.org/abs/2606.09949)
[![Project page](https://img.shields.io/badge/project-page-lightgrey.svg)](#) 
[![Data](https://img.shields.io/badge/data-TODO-lightgrey.svg)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**Pierre Cesar, Sofya Dymchenko, Abhishek Purandare, Bruno Raffin**<br/>
Univ. Grenoble Alpes, Inria, CNRS, Grenoble INP, LIG (DataMove team)

**Accepted at NeurIPS 2026.**

[Paper](https://arxiv.org/abs/2606.09949) · [Project page](#) <!-- TODO --> · [Citation](#citation)

Neural PDE surrogates trained on uniformly sampled simulations are accurate on average but make their largest errors
in the hardest regimes. In online training, simulations stream to the surrogate while it trains, so the next ones
can target those regimes. **OGAS** does this with a conditional diffusion model trained alongside the surrogate: it
learns which solver parameters are currently hard and generates the next ones, mixed with uniform draws.

On 3 PDEs × 3 architectures (189 runs of 10,000 simulations), at the same simulation budget as uniform sampling:

| Ratio to Uniform (> 1: lower error) | RMSE-max | RMSE-p99 | RMSE-std | RMSE-mean |
|---|---|---|---|---|
| **OGAS-L** (training-loss signal) | 1.64 | 1.42 | 1.66 | 0.97 |
| **OGAS-U** (ensemble-disagreement signal) | 1.47 | 1.32 | 1.50 | 0.96 |

Details are in the [paper](https://arxiv.org/abs/2606.09949).

## Installation

```bash
git clone --recursive https://github.com/melissa-sa/OGAS ogas && cd ogas
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r frozen_requirements.txt
pip install --no-deps -e . "./melissa[launcher,server,torch]"
```

This needs an MPI implementation, CMake and a C/C++ compiler (Melissa is compiled), and CUDA 12 GPUs for training.
On Leonardo, `source leo_melissa_init.sh install` loads the modules and builds the same `.venv`; afterwards,
`source leo_melissa_init.sh` activates it.

## Usage

Everything is written under `experiments/` in the repository (`export EXP_DIR=/path/to/scratch` to put it
elsewhere). The job scripts are for Slurm on Leonardo: set your account in their `#SBATCH` lines.

```bash
cd snakemake_setup && snakemake --cores all && cd ..       # configs of the whole study
scripts/leo/job_global_offline_cpu.sh                        # validation sets, one CPU job per PDE
scripts/leo/submit_benchmark.sh ddpm_conf_ratio_proportional_normalized   # one method: all PDEs, architectures, seeds
scripts/leo/job_eval_fast_chain.sh                           # evaluate all runs -> experiments/eval/metrics_long.csv
```

Methods for `submit_benchmark.sh`:

| Uniform | Sobol | Breed | SBAL / Top-K | **OGAS-L** | **OGAS-U** |
|---|---|---|---|---|---|
| `no_resampling` | `no_resampling_sobol` | `mixed_nrmse` | `sbal` / `sbal_topk` | `ddpm_conf_ratio_proportional_normalized` | `ddpm_conf_ratio_proportional_uncertainty` |

The paper tables and figures are built from `metrics_long.csv` by
[`scripts/analysis/paper/`](scripts/analysis/paper/README.md).

**Ablations** (Kuramoto-Sivashinsky: ensemble size M = 2, 3, 5 for OGAS-U, and the ensemble CRPS as signal):

```bash
scripts/leo/submit_ablations.sh
scripts/leo/job_eval_fast_chain.sh experiments/ablations/ensemble_size experiments/eval_ensemble --prediction-mode member0
```

The ensemble-size runs are evaluated on their first member only, so that larger ensembles are not favored by
averaging.

**A single run**, e.g. OGAS-L on Kuramoto-Sivashinsky with a UNet, seed 1:

```bash
python scripts/configs/make_study_config.py \
    experiments/test_ensemble/kuramoto_sivashinsky_2d_low_res/unet_cond/ddpm_conf_ratio_proportional_normalized/config_online_1.json \
    ks_ogasl_s1.json experiments/runs/ks_ogasl_s1          # options: --n-models, --signal, --seed
scripts/leo/job_study.sh ks_ogasl_s1.json ks_ogasl_s1 --qos=normal --time=07:00:00
```

One GPU per ensemble member is allocated automatically. A run directory holds the checkpoints (`checkpoints/model_*.pt`),
the sampled parameters (`sampled_parameters.npy`) and the final DDPM (`ddpm_final.pt`).

### Configuration reference

| File | Content |
|---|---|
| [`snakemake_setup/config.yaml`](snakemake_setup/config.yaml) | sweep: PDEs, methods, architectures, seeds, 10,000 training / 750 validation simulations |
| [`snakemake_setup/pde_set.csv`](snakemake_setup/pde_set.csv) | one row per PDE scenario |
| [`snakemake_setup/default_configs_slurm.json`](snakemake_setup/default_configs_slurm.json) | Melissa config template |
| [`apebench_online/scenarios/2d_config.yaml`](apebench_online/scenarios/2d_config.yaml) | PDE registry: solver settings, time steps, parameter ranges |

| PDE | Initial condition | Physical parameter |
|---|---|---|
| Gray-Scott | 8 Gaussian blobs (48 parameters) | domain size ∈ [0.5, 10] |
| Kuramoto-Sivashinsky | truncated Fourier series (308 parameters) | domain size ∈ [10, 130] |
| Kolmogorov flow | truncated Fourier series (308 parameters) | viscosity ∈ [0.003, 0.008] |

## Repository structure

```
apebench_online/        Python package: solver client (core/solver.py), training servers, models and ensemble,
                        parameter samplers, SBAL baseline, PDE scenarios, in-training validator
melissa/                Melissa (submodule): launcher, server, OGAS sampler (server/deep_learning/active_sampling)
scripts/
  generate_config_cli.py, configs/   config generation; make_study_config.py
  leo/                               job scripts (study, benchmark, ablations, evaluation chain) on leonardo cluster
  analysis/                          evaluation, validation sets, figures; paper/ for the paper tables and figures
snakemake_setup/        experiment sweep definition
```

## Citation

```bibtex
@inproceedings{cesar2026learning,
  title     = {Learning Where to Simulate: Generative Active Sampling for Online {PDE} Surrogate Training},
  author    = {Cesar, Pierre and Dymchenko, Sofya and Purandare, Abhishek and Raffin, Bruno},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). Melissa is distributed under its own BSD 3-Clause license; adapted third-party model
code (AL4PDE, PDEArena, scOT) is in [`apebench_online/vendor/`](apebench_online/vendor) with its license. The PDE
solvers come from [APEBench](https://github.com/tum-pbs/apebench) and [Exponax](https://github.com/Ceyron/exponax).

## Acknowledgements

This work was supported by the Exa-DoST project of the NumPEx PEPR program (France 2030, ANR-22-EXNU-0004) and by
the European Union's Horizon programme under grant agreement No 101144014 (EoCoE-III). It used HPC resources of
IDRIS (GENCI allocation AD010610366R4) and the EuroHPC supercomputer LEONARDO hosted by CINECA, and the Grid'5000
and GRICAD infrastructures for early testing.
