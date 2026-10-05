#!/usr/bin/env python3
import argparse
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WRAPPER = os.path.join(REPO, "scripts", "leo", "mpirun_alloc.sh")
UNSET_SLURM = "for v in $(env | grep -oE '^SLURM_[A-Z_]*'); do unset $v; done"
SINGLE_THREAD = [
    UNSET_SLURM,
    "unset CUDA_VISIBLE_DEVICES",
    "export JAX_PLATFORMS=cpu",
    "export OMP_NUM_THREADS=1",
    "export MKL_NUM_THREADS=1",
    "export OPENBLAS_NUM_THREADS=1",
    "export NUMEXPR_NUM_THREADS=1",
    "export SLURM_CPUS_PER_TASK=1",
    'export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"',
]


def port_legacy(cfg):
    so, ac, dl = cfg["study_options"], cfg["active_sampling_config"], cfg["dl_config"]
    if "field_names" in so:
        so["node_names"] = so.pop("field_names")
    if ac.get("breeder_backend") == "dl_breeder":
        ac["breeder_backend"] = "ogas_breeder"
        bp = ac["breed_params"]
        bp.pop("log_extra", None)
        bp["data"].pop("use_timesteps", None)
        for key in ("nb_candidates", "loss_sampling_power", "use_kmeans_selection", "use_batch_priors"):
            bp["breeder"].pop(key, None)
        bp["breeder"].update({"max_mlp_steps_per_collect": 10**9, "ratio_mlp_hidden_dim": 64, "ratio_mlp_dropout": 0.0,
                              "ratio_mlp_weight_decay": 0.01, "ratio_mlp_label_smoothing": 0.0})
    if ac.pop("use_normalized_loss_for_breed", False):
        ac["loss_type"] = "nrmse"
    limit = dl.pop("simple_rate_limit", None)
    if limit is not None:
        dl["rate_limit_speed_batch_per_sim"] = float(limit["speed_batch_per_sim"])
    cfg["client_config"].pop("client_wrapper_command", None)
    cfg.get("monitoring_config", {}).pop("log_memory", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("output_dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-models", type=int, help="ensemble size (one training rank and GPU per member)")
    ap.add_argument("--signal", choices=["uncertainty", "ecrps"], help="ensemble signal fed to the breeder")
    ap.add_argument("--train-ranks", type=int, help="training ranks the members are spread over (default: one per member)")
    ap.add_argument("--sampler-gpu", action="store_true", help="give the DDPM sampler its own GPU (job_study.sh)")
    ap.add_argument("--clients", type=int, help="concurrent solvers (default: 8 per member; offline: one per CPU core)")
    ap.add_argument("--wandb-group", help="W&B group (default: the source group, with the signal and _M<m>)")
    a = ap.parse_args()

    cfg = json.loads(re.sub(r",(\s*[}\]])", r"\1", open(a.src).read()))
    if "dl_config" not in cfg:  # offline (validation-set) config: no training, CPU clients only
        return offline(cfg, a)
    port_legacy(cfg)
    tm, dl = cfg["dl_config"]["torch_model"], cfg["dl_config"]
    tm.pop("distribute_on_gpus", None)
    if a.n_models:
        tm.update(use_ensemble=a.n_models > 1, n_models=a.n_models)
    if a.signal:
        cfg["active_sampling_config"]["value_to_register"] = a.signal
    if a.seed is not None:
        cfg["study_options"]["seed"] = a.seed
    n_members = int(tm.get("n_models", 2 if tm.get("use_ensemble") else 1))
    ranks = a.train_ranks or n_members
    if n_members % ranks:
        raise SystemExit(f"{n_members} members cannot be split over {ranks} training ranks")
    group = dl.get("wandb_group", "")
    if a.signal and "uncertainty" in group:
        group = group.replace("uncertainty", a.signal)
    if a.n_models:
        group += f"_M{n_members}"
    dl["wandb_group"] = a.wandb_group or group

    lc = cfg["launcher_config"]
    for key in ("scheduler_client_command_options", "scheduler_server_command_options"):
        lc.pop(key, None)
    lc.update({
        "std_output": True, "scheduler": "openmpi", "protocol": "tcp", "fault_tolerance": True,
        "scheduler_server_command": WRAPPER, "scheduler_client_command": WRAPPER,
        "scheduler_server_command_options": ["--server"],
        "scheduler_arg_server": ["-n", str(ranks + 1)],
        "scheduler_arg_client": ["-n", "1"],
        "job_limit": a.clients or 8 * n_members,
    })
    dl["external_validation"] = False
    cfg["study_options"]["persistent_client_mode"] = True
    srv = cfg.setdefault("server_config", {}).setdefault("preprocessing_commands", [])
    srv[:] = [UNSET_SLURM] + [c for c in srv if c != UNSET_SLURM]
    cli = cfg["client_config"].setdefault("preprocessing_commands", [])
    cfg["client_config"]["preprocessing_commands"] = SINGLE_THREAD + [
        c for c in cli if c not in SINGLE_THREAD and "JAX_PLATFORMS" not in c and not c.startswith("export CONFIG_FILE=")
    ] + [f"export CONFIG_FILE={os.path.abspath(a.out)}"]
    cfg["output_dir"] = os.path.abspath(a.output_dir)
    cfg.setdefault("campaign_metadata", {})["description"] = (
        f"{os.path.basename(a.src)}: {n_members} member(s) on {ranks} training rank(s), {lc['job_limit']} persistent clients, seed "
        f"{cfg['study_options'].get('seed')}")
    if a.sampler_gpu:
        cfg["campaign_metadata"]["sampler_gpu"] = True
    json.dump(cfg, open(a.out, "w"), indent=2)
    print("wrote", a.out)


def offline(cfg, a):
    lc = cfg["launcher_config"]
    lc.update({
        "std_output": True, "scheduler": "openmpi", "protocol": "tcp",
        "scheduler_server_command": WRAPPER, "scheduler_client_command": WRAPPER,
        "scheduler_server_command_options": ["--server"],
        "scheduler_arg_server": ["-n", "1"], "scheduler_arg_client": ["-n", "1"],
        "job_limit": a.clients or os.cpu_count(),
    })
    srv = cfg.setdefault("server_config", {}).setdefault("preprocessing_commands", [])
    srv[:] = [UNSET_SLURM] + [c for c in srv if c != UNSET_SLURM]
    cli = cfg["client_config"].setdefault("preprocessing_commands", [])
    cfg["client_config"]["preprocessing_commands"] = SINGLE_THREAD + [c for c in cli if c not in SINGLE_THREAD]
    if a.seed is not None:
        cfg["study_options"]["seed"] = a.seed
    cfg["output_dir"] = os.path.abspath(a.output_dir)
    json.dump(cfg, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
