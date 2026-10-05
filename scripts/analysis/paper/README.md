# Paper tables and figures

Scripts that turn evaluation results into the tables and figures of the paper. They need pandas,
matplotlib, seaborn and scipy (not torch/JAX), so they can run on a laptop; the paper fonts
(Times New Roman) must be installed for the final figures. Set `SOURCE_DATE_EPOCH=0` to get
reproducible PDFs.

## Inputs

| Input | Produced by |
|---|---|
| `metrics_long.csv` | `scripts/analysis/eval_fast.py` then `aggregate_fast.py` (cluster), on the validation sets of the paper |
| per-seed CSV (legacy layout, `metrics_per_seed_*.csv`) | `report/make_seed_csv_from_fast.py` from `metrics_long.csv`; it keeps exactly the checkpoints (model_curK + model_last/best of each seed) listed in the legacy CSV `metrics_per_seed_true_normalized.csv` |
| W&B history export (`all_experiments_aggregated*.csv`) | W&B export of the training runs (DDPM loss) |

The legacy CSV is needed by every script that reads `metrics_long.csv`: it fixes which checkpoints
are compared (`--legacy`).

## Commands

```bash
cd scripts/analysis
# per-seed CSV from the fast re-evaluation (one-step, target-RMS normalisation, per transition)
python report/make_seed_csv_from_fast.py --long metrics_long.csv \
    --legacy metrics_per_seed_true_normalized.csv --out metrics_per_seed_fast.csv

# tables: table_{navier,kuramoto,gray}.tex, tables_all.tex (+ ratio_summary.tex)
python paper/tables.py --input metrics_per_seed_fast.csv --out-dir out/ --ratio-summary
# point plots: strategy_pointplot_{raw,ratio}_custom_metrics.pdf
python paper/pointplots.py --input metrics_per_seed_fast.csv --out-dir out/
# DDPM loss of OGAS-L: ddpm_loss_<pde>_smooth50.pdf (--wandb-export only the first time)
python paper/ddpm_loss.py --wandb-export all_experiments_aggregated.csv \
    --history ddpm_loss_evolution.csv --out-dir out/
```

Every script prints its options with `--help`. `--stats` changes the columns of the tables and point
plots (any `rmse_normalized_<stat>` column, e.g. `cvar_0.99`).

## Notes

- The published tables and point plots were made from the legacy CSV itself
  (`--input metrics_per_seed_true_normalized.csv`); the scripts reproduce them exactly. Rebuilt from
  `metrics_long.csv`, the tables are the same except the Gray-Scott/UNet OGAS-U row (the legacy row of
  seed 2 holds the evaluation of seed 4; one bold cell moves to Top-K) and two last-digit roundings
  (Kuramoto-Sivashinsky).
- Strategies of the tables: OGAS-L = `ddpm_conf_ratio_proportional_normalized_sm`,
  OGAS-L2 = `ddpm_conf_ratio_proportional_normalized`, OGAS-U, SBAL, Top-K, Sobol, Uniform
  (`TABLE_STRATEGIES` in `paper_common.py`).
- `pointplots.py` and `ddpm_loss.py` keep the SciencePlots-based style of the published figures.
