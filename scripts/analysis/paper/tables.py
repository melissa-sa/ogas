#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from paper_common import (ARCH_NAMES, BASELINE, METRIC_LABEL, METRIC_PREFIX, PAPER_STATS, PDE_NAMES,
                          STAT_NAMES, TABLE_STRATEGIES, load_per_seed, ordered, ratio_to_uniform)


def common_exponent(values) -> int:
    exps = sorted(math.floor(math.log10(abs(v))) for v in values if math.isfinite(v) and v != 0)
    return exps[len(exps) // 2] if exps else 0


def cell(mean: float, std: float, exp: int) -> str:
    if not math.isfinite(mean):
        return "--"
    body = f"{mean / 10 ** exp:.2f}" if mean != 0 else "0.00"
    if math.isfinite(std) and std > 0:
        body += f" \\pm {std / 10 ** exp:.2f}"
    return f"${body}$"


def bold(text: str) -> str:
    return text if text == "--" else f"$\\mathbf{{{text[1:-1]}}}$"


def pde_table(stats, pde: str, cols: list, keys: list) -> str:
    name = PDE_NAMES.get(pde, pde)
    sub = stats[stats.pde == pde]
    archs = ordered(sub.architecture, ARCH_NAMES)
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{Performance comparison on {name}. Values show mean $\\pm$ std across seeds. "
             "Best per architecture in \\textbf{bold}.}",
             f"\\label{{tab:{pde.replace('_', '-')}}}"]
    for arch in archs:
        a = sub[sub.architecture == arch].set_index("strategy")
        exps = [common_exponent(a[f"{c}_mean"].dropna()) for c in cols]
        header = ["Strategy"] + [STAT_NAMES.get(k, k.upper()) + (f" ($\\times 10^{{{e}}}$)" if e else "")
                                 for k, e in zip(keys, exps)]
        lines += [f"% {ARCH_NAMES.get(arch, arch)}", "\\begin{subtable}{\\linewidth}", "\\centering",
                  f"\\caption{{{ARCH_NAMES.get(arch, arch)}}}", f"\\begin{{tabular}}{{l{'c' * len(cols)}}}",
                  "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
        for strat in [s for s in TABLE_STRATEGIES if s in a.index]:
            row = [TABLE_STRATEGIES[strat]]
            for c, e in zip(cols, exps):
                mean, best = a.at[strat, f"{c}_mean"], a[f"{c}_mean"].min()
                text = cell(mean, a.at[strat, f"{c}_std"], e)
                row.append(bold(text) if math.isfinite(mean) and abs(mean - best) < 1e-9 * max(1.0, abs(best))
                           else text)
            lines.append(" & ".join(row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{subtable}"]
        if arch != archs[-1]:
            lines.append("\\vspace{0.5em}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def ratio_tabular(r, cols: list, keys: list) -> list:
    g = r.groupby("strategy")[cols].agg(["mean", "std"])
    header = ["\\textbf{Strategy}"] + [f"\\textbf{{{METRIC_LABEL}-{k}}}" for k in keys]
    lines = [f"\\begin{{tabular}}{{l{' c' * len(cols)}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for strat in [s for s in TABLE_STRATEGIES if s in g.index]:
        row = [TABLE_STRATEGIES[strat]] + [f"${g.at[strat, (c, 'mean')]:.2f} \\pm {g.at[strat, (c, 'std')]:.2f}$"
                                           for c in cols]
        lines.append(" & ".join(row) + " \\\\")
    return lines + ["\\bottomrule", "\\end{tabular}"]


def ratio_summary(df, cols: list, keys: list) -> str:
    r = ratio_to_uniform(df, cols)
    r = r[r.strategy != BASELINE]
    blocks = ["\n".join(["\\begin{table}[h]", "\\centering",
                         "\\caption{Aggregate improvement ratio vs Uniform (Baseline). "
                         "Averaged over all PDEs, Architectures, and Seeds.}",
                         "\\label{tab:aggregate_stats}"] + ratio_tabular(r, cols, keys) + ["\\end{table}"])]
    for col, names, what, label in (("architecture", ARCH_NAMES, "Architectures", "archs"),
                                    ("pde", PDE_NAMES, "PDEs", "pdes")):
        lines = ["\\begin{table}[h]", "\\centering",
                 f"\\caption{{Improvement ratio vs Uniform across {what}. Values are Mean $\\pm$ Std.}}",
                 f"\\label{{tab:rel_imp_{label}}}"]
        for key in ordered(df[col], names):
            sub = r[r[col] == key]
            if sub.empty:
                continue
            lines += ["\\begin{subtable}{\\linewidth}", "\\centering", f"\\caption{{{names.get(key, key)}}}",
                      f"\\label{{tab:rel_imp_{key}}}"] + ratio_tabular(sub, cols, keys)
            lines += ["\\end{subtable}", "\\vspace{0.5em}"]
        blocks.append("\n".join(lines + ["\\end{table}"]))
    return "\n\n".join(blocks) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per-seed CSV (legacy layout)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stats", default=PAPER_STATS, help=f"comma-separated statistics (default {PAPER_STATS})")
    ap.add_argument("--ratio-summary", action="store_true", help="also write ratio_summary.tex")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = [k.strip() for k in a.stats.split(",")]
    cols = [f"{METRIC_PREFIX}_{k}" for k in keys]
    df = load_per_seed(a.input)
    stats = df.groupby(["pde", "architecture", "strategy"])[cols].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    stats = stats.reset_index()

    tables = []
    for pde in ordered(stats.pde, PDE_NAMES):
        tables.append(pde_table(stats, pde, cols, keys))
        path = out / f"table_{pde.split('_')[0]}.tex"
        path.write_text(tables[-1] + "\n", encoding="utf-8")
        print("wrote", path)
    (out / "tables_all.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print("wrote", out / "tables_all.tex")
    if a.ratio_summary:
        (out / "ratio_summary.tex").write_text(ratio_summary(df, cols, keys), encoding="utf-8")
        print("wrote", out / "ratio_summary.tex")


if __name__ == "__main__":
    main()
