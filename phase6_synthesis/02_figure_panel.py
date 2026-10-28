#!/usr/bin/env python3
"""
PHASE 6 / STEP 2 — Build the paper's main multi-panel figure.

An IEEE paper at eight pages has room for two or three figures. This
assembles the headline result into a single page-width panel: overall
coverage by corpus, coverage by element role, and the touched-versus-
background comparison that carries the argument.

    python phase6_synthesis/02_figure_panel.py

Reads the result JSON written by the RQ scripts rather than re-analysing,
so the panel can never disagree with the reports.

Outputs
-------
phase6_synthesis/outputs/figure_main_panel.pdf   <- the one to submit
phase6_synthesis/outputs/figure_main_panel.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                            # noqa: E402

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import PATHS, PROJECT_ROOT               # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.figures import (                                # noqa: E402
    CATEGORICAL, DATASET_COLOURS, hatch_for, save, use_paper_style,
)

RQ1 = "phase5_analysis/rq1_label_coverage/outputs/reports/rq1_results.json"
RQ3 = "phase5_analysis/rq3_trajectory_bias/outputs/reports/rq3_results.json"


def load(rel: str) -> dict | None:
    path = PROJECT_ROOT / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table_from(blob: dict, name: str) -> dict | None:
    for table in blob.get("tables", []):
        if table.get("name") == name:
            return table
    return None


def main() -> int:
    parser = base_parser("Assemble the paper's main multi-panel figure.")
    args = parser.parse_args()

    con = Console("02_figure_panel", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 6 / STEP 2", "Build the main figure panel")

    outdir = Path(__file__).resolve().parent / "outputs"
    rq1 = load(RQ1)
    rq3 = load(RQ3)
    if rq1 is None:
        con.fail("RQ1 results not found.",
                 hint="Run: python phase5_analysis/rq1_label_coverage/run_rq1.py")

    con.step(1, 3, "Read results")
    overall = table_from(rq1, "overall_coverage")
    by_role = table_from(rq1, "coverage_by_role")
    touched = table_from(rq3, "touched_vs_background") if rq3 else None
    con.kv("Overall table", "found" if overall else "missing")
    con.kv("Role table", "found" if by_role else "missing")
    con.kv("Touched table", "found" if touched else "missing (RQ3 not run)")

    con.step(2, 3, "Draw")
    use_paper_style()
    import matplotlib.pyplot as plt

    n_panels = 2 + (1 if touched else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.16, 2.5))
    if n_panels == 1:
        axes = [axes]

    # Panel A: overall coverage
    ax = axes[0]
    datasets = [row[0] for row in overall["rows"]]
    values = [float(row[4]) for row in overall["rows"]]
    bars = ax.bar(range(len(datasets)), values,
                  color=[DATASET_COLOURS.get(d, CATEGORICAL[0]) for d in datasets],
                  hatch=[hatch_for(d) for d in datasets],
                  edgecolor="white", linewidth=0.6, width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value + 2),
                    ha="center", fontsize=7)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=7)
    ax.set_ylabel("Label coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Overall", loc="left", fontsize=8)

    # Panel B: coverage by role
    ax = axes[1]
    roles = sorted({row[1] for row in by_role["rows"]})
    width = 0.8 / max(1, len(datasets))
    for i, dataset in enumerate(datasets):
        vals, pos = [], []
        for j, role in enumerate(roles):
            match = [r for r in by_role["rows"] if r[0] == dataset and r[1] == role]
            vals.append(float(match[0][5]) if match else 0.0)
            pos.append(j + (i - (len(datasets) - 1) / 2) * width)
        ax.bar(pos, vals, width=width * 0.9,
               color=DATASET_COLOURS.get(dataset, CATEGORICAL[i]),
               hatch=hatch_for(dataset), edgecolor="white", linewidth=0.6,
               label=dataset.replace("_", " "))
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("Label coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) By element role", loc="left", fontsize=8)
    ax.legend(loc="upper right", fontsize=6.5)

    # Panel C: touched vs background
    if touched:
        ax = axes[2]
        labels = [r[0].replace(" only", "").replace("elements ", "") for r in touched["rows"]]
        t_vals = [float(r[2]) for r in touched["rows"]]
        b_vals = [float(r[4]) for r in touched["rows"]]
        x = np.arange(len(labels))
        # Panel C is entirely AndroidControl, so it must NOT reuse the two
        # dataset hues from panels A and B. Doing so made blue mean "rico" in
        # one panel and "touched" in another, inside a single figure. Both bars
        # therefore take the AndroidControl hue and are separated by fill:
        # touched is solid, background is open with a coloured edge.
        ac_hue = DATASET_COLOURS.get("androidcontrol_2024", CATEGORICAL[1])
        ax.bar(x - 0.18, t_vals, 0.34, label="Touched by a task",
               color=ac_hue, edgecolor="white", linewidth=0.6)
        ax.bar(x + 0.18, b_vals, 0.34, label="Background on same screens",
               facecolor="white", edgecolor=ac_hue, linewidth=0.9, hatch="///")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
        ax.set_ylabel("Label coverage (%)")
        ax.set_ylim(0, 105)
        ax.set_title("(c) Trajectory selection (androidcontrol 2024)",
                     loc="left", fontsize=8)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.9)

    fig.tight_layout()

    con.step(3, 3, "Save")
    paths = save(fig, outdir / "figure_main_panel")
    plt.close(fig)
    for path in paths:
        con.ok(f"{path.name}")
    con.note("Check it at printed size before submitting: open the PDF and zoom "
             "to 100% at 7.16 inches wide, then look at it in greyscale.")
    con.done({p.name: p for p in paths})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
