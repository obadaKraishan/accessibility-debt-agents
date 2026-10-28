#!/usr/bin/env python3
"""
RQ3 — Do benchmark task trajectories touch labelled elements at a higher
rate than the background population of elements on the same screens?

This is the question that turns a coverage table into a paper. If the
elements agents are asked to operate are systematically better labelled
than the elements around them, then agent benchmarks are easier than the
real app population, and that optimism bias is an accessibility artifact.

    python phase5_analysis/rq3_trajectory_bias/run_rq3.py
    python phase5_analysis/rq3_trajectory_bias/run_rq3.py --n-perm 20000

HYPOTHESES
----------
H3.1  Touched elements are labelled at a higher rate than the background
      population on the same screens.
H3.2  The difference survives restricting the background to clickable
      elements only, which is the fairer comparison.
H3.3  The difference survives restricting to icon-like elements.

WHY THE BACKGROUND IS WITHIN-SCREEN
-----------------------------------
The comparison set is every included element on screens a trajectory
actually visited, not every element in the corpus. Comparing touched
elements against the whole corpus would confound the question with which
screens agents happen to reach. Holding the screen fixed isolates the
element-selection effect, which is what the hypothesis is about.

HOW 'TOUCHED' IS DETERMINED
---------------------------
AndroidControl actions carry only coordinates: there is no element
reference anywhere in the action record. Touched elements are therefore
inferred by point-in-bounds hit testing (see src/a11ydebt/tfrecord.py:
resolve_click). The hit rate is reported here as well as at parse time,
because a low rate would silently bias this analysis.

Outputs
-------
outputs/reports/rq3_apa_report.txt
outputs/tables/rq3_*.csv
outputs/figures/rq3_*.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.figures import (                                # noqa: E402
    COLUMN, CATEGORICAL, save, use_paper_style,
)
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.stats import (                                  # noqa: E402
    permutation_test, proportion, two_proportion_test,
)

NEEDS = ("labelled", "unlabelled")
MIN_TOUCHED = 100


def rates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (touched labelled 0/1, background labelled 0/1)."""
    needs = frame[frame["needs_label"]]
    labelled = (needs["label_status"] == "labelled").astype(int)
    touched = needs["was_touched"].astype(bool)
    return labelled[touched].to_numpy(), labelled[~touched].to_numpy()


def main() -> int:
    parser = base_parser("RQ3: are touched elements better labelled than their neighbours?")
    parser.add_argument("--n-perm", type=int, default=None,
                        help="permutations (default: config run.n_permutations)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    outdir = Path(args.out) if args.out else here / "outputs"
    con = Console("run_rq3", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("RQ3", "Trajectory selection bias in benchmark tasks")

    path = PATHS.processed / "rq3_touched.parquet"
    if not path.exists():
        con.fail(f"Missing {path}",
                 hint="Run phase2_processing/02_parse_androidcontrol.py then "
                      "phase2_processing/03_build_analysis_tables.py")

    n_perm = args.n_perm or int(CFG.run.n_permutations)

    con.step(1, 5, "Load")
    df = pd.read_parquet(path)
    n_touched = int(df["was_touched"].sum())
    con.kv("Elements on visited screens", len(df))
    con.kv("Touched elements", n_touched)
    con.kv("Screens visited", df["screen_uid"].nunique())
    con.kv("Episodes", df["episode_id"].nunique())

    if n_touched < MIN_TOUCHED:
        con.fail(
            f"Only {n_touched} touched elements; RQ3 needs at least {MIN_TOUCHED} "
            f"for a stable permutation test.",
            hint="Download and parse more AndroidControl shards:\n"
                 "             python phase1_collection/02_download_androidcontrol.py --shards 10\n"
                 "             python phase2_processing/02_parse_androidcontrol.py\n"
                 "             python phase2_processing/03_build_analysis_tables.py",
        )

    rep = ResultsReport(
        stem="rq3",
        title="RQ3: Trajectory selection bias",
        subtitle="Are elements touched by benchmark tasks better labelled than "
                 "the elements around them?",
        outdir=outdir,
        script="phase5_analysis/rq3_trajectory_bias/run_rq3.py",
    )
    rep.provenance(
        table=str(path), seed=args.seed, n_permutations=n_perm,
        hit_test_rule=CFG.analysis.hit_test.as_dict(),
    )
    rep.section(
        "Design",
        "The comparison set is every included element on screens a trajectory "
        "actually visited, not the whole corpus. Holding the screen fixed "
        "isolates element selection from screen selection, which is what the "
        "hypothesis concerns. Touched elements are inferred by point-in-bounds "
        "hit testing, because AndroidControl actions record coordinates and no "
        "element reference.",
    )
    rep.stat("Elements on visited screens", int(len(df)))
    rep.stat("Touched elements", n_touched)
    rep.stat("Screens visited", int(df["screen_uid"].nunique()))
    rep.stat("Episodes contributing", int(df["episode_id"].nunique()))

    report_path = PATHS.interim / "androidcontrol_parse_report.json"
    if report_path.exists():
        blob = json.loads(report_path.read_text(encoding="utf-8"))
        hit = float(blob.get("hit_rate_pct", 0))
        rep.stat("Click-to-element hit rate", f"{hit:.2f}%",
                 note="Share of click actions resolved to a visible element. A low "
                      "rate would bias this analysis toward whatever the rule can see.")
        con.kv("Hit rate at parse time", f"{hit:.2f}%")

    con.step(2, 5, "H3.1 — all elements needing a label")
    rep.section("Results")
    comparisons = [
        ("h3_1_all", "all elements requiring a label", df,
         "H3.1. Touched elements are labelled at a higher rate than the "
         "background population on the same screens."),
        ("h3_2_clickable", "clickable elements only", df[df["clickable"]],
         "H3.2. The difference survives restricting the background to clickable "
         "elements, which is the fairer comparison since trajectories can only "
         "touch clickable targets."),
        ("h3_3_icon_like", "icon-like elements only", df[df["icon_like"]],
         "H3.3. The difference survives restricting to tappable graphics with "
         "no text, where a missing description is unambiguously a defect."),
    ]

    rows = []
    for i, (key, label, subset, hypothesis) in enumerate(comparisons, start=2):
        if i > 2:
            con.step(i, 5, f"{key.upper()} — {label}")
        touched, background = rates(subset)
        if touched.size < 30 or background.size < 30:
            con.warn(f"{label}: too few elements ({touched.size} touched, "
                     f"{background.size} background). Skipped.")
            rep.note(f"{hypothesis.split('.')[0]} was not tested: only "
                     f"{touched.size} touched and {background.size} background "
                     f"elements in the '{label}' subset.")
            continue

        pt = proportion(int(touched.sum()), int(touched.size))
        pb = proportion(int(background.sum()), int(background.size))
        rows.append([label, f"{touched.size:,}", f"{pt['pct']:.2f}",
                     f"{background.size:,}", f"{pb['pct']:.2f}",
                     f"{pt['pct'] - pb['pct']:+.2f}"])
        con.kv(f"{label}: touched", f"{pt['pct']:.2f}% ({touched.sum():,}/{touched.size:,})")
        con.kv(f"{label}: background", f"{pb['pct']:.2f}% ({background.sum():,}/{background.size:,})")

        rep.test(
            key,
            permutation_test(
                touched, background, n_perm=n_perm, seed=args.seed,
                alternative="two-sided",
                name=f"Permutation test, {label}",
                effect_name="difference in label rate",
            ),
            hypothesis=hypothesis,
        )
        # A parametric companion, so a reader who prefers z has one.
        rep.test(
            f"{key}_z",
            two_proportion_test(int(touched.sum()), int(touched.size),
                                int(background.sum()), int(background.size),
                                label1="touched", label2="background"),
            hypothesis=f"Parametric companion to {key.upper()} ({label}).",
        )

    rep.table(Table(
        "touched_vs_background",
        ["Comparison", "Touched n", "Touched %", "Background n", "Background %",
         "Difference (pp)"],
        rows,
        caption="Label coverage among touched versus background elements on the "
                "same screens.",
        note="Background excludes touched elements. Permutation tests shuffle the "
             "touched/background assignment while holding the pooled outcomes fixed.",
    ))

    con.step(5, 5, "Figure and report")
    use_paper_style()
    import matplotlib.pyplot as plt

    if rows:
        fig, ax = plt.subplots(figsize=COLUMN)
        labels = [r[0].replace(" only", "").replace("elements ", "") for r in rows]
        touched_vals = [float(r[2]) for r in rows]
        background_vals = [float(r[4]) for r in rows]
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(x - width / 2, touched_vals, width * 0.92, label="Touched",
               color=CATEGORICAL[0], hatch="", edgecolor="white", linewidth=0.6)
        ax.bar(x + width / 2, background_vals, width * 0.92, label="Background",
               color=CATEGORICAL[1], hatch="///", edgecolor="white", linewidth=0.6)
        for xi, (t, b) in enumerate(zip(touched_vals, background_vals)):
            ax.annotate(f"{t:.0f}", (xi - width / 2, t + 1), ha="center",
                        fontsize=7, color="#1a1a22")
            ax.annotate(f"{b:.0f}", (xi + width / 2, b + 1), ha="center",
                        fontsize=7, color="#1a1a22")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Label coverage (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        fig_paths = save(fig, outdir / "figures" / "rq3_touched_vs_background")
        plt.close(fig)
        rep.figure("rq3_touched_vs_background", fig_paths[0],
                   "Label coverage among elements touched by task trajectories "
                   "versus the background population on the same screens.")
    else:
        fig_paths = []

    rep.note("Permutation p-values use the (r+1)/(m+1) convention and can never "
             "be exactly zero, which is the honest way to report a test whose "
             "null was never matched.")
    rep.limitation(
        "Touched elements are inferred from coordinates, not read from the data. "
        "A systematic bias in the hit-testing rule, for instance a preference "
        "for larger targets, would propagate into this result. The reported hit "
        "rate and the clickable-only comparison are the two guards against that."
    )
    rep.limitation(
        "Elements within a screen are not independent, and one screen can "
        "contribute many background elements and one touched element. The "
        "permutation test respects the observed marginals but not the nesting; "
        "a screen-level cluster bootstrap would be the stronger design and is "
        "the natural extension for the journal version."
    )
    rep.narrative(
        "If touched elements are better labelled than their neighbours, the "
        "practical reading is that benchmark tasks are easier than the app "
        "population they are drawn from, and that the gap is an accessibility "
        "artifact rather than a property of the task design. Reported agent "
        "accuracy would then be an optimistic estimate of performance on "
        "arbitrary screens."
    )

    written = rep.write(verbose_console=con)
    if fig_paths:
        written["rq3_touched_vs_background.pdf"] = fig_paths[0]
    con.done(written)
    con.info("Next:  python phase5_analysis/rq4_difficulty_model/run_rq4.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
