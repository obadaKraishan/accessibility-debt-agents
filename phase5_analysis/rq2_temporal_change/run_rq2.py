#!/usr/bin/env python3
"""
RQ2 — Has coverage improved between the 2017 Rico snapshot and the 2024-25
AndroidControl apps, and for which element classes?

    python phase5_analysis/rq2_temporal_change/run_rq2.py

RESEARCH QUESTION
-----------------
RQ2. Has content-description coverage improved over time, and where?

HYPOTHESES
----------
H2.1  Overall coverage is higher in the newer corpus.
H2.2  The improvement is not uniform across element roles.
H2.3  The improvement holds on the strict icon-like subpopulation.

THE CONFOUND YOU MUST NOT BURY
------------------------------
These are two different corpora collected by different teams with different
sampling frames, not two waves of one panel. The gap between them therefore
confounds at least four things:

  * genuine change in developer labelling practice over seven years;
  * app selection (Rico crawled the 2017 Play Store broadly; AndroidControl
    was assembled around task demonstrations on 833 apps);
  * platform change (Android's accessibility APIs, lint checks and Material
    component defaults all changed in between);
  * instrumentation (Rico dumped view hierarchies, AndroidControl captured
    accessibility forests, and these are related but not identical views).

This script therefore reports the gap as a corpus difference and never as a
causal estimate of change over time. The wording in the auto-composed draft
follows that rule; keep it when you edit.

Outputs
-------
outputs/reports/rq2_apa_report.txt
outputs/tables/rq2_*.csv
outputs/figures/rq2_*.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd                                           # noqa: E402

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import PATHS                             # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.figures import (                                # noqa: E402
    WIDE, CATEGORICAL, save, use_paper_style,
)
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.stats import fdr_bh, proportion, two_proportion_test  # noqa: E402

NEEDS = ("labelled", "unlabelled")
OLD, NEW = "rico_2017", "androidcontrol_2024"
MIN_CELL = 30


def coverage(frame: pd.DataFrame) -> tuple[int, int]:
    """
    (labelled, needs a label).

    The denominator is `needs_label`, NOT 'label_status is labelled or
    unlabelled'. Those differ for exempt roles: a container or web view that
    happens to carry a content description is coded 'labelled' but was never
    required to have one, so including it would report container coverage as
    100% by construction. Exempt roles must return a zero denominator.
    """
    needs = frame[frame["needs_label"]]
    return int((needs["label_status"] == "labelled").sum()), int(len(needs))


def main() -> int:
    parser = base_parser("RQ2: the cross-corpus coverage difference.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    outdir = Path(args.out) if args.out else here / "outputs"
    con = Console("run_rq2", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("RQ2", "Coverage difference between the 2017 and 2024 corpora")

    path = PATHS.processed / "elements.parquet"
    if not path.exists():
        con.fail(f"Missing {path}", hint="Run phase2_processing/03_build_analysis_tables.py")

    con.step(1, 5, "Load")
    df = pd.read_parquet(path)
    df = df[df["included"]]
    present = set(df["dataset"].unique())
    if not {OLD, NEW} <= present:
        con.fail(
            f"RQ2 needs both corpora; found {sorted(present)}.",
            hint="Run phase1_collection/02_download_androidcontrol.py and "
                 "phase2_processing/02_parse_androidcontrol.py, then rebuild the tables.",
        )
    old, new = df[df["dataset"] == OLD], df[df["dataset"] == NEW]
    con.kv(f"{OLD} elements", len(old))
    con.kv(f"{NEW} elements", len(new))

    rep = ResultsReport(
        stem="rq2",
        title="RQ2: Coverage difference between corpora",
        subtitle="Rico (2017) compared with AndroidControl (2024-25)",
        outdir=outdir,
        script="phase5_analysis/rq2_temporal_change/run_rq2.py",
    )
    rep.provenance(elements_table=str(path), seed=args.seed,
                   older_corpus=OLD, newer_corpus=NEW)

    rep.section(
        "Design and what the comparison can support",
        "Rico and AndroidControl are two independently collected corpora, not "
        "two waves of a panel. Any difference between them confounds genuine "
        "change in developer practice with app selection, platform evolution, "
        "and differences in how each corpus captured the interface. The "
        "difference is therefore reported as a corpus difference throughout. "
        "It is not a causal estimate of change over time, and it should not be "
        "written up as one.",
    )

    con.step(2, 5, "Overall difference")
    rep.section("Overall difference")
    k_old, n_old = coverage(old)
    k_new, n_new = coverage(new)
    p_old, p_new = proportion(k_old, n_old), proportion(k_new, n_new)
    rep.proportion(f"Coverage, {OLD}", k_old, n_old)
    rep.proportion(f"Coverage, {NEW}", k_new, n_new)
    con.kv(f"{OLD}", f"{p_old['pct']:.2f}%")
    con.kv(f"{NEW}", f"{p_new['pct']:.2f}%")
    con.kv("Difference", f"{p_new['pct'] - p_old['pct']:+.2f} percentage points")

    rep.test(
        "h2_1_overall",
        two_proportion_test(k_new, n_new, k_old, n_old, label1=NEW, label2=OLD),
        hypothesis="H2.1. Coverage is higher in the newer corpus.",
    )

    con.step(3, 5, "By element role")
    rep.section("Difference by element role")
    rows, tests, p_values = [], [], []
    for role in sorted(set(old["role"]) | set(new["role"])):
        ko, no = coverage(old[old["role"] == role])
        kn, nn = coverage(new[new["role"] == role])
        if no == 0 and nn == 0:
            continue
        po = proportion(ko, no) if no else {"pct": float("nan")}
        pn = proportion(kn, nn) if nn else {"pct": float("nan")}
        rows.append([role, f"{no:,}", f"{po['pct']:.2f}", f"{nn:,}",
                     f"{pn['pct']:.2f}",
                     f"{pn['pct'] - po['pct']:+.2f}" if no and nn else "n/a"])
        if no >= MIN_CELL and nn >= MIN_CELL:
            t = two_proportion_test(kn, nn, ko, no,
                                    label1=f"{NEW} {role}", label2=f"{OLD} {role}")
            tests.append((role, t))
            p_values.append(t.p_value)

    rep.table(Table(
        "difference_by_role",
        ["Role", f"{OLD} n", f"{OLD} %", f"{NEW} n", f"{NEW} %", "Difference (pp)"],
        rows,
        caption="Label coverage by element role in each corpus.",
        note="Roles are assigned from the class name alone in both corpora, so "
             "neither benefits from richer metadata. Cells below "
             f"{MIN_CELL} elements are shown but not tested.",
    ))
    con.table(["Role", f"{OLD} %", f"{NEW} %", "Diff (pp)"],
              [[r[0], r[2], r[4], r[5]] for r in rows])

    if p_values:
        _, adjusted = fdr_bh(p_values)
        for (role, t), adj in zip(tests, adjusted):
            t.note += f" Benjamini-Hochberg adjusted p = {adj:.4f}."
            rep.test(f"h2_2_role_{role}", t,
                     hypothesis=f"H2.2. The corpus difference for '{role}' elements.")
        rep.note(f"H2.2 comprises {len(p_values)} role comparisons; each line "
                 f"reports the unadjusted p with the Benjamini-Hochberg "
                 f"adjusted value appended.")

    con.step(4, 5, "Icon-like subpopulation")
    rep.section("The strict subpopulation")
    kio, nio = coverage(old[old["icon_like"]])
    kin, nin = coverage(new[new["icon_like"]])
    if nio >= MIN_CELL and nin >= MIN_CELL:
        rep.test(
            "h2_3_icon_like",
            two_proportion_test(kin, nin, kio, nio,
                                label1=f"{NEW} icon-like", label2=f"{OLD} icon-like"),
            hypothesis="H2.3. The corpus difference holds on tappable graphics "
                       "with no text, where a missing description is unambiguously "
                       "a defect.",
        )
    else:
        con.warn("Too few icon-like elements for H2.3; parse more data.")

    con.step(5, 5, "Figure and report")
    use_paper_style()
    import matplotlib.pyplot as plt

    roles = [r[0] for r in rows if r[5] != "n/a"]
    diffs = [float(r[5]) for r in rows if r[5] != "n/a"]
    order = sorted(range(len(roles)), key=lambda i: diffs[i])
    fig, ax = plt.subplots(figsize=WIDE)
    colours = [CATEGORICAL[2] if diffs[i] >= 0 else CATEGORICAL[1] for i in order]
    hatches = ["" if diffs[i] >= 0 else "///" for i in order]
    bars = ax.barh([roles[i] for i in order], [diffs[i] for i in order],
                   color=colours, hatch=hatches, edgecolor="white",
                   linewidth=0.6, height=0.6)
    ax.axvline(0, color="#1a1a22", linewidth=0.8)
    for bar, i in zip(bars, order):
        value = diffs[i]
        ax.annotate(f"{value:+.1f}",
                    (value, bar.get_y() + bar.get_height() / 2),
                    xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center",
                    fontsize=7, color="#1a1a22")
    ax.set_xlabel("Coverage difference, newer minus older (percentage points)")
    ax.grid(axis="y", visible=False)
    fig_paths = save(fig, outdir / "figures" / "rq2_difference_by_role")
    plt.close(fig)
    rep.figure("rq2_difference_by_role", fig_paths[0],
               "Coverage difference by element role, newer corpus minus older. "
               "Direction is encoded by hatching as well as colour.")

    rep.limitation(
        "Corpus differences are not causal estimates of change over time. App "
        "selection, platform evolution and capture method all differ between "
        "the two datasets and cannot be separated with these data alone."
    )
    rep.limitation(
        "AndroidControl covers 833 apps chosen to support task demonstrations, "
        "which is a narrower and probably better-maintained population than "
        "Rico's broad 2017 crawl of 9,384 apps. That alone would push the "
        "newer corpus's coverage upward."
    )
    rep.note("A defensible way to state the finding is: the corpus that "
             "represents contemporary, actively maintained apps shows markedly "
             "higher coverage, and the residual debt in it is the quantity that "
             "matters for agents today.")

    written = rep.write(verbose_console=con)
    written["rq2_difference_by_role.pdf"] = fig_paths[0]
    con.done(written)
    con.info("Next:  python phase5_analysis/rq3_trajectory_bias/run_rq3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
