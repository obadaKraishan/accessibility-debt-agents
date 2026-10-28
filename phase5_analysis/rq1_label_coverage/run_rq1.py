#!/usr/bin/env python3
"""
RQ1 — What is content-description coverage, by element type, app category and rating?

The census. Establishes the base rates every later question is measured
against, and breaks them down by the dimensions a reviewer will ask about.

    python phase5_analysis/rq1_label_coverage/run_rq1.py

RESEARCH QUESTION
-----------------
RQ1. What is content-desc coverage across Rico and AndroidControl, by element
     type, app category, and Play Store rating?

HYPOTHESES TESTED
-----------------
H1.1  Label status is not independent of element role.
H1.2  Coverage among icon-like elements differs from coverage overall.
H1.3  Coverage varies across app categories (Rico only, since AndroidControl
      ships no Play Store metadata).
H1.4  Coverage is associated with app rating (Rico only).

Outputs
-------
outputs/reports/rq1_apa_report.txt      <- Method and Results source
outputs/reports/rq1_results.json
outputs/reports/rq1_results.csv
outputs/tables/rq1_*.csv
outputs/figures/rq1_*.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.figures import (                                # noqa: E402
    COLUMN, WIDE, DATASET_COLOURS, CATEGORICAL, annotate_bars,
    hatch_for, save, use_paper_style,
)
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.stats import (                                  # noqa: E402
    chi_square_test, fdr_bh, proportion, two_proportion_test,
)

NEEDS = ("labelled", "unlabelled")
MIN_CELL = 30          # smallest denominator worth reporting as a rate


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
    parser = base_parser("RQ1: the label-coverage census.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    outdir = Path(args.out) if args.out else here / "outputs"
    con = Console("run_rq1", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("RQ1", "Label coverage by element type, app category and rating")

    path = PATHS.processed / "elements.parquet"
    if not path.exists():
        con.fail(f"Missing {path}", hint="Run phase2_processing/03_build_analysis_tables.py")

    con.step(1, 6, "Load and filter")
    df = pd.read_parquet(path)
    total_rows = len(df)
    df = df[df["included"]].copy()
    con.kv("Elements parsed", total_rows)
    con.kv("Elements in the analysis population", len(df))
    con.kv("Excluded by the inclusion rule", total_rows - len(df))

    rep = ResultsReport(
        stem="rq1",
        title="RQ1: Label coverage in GUI corpora",
        subtitle="Content-description coverage by element role, app category and rating",
        outdir=outdir,
        script="phase5_analysis/rq1_label_coverage/run_rq1.py",
    )
    rep.provenance(
        elements_table=str(path), seed=args.seed,
        inclusion_rule=CFG.analysis.inclusion.as_dict(),
        codebook="phase3_codebook/CODEBOOK.md",
    )

    rep.section(
        "Sample and inclusion",
        "An element enters the analysis population when it is flagged visible to "
        "the user and has positive area. Both conditions are necessary: a large "
        "share of nodes in these hierarchies have zero or negative area while "
        "still carrying a visibility flag, so filtering on visibility alone "
        "would inflate the denominator and deflate every rate reported here.",
    )
    rep.stat("Elements parsed", int(total_rows))
    rep.stat("Elements in the analysis population", int(len(df)))
    rep.stat("Excluded by the inclusion rule", int(total_rows - len(df)))
    rep.stat("Screens or observations", int(df["screen_uid"].nunique()))
    rep.stat("Datasets", int(df["dataset"].nunique()))

    con.step(2, 6, "Overall coverage")
    rep.section("Overall coverage")
    overall = {}
    rows = []
    for dataset, group in df.groupby("dataset"):
        k, n = coverage(group)
        overall[dataset] = (k, n)
        p = rep.proportion(f"Label coverage, {dataset}", k, n,
                           note="among elements requiring a content description")
        ki, ni = coverage(group[group["icon_like"]])
        pi = rep.proportion(f"Icon-like coverage, {dataset}", ki, ni,
                            note="tappable graphics with no text of their own")
        rows.append([dataset, f"{len(group):,}", f"{n:,}", f"{k:,}",
                     f"{p['pct']:.2f}", f"{ni:,}", f"{pi['pct']:.2f}"])
        con.kv(f"{dataset} coverage", f"{p['pct']:.2f}% ({k:,}/{n:,})")
        con.kv(f"{dataset} icon-like coverage", f"{pi['pct']:.2f}% ({ki:,}/{ni:,})")

    rep.table(Table(
        "overall_coverage",
        ["Dataset", "Elements", "Needs label", "Labelled", "Coverage %",
         "Icon-like n", "Icon-like coverage %"],
        rows,
        caption="Overall label coverage by corpus.",
        note="'Needs label' excludes containers, web views, and elements named "
             "by their own visible text. Icon-like is the strictest subpopulation: "
             "a tappable graphic with no text.",
    ))

    con.step(3, 6, "Coverage by element role")
    rep.section("Coverage by element role")
    role_rows, contingency, role_labels = [], [], []
    for dataset, group in df.groupby("dataset"):
        for role, sub in group.groupby("role"):
            k, n = coverage(sub)
            if n == 0:
                continue
            p = proportion(k, n)
            role_rows.append([
                dataset, role, f"{len(sub):,}", f"{n:,}", f"{k:,}",
                f"{p['pct']:.2f}", f"[{p['ci_low_pct']:.2f}, {p['ci_high_pct']:.2f}]",
            ])
    rep.table(Table(
        "coverage_by_role",
        ["Dataset", "Role", "Elements", "Needs label", "Labelled", "Coverage %", "95% CI"],
        role_rows,
        caption="Label coverage by element role and corpus.",
        note="Roles are assigned from the class name alone in both corpora, so "
             "the two are classified from identical information. Containers and "
             "web views are exempt from labelling and therefore have no "
             "denominator; they are omitted rather than shown at a spurious 100%.",
    ))
    con.table(["Dataset", "Role", "Needs", "Labelled", "Coverage %"],
              [[r[0], r[1], r[3], r[4], r[5]] for r in role_rows])

    for dataset, group in df.groupby("dataset"):
        table, labels = [], []
        for role, sub in group.groupby("role"):
            k, n = coverage(sub)
            if n >= MIN_CELL:
                table.append([k, n - k])
                labels.append(role)
        if len(table) >= 2:
            rep.test(
                f"h1_1_role_dependence_{dataset}",
                chi_square_test(table, row_labels=labels,
                                col_labels=["labelled", "unlabelled"]),
                hypothesis=f"H1.1 ({dataset}). Label status is not independent "
                           f"of element role.",
            )

    con.step(4, 6, "Icon-like versus the broader population")
    rep.section("Icon-like elements")
    rep.text(
        "Icon-like elements are the strictest unit: a tappable graphic with no "
        "text of its own, where a missing content description is unambiguously "
        "a defect rather than a judgement call. Every claim in this paper should "
        "survive being restated on this subpopulation alone."
    )
    for dataset, group in df.groupby("dataset"):
        ki, ni = coverage(group[group["icon_like"]])
        ko, no = coverage(group[~group["icon_like"]])
        if ni >= MIN_CELL and no >= MIN_CELL:
            rep.test(
                f"h1_2_icon_gap_{dataset}",
                two_proportion_test(ki, ni, ko, no,
                                    label1=f"{dataset} icon-like",
                                    label2=f"{dataset} other"),
                hypothesis=f"H1.2 ({dataset}). Coverage among icon-like elements "
                           f"differs from coverage among other elements needing a label.",
            )

    con.step(5, 6, "App category and rating (Rico only)")
    rico = df[(df["dataset"] == "rico_2017") & df["app_category"].notna()]
    if len(rico):
        rep.section("App category and rating")
        rep.text(
            "AndroidControl ships no Play Store metadata, so this section is a "
            "Rico-only analysis and must be described as such. Categories with "
            f"fewer than {MIN_CELL} elements requiring a label are reported but "
            "excluded from testing."
        )
        cat_rows, p_values, cat_names = [], [], []
        for category, sub in rico.groupby("app_category"):
            k, n = coverage(sub)
            if n == 0:
                continue
            p = proportion(k, n)
            cat_rows.append([category, f"{sub['app_package'].nunique():,}",
                             f"{n:,}", f"{k:,}", f"{p['pct']:.2f}",
                             f"[{p['ci_low_pct']:.2f}, {p['ci_high_pct']:.2f}]"])
            if n >= MIN_CELL:
                k_rest, n_rest = coverage(rico[rico["app_category"] != category])
                if n_rest >= MIN_CELL:
                    t = two_proportion_test(k, n, k_rest, n_rest,
                                            label1=str(category), label2="all other categories")
                    p_values.append(t.p_value)
                    cat_names.append((category, t))
        cat_rows.sort(key=lambda r: float(r[4]), reverse=True)
        rep.table(Table(
            "coverage_by_category",
            ["App category", "Apps", "Needs label", "Labelled", "Coverage %", "95% CI"],
            cat_rows,
            caption="Label coverage by Play Store category (Rico only).",
            note="Sorted by coverage. Categories are self-reported by developers "
                 "and are not mutually exclusive in practice.",
        ))
        if p_values:
            _, adjusted = fdr_bh(p_values)
            for (category, t), adj in zip(cat_names, adjusted):
                t.note += f" Benjamini-Hochberg adjusted p = {adj:.4f}."
                rep.test(f"h1_3_category_{str(category).lower().replace(' ', '_')}", t,
                         hypothesis=f"H1.3. Coverage in '{category}' differs from "
                                    f"all other categories.")
            rep.note(f"H1.3 involved {len(p_values)} category comparisons; p-values "
                     f"are reported unadjusted in each line with the "
                     f"Benjamini-Hochberg adjusted value appended.")

        rated = rico[rico["app_rating"].notna()].copy()
        if len(rated) >= MIN_CELL * 4:
            rated["rating_band"] = pd.cut(
                rated["app_rating"].astype(float),
                bins=[0, 3.5, 4.0, 4.5, 5.01],
                labels=["< 3.5", "3.5-4.0", "4.0-4.5", "> 4.5"], right=False)
            band_rows, table, labels = [], [], []
            for band, sub in rated.groupby("rating_band", observed=True):
                k, n = coverage(sub)
                if n == 0:
                    continue
                p = proportion(k, n)
                band_rows.append([str(band), f"{n:,}", f"{k:,}", f"{p['pct']:.2f}",
                                  f"[{p['ci_low_pct']:.2f}, {p['ci_high_pct']:.2f}]"])
                if n >= MIN_CELL:
                    table.append([k, n - k])
                    labels.append(str(band))
            rep.table(Table(
                "coverage_by_rating",
                ["Rating band", "Needs label", "Labelled", "Coverage %", "95% CI"],
                band_rows,
                caption="Label coverage by Play Store rating band (Rico only).",
                note="Rating is an app-level attribute applied to every element of "
                     "that app, so elements are not independent within an app. "
                     "Treat this as descriptive.",
            ))
            if len(table) >= 2:
                rep.test(
                    "h1_4_rating_association",
                    chi_square_test(table, row_labels=labels,
                                    col_labels=["labelled", "unlabelled"]),
                    hypothesis="H1.4. Label coverage is associated with app rating.",
                )
    else:
        con.warn("No Rico app metadata available; skipping category and rating analysis")
        rep.limitation("App category and rating analysis was skipped because no "
                       "Rico app metadata was available in this run.")

    con.step(6, 6, "Figures and report")
    use_paper_style()
    import matplotlib.pyplot as plt

    datasets = sorted(overall)
    roles = sorted({r[1] for r in role_rows})
    fig, ax = plt.subplots(figsize=WIDE)
    width = 0.8 / max(1, len(datasets))
    for i, dataset in enumerate(datasets):
        values, positions = [], []
        for j, role in enumerate(roles):
            match = [r for r in role_rows if r[0] == dataset and r[1] == role]
            values.append(float(match[0][5]) if match else 0.0)
            positions.append(j + (i - (len(datasets) - 1) / 2) * width)
        bars = ax.bar(positions, values, width=width * 0.9,
                      color=DATASET_COLOURS.get(dataset, CATEGORICAL[i]),
                      hatch=hatch_for(dataset), edgecolor="white", linewidth=0.6,
                      label=dataset.replace("_", " "))
        annotate_bars(ax, bars, values)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles)
    ax.set_ylabel("Label coverage (%)")
    ax.set_xlabel("Element role")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    role_fig = save(fig, outdir / "figures" / "rq1_coverage_by_role")
    plt.close(fig)
    rep.figure("rq1_coverage_by_role", role_fig[0],
               "Label coverage by element role and corpus. Hatching duplicates "
               "the colour encoding for greyscale print.")

    fig, ax = plt.subplots(figsize=COLUMN)
    labels_o, values_o, errs_lo, errs_hi = [], [], [], []
    for dataset in datasets:
        k, n = overall[dataset]
        p = proportion(k, n)
        labels_o.append(dataset.replace("_", "\n"))
        values_o.append(p["pct"])
        errs_lo.append(p["pct"] - p["ci_low_pct"])
        errs_hi.append(p["ci_high_pct"] - p["pct"])
    bars = ax.bar(range(len(labels_o)), values_o,
                  color=[DATASET_COLOURS.get(d, CATEGORICAL[0]) for d in datasets],
                  hatch=[hatch_for(d) for d in datasets],
                  edgecolor="white", linewidth=0.6, width=0.55)
    ax.errorbar(range(len(labels_o)), values_o, yerr=[errs_lo, errs_hi],
                fmt="none", ecolor="#1a1a22", elinewidth=0.9, capsize=3)
    ax.set_xticks(range(len(labels_o)))
    ax.set_xticklabels(labels_o)
    ax.set_ylabel("Label coverage (%)")
    ax.set_ylim(0, 105)
    annotate_bars(ax, bars, values_o, offset=3.0)
    overall_fig = save(fig, outdir / "figures" / "rq1_overall_coverage")
    plt.close(fig)
    rep.figure("rq1_overall_coverage", overall_fig[0],
               "Overall label coverage with 95% Wilson intervals.")

    rep.note("All proportions carry Wilson score intervals, which stay correct "
             "at the extreme rates seen in these corpora where Wald intervals "
             "would not.")
    rep.note(f"Cells with fewer than {MIN_CELL} elements requiring a label are "
             f"reported in tables but excluded from significance testing.")
    rep.limitation(
        "The view hierarchy cannot distinguish a decorative image from a "
        "meaningful one, so elements coded 'unlabelled' include some that "
        "genuinely need no description. Every coverage figure is therefore a "
        "lower bound and every debt figure an upper bound."
    )
    rep.limitation(
        "Elements are nested within screens and screens within apps, so they "
        "are not independent observations. The chi-square tests here treat them "
        "as such and are best read as descriptive; RQ4 models the dependence "
        "explicitly."
    )

    written = rep.write(verbose_console=con)
    written["rq1_coverage_by_role.pdf"] = role_fig[0]
    written["rq1_overall_coverage.pdf"] = overall_fig[0]
    con.done(written)
    con.info("Next:  python phase5_analysis/rq2_temporal_change/run_rq2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
