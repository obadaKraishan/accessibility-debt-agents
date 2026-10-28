#!/usr/bin/env python3
"""
PHASE 4 / STEP 2 — Sensitivity of the headline numbers to coding decisions.

Every coverage figure in this paper rests on choices that a reviewer can
reasonably question: what counts as an element, what counts as needing a
label, whether Rico's richer metadata gives it an unfair advantage, and
whether a string in `content-desc` is really a label at all. This script
re-computes the headline numbers under each alternative and reports how far
they move. Run it before writing the Results section and put the table in an
appendix.

    python phase4_validation/02_sensitivity_checks.py

VARIANTS TESTED
---------------
 A. Primary specification (the codebook as written).
 B. Exclude disabled elements from the analysis population.
 C. Icon-like elements only, the strictest and most defensible unit.
 D. Rico classified with `ancestors` instead of class name alone, with
    AndroidControl unchanged. This is the specification most favourable to
    Rico, so the gap that survives it is the gap that survives the
    "your 2017 classifier was just worse" objection.
 E. Uninformative content descriptions counted as absent. A description that
    names the widget rather than its function ("image", "button"), echoes the
    resource id, or is a single character or a bare number, tells a screen
    reader nothing. This variant attacks the paper's own headline in the
    honest direction by asking how much of the 2024 figure is real labelling.

WHY VARIANT E CHANGED
---------------------
It previously tested descriptions that duplicate the element's own visible
text. That variant is vacuous by construction: an element with visible text
does not need a label, so it is never in the coverage denominator, and
flipping its status cannot move any reported number. The 2026-09-10 run
confirmed this, reporting variant E identical to variant A to the last
element. The worry behind it was real, so it is now tested where it can
actually bite: on elements that DO need a label and were counted as having
one.

Outputs
-------
phase4_validation/outputs/reports/sensitivity_apa_report.txt
phase4_validation/outputs/reports/sensitivity_results.json
phase4_validation/outputs/tables/sensitivity_variants.csv
phase4_validation/outputs/tables/sensitivity_uninformative.csv
phase4_validation/outputs/figures/sensitivity_coverage.pdf
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import PATHS                             # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.figures import (                                # noqa: E402
    COLUMN, DATASET_COLOURS, hatch_for, save, use_paper_style,
)
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.stats import proportion, two_proportion_test    # noqa: E402

NEEDS = ("labelled", "unlabelled")
ABSENT = {"", "none", "null", "nan"}
CORE = ["dataset", "needs_label", "label_status"]

#: Strings that name the widget rather than its function, or stand in for a
#: description that was never written. Deliberately conservative: real labels
#: like "Menu", "Back", "Search" or "Play" are informative and stay out of it.
GENERIC_DESC = {
    # the widget's own type
    "image", "imageview", "imagebutton", "button", "icon", "iconview",
    "view", "textview", "text", "label", "img", "picture", "photo",
    "thumbnail", "avatar", "logo", "item", "cell", "row", "container",
    # placeholders for a description nobody wrote
    "untitled", "default", "placeholder", "unlabeled", "unlabelled",
    "nolabel", "nodescription", "todo", "tbd", "test", "dummy", "temp",
    "unknown", "empty", "blank", "na",
}

_NONALNUM = re.compile(r"[^a-z0-9]+")


def normalise_desc(series: pd.Series) -> pd.Series:
    """Lowercase, strip, and drop punctuation, for set membership only."""
    return (series.fillna("").astype(str).str.strip().str.lower()
            .str.replace(_NONALNUM, "", regex=True))


def clean_field(series: pd.Series) -> pd.Series:
    """Stripped text with the corpora's absent-value spellings mapped to ''."""
    out = series.fillna("").astype(str).str.strip()
    return out.mask(out.str.lower().isin(ABSENT), "")


def resource_leaf(series: pd.Series) -> pd.Series:
    """'com.app:id/btn_search' -> 'btnsearch'."""
    leaf = series.fillna("").astype(str).str.rsplit("/", n=1).str[-1]
    return normalise_desc(leaf)


def recode(frame: pd.DataFrame, role_column: str) -> pd.DataFrame:
    """
    Recompute `needs_label` and `label_status` from a different role column.

    Vectorised restatement of taxonomy.needs_label and taxonomy.label_status.
    Used only for variant D, where Rico is reclassified with its superclass
    chain; the primary tables always come from the taxonomy module itself.
    """
    role = frame[role_column].astype(str)
    text = clean_field(frame["text"])
    desc = clean_field(frame["content_desc"])

    has_text = text != ""
    named = desc != ""
    exempt = role.isin(["container", "web"])
    interactive = (frame["clickable"].fillna(False).astype(bool)
                   | frame["long_clickable"].fillna(False).astype(bool)
                   | role.isin(["input", "button"]))
    needs = (~exempt) & (~has_text) & ((role == "image") | interactive)

    status = np.where(named, "labelled",
             np.where(needs, "unlabelled",
             np.where(has_text, "named_by_text", "not_applicable")))

    out = frame[["dataset"]].copy()
    out["needs_label"] = needs.to_numpy()
    out["label_status"] = status
    return out


def uninformative(frame: pd.DataFrame) -> pd.Series:
    """True where a present content description carries no functional meaning."""
    desc_raw = clean_field(frame["content_desc"])
    desc = normalise_desc(desc_raw)
    present = desc_raw != ""
    return present & (
        desc.isin(GENERIC_DESC)
        | (desc.str.len() <= 1)
        | desc.str.fullmatch(r"\d+").fillna(False)
        | ((desc != "") & (desc == resource_leaf(frame["resource_id"])))
    )


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
    parser = base_parser("How far do the headline numbers move under alternative rules?")
    args = parser.parse_args()

    con = Console("02_sensitivity_checks", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 4 / STEP 2", "Sensitivity of the headline numbers")

    here = Path(__file__).resolve().parent
    outdir = here / "outputs"
    path = PATHS.processed / "elements.parquet"
    if not path.exists():
        con.fail(f"Missing {path}",
                 hint="Run phase2_processing/03_build_analysis_tables.py first.")

    con.step(1, 4, "Build variants")
    df = pd.read_parquet(path)
    included = df[df["included"]]

    variants: dict[str, pd.DataFrame] = {}
    variants["A. Primary specification"] = included[CORE]
    variants["B. Exclude disabled elements"] = included[included["enabled"]][CORE]
    variants["C. Icon-like elements only"] = included[included["icon_like"]][CORE]

    # -- variant D: Rico with its superclass chain, AndroidControl unchanged
    rico_nodes = PATHS.processed / "rico_nodes.parquet"
    d_rows, disagree, n_rico = None, None, None
    if rico_nodes.exists():
        rn = pd.read_parquet(rico_nodes)
        rn = rn[rn["included"]]
        n_rico = len(rn)
        disagree = int((rn["role"] != rn["role_with_ancestors"]).sum())
        con.kv("Rico elements where the two classifiers disagree", f"{disagree:,}")
        con.kv("As a share of included Rico elements",
               f"{100 * disagree / max(1, n_rico):.2f}%")
        rico_d = recode(rn, "role_with_ancestors")
        ac_primary = included[included["dataset"] != "rico_2017"][CORE]
        variants["D. Rico via superclass chain"] = pd.concat(
            [rico_d[CORE], ac_primary], ignore_index=True)
        d_rows = rico_d
    else:
        con.warn(f"{rico_nodes.name} not found; variant D skipped.")

    # -- variant E: uninformative descriptions counted as absent
    e = included[CORE + ["content_desc", "resource_id"]].copy()
    junk = uninformative(e)
    e.loc[junk & e["needs_label"] & (e["label_status"] == "labelled"),
          "label_status"] = "unlabelled"
    variants["E. Uninformative labels count as absent"] = e[CORE]

    junk_rows = []
    for dataset, group in included.assign(_junk=uninformative(included)).groupby("dataset"):
        needs = group[group["needs_label"]]
        labelled = needs[needs["label_status"] == "labelled"]
        n_junk = int(labelled["_junk"].sum())
        junk_rows.append([
            dataset, f"{len(labelled):,}", f"{n_junk:,}",
            f"{100 * n_junk / max(1, len(labelled)):.2f}",
        ])
    con.table(["Dataset", "Labelled (needs a label)", "Uninformative", "Share %"],
              junk_rows)

    con.step(2, 4, "Compute coverage under each variant")
    rows = []
    for name, frame in variants.items():
        for dataset, group in frame.groupby("dataset"):
            k, n = coverage(group)
            p = proportion(k, n)
            rows.append([name, dataset, f"{n:,}", f"{k:,}",
                         f"{p['pct']:.2f}",
                         f"[{p['ci_low_pct']:.2f}, {p['ci_high_pct']:.2f}]"])
    con.table(["Variant", "Dataset", "Needs label", "Labelled", "Coverage %", "95% CI"],
              rows)

    con.step(3, 4, "Cross-corpus gap under each variant")
    gap_rows = []
    for name, frame in variants.items():
        groups = {d: coverage(g) for d, g in frame.groupby("dataset")}
        if len(groups) != 2:
            continue
        (d1, (k1, n1)), (d2, (k2, n2)) = sorted(groups.items())
        if not (n1 and n2):
            continue
        result = two_proportion_test(k2, n2, k1, n1, label1=d2, label2=d1)
        gap_rows.append([
            name.split(".")[0],
            f"{100 * k1 / n1:.2f}", f"{100 * k2 / n2:.2f}",
            f"{100 * (k1 / n1 - k2 / n2):+.2f}",
            f"{result.effect_size:.3f}",
        ])
        con.kv(f"Variant {name.split('.')[0]} gap",
               f"{100 * (k1 / n1 - k2 / n2):+.2f} pp, "
               f"Cohen's h = {result.effect_size:.3f}")
    con.table(["Variant", "androidcontrol %", "rico %", "Gap (pp)", "Cohen's h"],
              gap_rows)

    con.step(4, 4, "Report")
    rep = ResultsReport(
        stem="sensitivity",
        title="Sensitivity to coding decisions",
        subtitle="How far the headline coverage figures move under alternative rules",
        outdir=outdir,
        script="phase4_validation/02_sensitivity_checks.py",
    )
    rep.provenance(elements_table=str(path), variants=list(variants))
    rep.section(
        "Purpose",
        "Coverage figures depend on choices a reviewer can reasonably question. "
        "Rather than defend one specification, this table shows the number under "
        "each alternative. A claim that survives every row is robust; a claim "
        "that appears only under the primary specification should be softened.",
    )
    rep.table(Table(
        "sensitivity_variants",
        ["Variant", "Dataset", "Needs label", "Labelled", "Coverage %", "95% CI"],
        rows,
        caption="Label coverage under alternative coding specifications.",
        note="Variant C is the strictest unit and the one to quote when a "
             "reviewer challenges the broader definitions. Variant D is the "
             "specification most favourable to Rico and variant E the one most "
             "hostile to the 2024 corpus, so a gap surviving both is not an "
             "artifact of either.",
    ))
    if gap_rows:
        rep.table(Table(
            "sensitivity_gap",
            ["Variant", "androidcontrol %", "rico %", "Gap (pp)", "Cohen's h"],
            gap_rows,
            caption="The cross-corpus coverage gap under each specification.",
            note="Read down the Gap column. The claim is robust to the extent "
                 "that this column does not change sign or order of magnitude.",
        ))
    for name, frame in variants.items():
        groups = {d: coverage(g) for d, g in frame.groupby("dataset")}
        if len(groups) == 2:
            (d1, (k1, n1)), (d2, (k2, n2)) = sorted(groups.items())
            if n1 and n2:
                rep.test(
                    f"gap_{name.split('.')[0].lower()}",
                    two_proportion_test(k2, n2, k1, n1, label1=d2, label2=d1),
                    hypothesis=f"Variant {name}: the cross-corpus coverage gap.",
                )

    if disagree is not None:
        rep.section("Variant D: classifier fairness")
        rep.stat("Rico elements classified differently with ancestors", disagree)
        rep.proportion("Share of included Rico elements affected", disagree, n_rico)
        if d_rows is not None:
            k_d, n_d = coverage(d_rows)
            k_a, n_a = coverage(included[included["dataset"] == "rico_2017"][CORE])
            rep.proportion("Rico coverage, class name only", k_a, n_a)
            rep.proportion("Rico coverage, superclass chain", k_d, n_d)
            rep.stat("Movement in Rico coverage",
                     round(100 * (k_d / max(1, n_d) - k_a / max(1, n_a)), 3),
                     unit="pp")
        rep.text(
            "The primary analysis classifies both corpora from the class name "
            "alone, because AndroidControl carries no superclass chain. Variant D "
            "reclassifies Rico with the richer signal while leaving "
            "AndroidControl untouched, which is the specification most "
            "favourable to Rico that the data permits. The cross-corpus gap "
            "under variant D is therefore the answer to the objection that the "
            "2017 figure is low because the classifier saw less of it."
        )

    rep.section("Variant E: is a description a label?")
    rep.table(Table(
        "sensitivity_uninformative",
        ["Dataset", "Labelled (needs a label)", "Uninformative", "Share %"],
        junk_rows,
        caption="Content descriptions that name the widget rather than its "
                "function, echo the resource id, or are a single character or "
                "bare number.",
        note="Counted among elements that need a label and were coded "
             "'labelled', which is the only place such a string can inflate "
             "coverage.",
    ))
    rep.text(
        "Variant E is deliberately hostile to this paper's own headline. If a "
        "large share of the 2024 corpus's descriptions were placeholders, the "
        "improvement since 2017 would be partly cosmetic. The variant E row of "
        "the coverage table is what settles that, and it should be reported "
        "whether or not it is flattering."
    )
    rep.limitation(
        "The uninformative-description list is a conservative keyword rule, not "
        "a semantic judgement. It will miss a description that is fluent but "
        "wrong ('Search' on a settings button), and that class of error cannot "
        "be detected from the view hierarchy at all. Variant E is therefore a "
        "lower bound on low-quality labelling."
    )

    con.info("Rendering the sensitivity figure ...")
    use_paper_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=COLUMN)
    labels = list(variants)
    width = 0.38
    for i, dataset in enumerate(sorted({r[1] for r in rows})):
        values, positions = [], []
        for j, name in enumerate(labels):
            match = [r for r in rows if r[0] == name and r[1] == dataset]
            if match:
                values.append(float(match[0][4]))
                positions.append(j + (i - 0.5) * width)
        ax.bar(positions, values, width=width * 0.92,
               color=DATASET_COLOURS.get(dataset, "#3457b2"),
               hatch=hatch_for(dataset), edgecolor="white", linewidth=0.6,
               label=dataset.replace("_", " "))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([n.split(".")[0] for n in labels])
    ax.set_ylabel("Label coverage (%)")
    ax.set_xlabel("Specification")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=1)
    figs = save(fig, outdir / "figures" / "sensitivity_coverage")
    plt.close(fig)
    rep.figure("sensitivity_coverage", figs[0],
               "Coverage under each specification, by corpus. Hatching duplicates "
               "the colour encoding so the figure survives greyscale printing.")

    rep.limitation(
        "None of these variants can distinguish a decorative image from a "
        "meaningful one, because the view hierarchy does not encode that. "
        "The estimates are therefore upper bounds on genuine debt, which is "
        "the conservative direction for the paper's claim."
    )

    written = rep.write(verbose_console=con)
    written["sensitivity_coverage.pdf"] = figs[0]
    con.done(written)
    con.info("Next:  python phase5_analysis/rq1_label_coverage/run_rq1.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
