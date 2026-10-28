#!/usr/bin/env python3
"""
PHASE 3 / STEP 3 - Inter-rater reliability between two independent human coders.

Step 2 scores the deterministic rules against one human. That answers "do the
rules assign the category a trained human would?" but it is explicitly NOT
inter-rater reliability, because a deterministic coder cannot disagree with
itself. This script closes that gap: two people code the same items blind to
each other and blind to the answer key, and their agreement with each other is
the reliability figure a reviewer actually asks for.

    python phase3_codebook/03_coder_agreement.py \
        --coder-a phase3_codebook/outputs/coded_obada.csv \
        --coder-b phase3_codebook/outputs/coded_coauthor.csv

WHAT THIS PRODUCES
------------------
1. Human A vs Human B kappa            <- the headline reliability figure
2. Human A vs the automated rules      <- rule validity, coder A
3. Human B vs the automated rules      <- rule validity, coder B
4. Confusion matrices for A vs B
5. A disagreements CSV listing every item the two coders coded differently,
   with the context columns, so the pair can adjudicate them line by line.

Report (1) in the Method section as the inter-rater reliability. Report the
mean of (2) and (3) as the validity of the automated scheme against human
judgement. Adjudicate the disagreements, then say how many were resolved.

Outputs
-------
phase3_codebook/outputs/reports/codebook_irr_apa_report.txt
phase3_codebook/outputs/reports/codebook_irr_results.json
phase3_codebook/outputs/reports/codebook_irr_results.csv
phase3_codebook/outputs/tables/codebook_irr_*.csv
phase3_codebook/outputs/codebook_irr_disagreements.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.stats import TestResult                         # noqa: E402

VARIABLES = (
    ("Element role", "human_role", "auto_role"),
    ("Label status", "human_label_status", "auto_label_status"),
)

CONTEXT = ["dataset", "class_name", "content_desc", "text",
           "clickable", "long_clickable"]


# --------------------------------------------------------------------------
# agreement statistics
# --------------------------------------------------------------------------

def cohens_kappa(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Cohen's kappa plus observed agreement and the number of usable pairs."""
    categories = sorted(set(a.dropna()) | set(b.dropna()))
    index = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)))
    for x, y in zip(a, b):
        if pd.isna(x) or pd.isna(y):
            continue
        matrix[index[x], index[y]] += 1
    total = matrix.sum()
    if total == 0:
        return float("nan"), float("nan"), 0
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum()) / (total ** 2)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return kappa, observed, int(total)


def bootstrap_kappa(a: pd.Series, b: pd.Series, n_boot: int,
                    seed: int) -> tuple[float, float]:
    """Percentile CI for kappa by resampling coded items with replacement."""
    rng = np.random.default_rng(seed)
    positions = np.arange(len(a))
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(positions, size=len(positions), replace=True)
        k, _, _ = cohens_kappa(a.iloc[idx].reset_index(drop=True),
                              b.iloc[idx].reset_index(drop=True))
        if not np.isnan(k):
            draws.append(k)
    if not draws:
        return float("nan"), float("nan")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def interpret_kappa(k: float) -> str:
    if np.isnan(k):
        return "not computable"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def clean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def load_coder(path: Path, label: str, con: Console) -> pd.DataFrame:
    if not path.exists():
        con.fail(f"Missing {label} sheet: {path}",
                 hint="Both coders must save their filled-in copy of "
                      "validation_sample_blank.csv before this runs.")
    frame = pd.read_csv(path)
    for column in ("item_id", "human_role", "human_label_status"):
        if column not in frame.columns:
            con.fail(f"{label} sheet has no '{column}' column ({path}).",
                     hint="Code the sheet in place. Do not delete or rename "
                          "columns, and do not re-order the rows.")
    keep = ["item_id", "human_role", "human_label_status"]
    if "human_notes" in frame.columns:
        keep.append("human_notes")
    frame = frame[keep].copy()
    frame["human_role"] = clean(frame["human_role"])
    frame["human_label_status"] = clean(frame["human_label_status"])
    blank = {"", "nan", "none"}
    coded = frame[~frame["human_role"].isin(blank)
                  & ~frame["human_label_status"].isin(blank)].copy()
    con.kv(f"{label}: rows in sheet", len(frame))
    con.kv(f"{label}: rows coded", len(coded))
    return coded


def check_vocabulary(frame: pd.DataFrame, label: str, con: Console) -> list[str]:
    """Warn about codes that are not in the codebook. Typos ruin a kappa."""
    problems = []
    allowed = {
        "human_role": {"image", "button", "text", "input",
                       "container", "web", "other"},
        "human_label_status": {"labelled", "unlabelled",
                               "named_by_text", "not_applicable"},
    }
    for column, valid in allowed.items():
        bad = sorted(set(frame[column]) - valid)
        if bad:
            problems.append(f"{label} used {len(bad)} value(s) outside the "
                            f"codebook in {column}: {', '.join(bad)}")
            con.warn(problems[-1])
    return problems


# --------------------------------------------------------------------------

def main() -> int:
    parser = base_parser(
        "Inter-rater reliability between two independent human coders.",
        epilog="Run after both coders have filled in their own copy of the sheet.",
    )
    parser.add_argument("--coder-a", type=str, required=True,
                        help="path to the first coder's completed CSV")
    parser.add_argument("--coder-b", type=str, required=True,
                        help="path to the second coder's completed CSV")
    parser.add_argument("--name-a", type=str, default="Coder A",
                        help="label for the first coder in the report")
    parser.add_argument("--name-b", type=str, default="Coder B",
                        help="label for the second coder in the report")
    args = parser.parse_args()

    con = Console("03_coder_agreement", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 3 / STEP 3", "Inter-rater reliability, two human coders")

    here = Path(__file__).resolve().parent
    outdir = here / "outputs"
    key_path = outdir / "validation_sample_key.csv"
    if not key_path.exists():
        con.fail(f"Missing {key_path}",
                 hint="Run 01_draw_validation_sample.py first.")

    con.step(1, 5, "Load both coding sheets")
    a = load_coder(Path(args.coder_a), args.name_a, con)
    b = load_coder(Path(args.coder_b), args.name_b, con)
    key = pd.read_csv(key_path)

    problems = (check_vocabulary(a, args.name_a, con)
                + check_vocabulary(b, args.name_b, con))

    con.step(2, 5, "Align on the items both coders completed")
    pair = a.merge(b, on="item_id", how="inner", suffixes=("_a", "_b"))
    key_columns = ["item_id", "auto_role", "auto_label_status"]
    key_columns += [c for c in CONTEXT if c in key.columns]
    pair = pair.merge(key[key_columns], on="item_id", how="inner")
    pair["auto_role"] = clean(pair["auto_role"])
    pair["auto_label_status"] = clean(pair["auto_label_status"])

    n_pairs = len(pair)
    con.kv("Items coded by both", n_pairs)
    only_a = len(set(a["item_id"]) - set(b["item_id"]))
    only_b = len(set(b["item_id"]) - set(a["item_id"]))
    if only_a or only_b:
        con.note(f"{only_a} item(s) coded only by {args.name_a}, "
                 f"{only_b} only by {args.name_b}. Reliability uses the overlap.")

    target_n = int(CFG.analysis.codebook.n_double_coded)
    if n_pairs < 30:
        con.fail(
            f"Only {n_pairs} item(s) coded by both. Kappa is unstable below "
            f"about 100 and this project targets {target_n}.",
            hint="Both coders must cover the SAME item_id range. Check that "
                 "neither sheet was re-sorted before coding.",
        )
    if n_pairs < target_n:
        con.warn(f"{n_pairs} double-coded, target is {target_n}. "
                 f"The interval on kappa will be wide.")

    con.step(3, 5, "Compute agreement")
    rep = ResultsReport(
        stem="codebook_irr",
        title="Inter-rater reliability",
        subtitle=f"{args.name_a} and {args.name_b} coding independently, "
                 f"plus each coder against the automated rules",
        outdir=outdir,
        script="phase3_codebook/03_coder_agreement.py",
    )
    rep.provenance(
        coder_a=str(Path(args.coder_a)), coder_b=str(Path(args.coder_b)),
        seed=args.seed, codebook="phase3_codebook/CODEBOOK.md",
        target_kappa=CFG.analysis.codebook.target_kappa,
    )
    rep.section(
        "Design",
        f"Two coders applied the definitions in CODEBOOK.md to the same "
        f"{n_pairs} stratified elements, independently and blind both to each "
        f"other and to the automated codes. Human-to-human agreement is the "
        f"inter-rater reliability of the coding scheme. Each coder is then "
        f"scored against the deterministic rules in taxonomy.py; because those "
        f"rules cannot disagree with themselves, that second comparison "
        f"measures the validity of the automated scheme against human "
        f"judgement, not reliability. The two are reported separately and "
        f"should be described separately in the Method section.",
    )
    rep.stat("Items double-coded", n_pairs)
    rep.stat(f"Items coded by {args.name_a} only", only_a)
    rep.stat(f"Items coded by {args.name_b} only", only_b)

    n_boot = min(2000, int(CFG.run.n_bootstrap))
    target = float(CFG.analysis.codebook.target_kappa)
    headline: dict[str, float] = {}

    comparisons = [
        ("human_human", f"{args.name_a} vs {args.name_b}",
         "_a", "_b", "inter-rater reliability"),
        ("a_auto", f"{args.name_a} vs automated rules",
         "_a", "auto", "scheme validity"),
        ("b_auto", f"{args.name_b} vs automated rules",
         "_b", "auto", "scheme validity"),
    ]

    for variable, human_col, auto_col in VARIABLES:
        for tag, title, left, right, kind in comparisons:
            left_col = f"{human_col}{left}"
            right_col = auto_col if right == "auto" else f"{human_col}{right}"
            kappa, observed, n = cohens_kappa(pair[left_col], pair[right_col])
            lo, hi = bootstrap_kappa(pair[left_col], pair[right_col],
                                     n_boot, args.seed)
            if tag == "human_human":
                headline[variable] = kappa
            con.kv(f"{variable} | {title}",
                   f"kappa {kappa:.3f} [{lo:.3f}, {hi:.3f}], "
                   f"raw {100 * observed:.1f}%")
            rep.test(
                f"kappa_{tag}_{human_col}",
                TestResult(
                    name=f"Cohen's kappa, {variable.lower()}: {title}",
                    statistic_name="agreement", statistic=observed,
                    p_value=float("nan"), df=None,
                    effect_name="Cohen's kappa", effect_size=kappa,
                    ci_low=lo, ci_high=hi, n=n,
                    groups={"raw_agreement_pct": 100 * observed,
                            "comparison": kind,
                            "n_categories": int(pd.concat(
                                [pair[left_col], pair[right_col]]).nunique())},
                    note=f"Percentile CI from {n_boot:,} bootstrap replicates of "
                         f"the double-coded items. Chance-corrected; raw "
                         f"agreement alone overstates reliability.",
                ),
                hypothesis=f"{title} agree on {variable.lower()} beyond chance.",
                interpretation=f"{interpret_kappa(kappa)} ({kind})",
            )

    con.step(4, 5, "Confusion matrices and disagreements")
    for variable, human_col, _ in VARIABLES:
        matrix = pd.crosstab(pair[f"{human_col}_a"], pair[f"{human_col}_b"])
        rows = [[index] + [int(v) for v in row]
                for index, row in zip(matrix.index, matrix.values)]
        rep.table(Table(
            f"irr_confusion_{human_col}",
            [f"{args.name_a} \\ {args.name_b}"] + list(matrix.columns), rows,
            caption=f"Confusion matrix for {variable.lower()} between the two "
                    f"human coders. Rows are {args.name_a}, columns are "
                    f"{args.name_b}.",
            note="Off-diagonal cells are the items to adjudicate. A "
                 "disagreement concentrated in one cell is an ambiguous rule "
                 "worth rewording in the codebook; a scattered one is coder "
                 "fatigue and is fixed by adjudication alone.",
        ))
        con.table([f"{args.name_a} \\ {args.name_b}"] + [str(c) for c in matrix.columns],
                  [[str(index)] + [str(v) for v in row]
                   for index, row in zip(matrix.index, matrix.values)])

    disagree = pair[(pair["human_role_a"] != pair["human_role_b"])
                    | (pair["human_label_status_a"] != pair["human_label_status_b"])].copy()
    disagree_columns = ["item_id"] + [c for c in CONTEXT if c in disagree.columns] + [
        "human_role_a", "human_role_b", "auto_role",
        "human_label_status_a", "human_label_status_b", "auto_label_status",
    ]
    disagree_columns += [c for c in ("human_notes_a", "human_notes_b")
                         if c in disagree.columns]
    disagree = disagree[disagree_columns]
    disagree.insert(1, "adjudicated_role", "")
    disagree.insert(2, "adjudicated_label_status", "")
    disagree.insert(3, "adjudication_note", "")
    disagree_path = outdir / "codebook_irr_disagreements.csv"
    disagree.to_csv(disagree_path, index=False)
    con.kv("Items where the coders disagree", len(disagree))
    con.ok(f"Disagreements written to {disagree_path}")
    rep.stat("Items requiring adjudication", int(len(disagree)))
    rep.stat("Items in exact agreement", int(n_pairs - len(disagree)))
    rep.note(f"Every disagreement is listed with its context in "
             f"{disagree_path.name}, with blank columns for the agreed code. "
             f"Adjudicate them together, record the resolution there, and state "
             f"in the Method section how many items were resolved by discussion.")

    con.step(5, 5, "Verdict")
    below = [v for v, k in headline.items()
             if not np.isnan(k) and k < target]
    if below:
        con.warn(f"Human-to-human agreement below the {target:.2f} target: "
                 f"{', '.join(below)}")
        con.note("Read the confusion matrix for that variable, agree on the "
                 "ambiguous rule, reword CODEBOOK.md, and re-code the affected "
                 "cells. Do not report a kappa you would not defend.")
        rep.limitation(
            f"Inter-rater agreement for {', '.join(below)} fell below the "
            f"pre-registered target of {target:.2f}. The shortfall, and the "
            f"categories responsible for it, are reported in the confusion "
            f"matrix above and must be stated in the paper."
        )
    else:
        con.ok(f"Both variables meet the {target:.2f} inter-rater target")
        rep.note(f"Both coded variables met the pre-set inter-rater "
                 f"reliability target of {target:.2f}.")

    for problem in problems:
        rep.limitation(problem)

    rep.limitation(
        "Kappa is sensitive to the marginal distribution of categories. The "
        "sample is stratified by role and by automated label status precisely "
        "so that rare categories are represented; on an unstratified draw the "
        "same rules would yield a higher kappa with less information in it."
    )

    written = rep.write(verbose_console=con)
    con.done(written)
    con.info("Next:  python phase6_synthesis/01_manuscript_numbers.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
