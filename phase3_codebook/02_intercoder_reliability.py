#!/usr/bin/env python3
"""
PHASE 3 / STEP 2 — Score the codebook against human coding.

Compares the human-coded sheet with the automated codes and writes a full
reliability report: Cohen's kappa with a bootstrap CI, per-category
agreement, and a confusion matrix showing exactly where the rules and the
human diverge. The kappa this produces belongs in the Method section.

    python phase3_codebook/02_intercoder_reliability.py
    python phase3_codebook/02_intercoder_reliability.py --coded my_coding.csv

WHAT THIS MEASURES
------------------
The machine coder is deterministic, so this is NOT inter-rater reliability
between two independent coders. It measures whether the deterministic rules
capture what a trained human would call the same thing. Say that plainly in
the paper rather than implying two independent raters.

Outputs
-------
phase3_codebook/outputs/reports/codebook_apa_report.txt
phase3_codebook/outputs/reports/codebook_results.json
phase3_codebook/outputs/reports/codebook_results.csv
phase3_codebook/outputs/tables/codebook_confusion_role.csv
phase3_codebook/outputs/tables/codebook_confusion_status.csv
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
from a11ydebt.stats import TestResult, fmt_pct                # noqa: E402


def cohens_kappa(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Cohen's kappa plus observed and expected agreement."""
    categories = sorted(set(a.dropna()) | set(b.dropna()))
    index = {c: i for i, c in enumerate(categories)}
    n = len(a)
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


def bootstrap_kappa(a: pd.Series, b: pd.Series, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.arange(len(a))
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(values, size=len(values), replace=True)
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


def main() -> int:
    parser = base_parser(
        "Score the automated codebook against human coding.",
        epilog="Run after a human has filled in validation_sample_blank.csv.",
    )
    parser.add_argument("--coded", type=str, default=None,
                        help="path to the human-coded CSV "
                             "(default: outputs/validation_sample_blank.csv)")
    args = parser.parse_args()

    con = Console("02_intercoder_reliability", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 3 / STEP 2", "Codebook reliability against human coding")

    here = Path(__file__).resolve().parent
    outdir = here / "outputs"
    coded_path = Path(args.coded) if args.coded else outdir / "validation_sample_blank.csv"
    key_path = outdir / "validation_sample_key.csv"

    for path, hint in ((coded_path, "Code the blank sheet first."),
                       (key_path, "Run 01_draw_validation_sample.py first.")):
        if not path.exists():
            con.fail(f"Missing {path}", hint=hint)

    con.step(1, 4, "Load coding sheets")
    coded = pd.read_csv(coded_path)
    key = pd.read_csv(key_path)
    merged = coded[["item_id"] + [c for c in ("human_role", "human_label_status", "human_notes")
                                  if c in coded.columns]].merge(
        key[["item_id", "dataset", "class_name", "auto_role", "auto_label_status"]],
        on="item_id", how="inner")
    con.kv("Items in the sheet", len(coded))
    con.kv("Items matched to the key", len(merged))

    complete = merged.dropna(subset=["human_role", "human_label_status"])
    complete = complete[(complete["human_role"].astype(str).str.strip() != "")
                        & (complete["human_label_status"].astype(str).str.strip() != "")]
    con.kv("Items coded by a human", len(complete))
    if len(complete) < 30:
        con.fail(
            f"Only {len(complete)} coded items. Kappa is unstable below about 100 "
            f"and this project targets {CFG.analysis.codebook.n_double_coded}.",
            hint="Code more of validation_sample_blank.csv and re-run.",
        )
    if len(complete) < int(CFG.analysis.codebook.n_double_coded):
        con.warn(f"{len(complete)} coded, target is "
                 f"{CFG.analysis.codebook.n_double_coded}. Kappa will be wide.")

    con.step(2, 4, "Compute agreement")
    rep = ResultsReport(
        stem="codebook",
        title="Codebook reliability",
        subtitle="Automated element coding scored against human coding",
        outdir=outdir,
        script="phase3_codebook/02_intercoder_reliability.py",
    )
    rep.provenance(
        sample_source=str(coded_path), seed=args.seed,
        codebook="phase3_codebook/CODEBOOK.md",
        target_kappa=CFG.analysis.codebook.target_kappa,
    )
    rep.section(
        "What this measures",
        "The automated coder is deterministic, so these figures are not "
        "inter-rater reliability between two independent human coders. They "
        "measure whether the deterministic rules in taxonomy.py assign the "
        "same category a trained human would, using the definitions in "
        "CODEBOOK.md. Report it that way in the Method section.",
    )
    rep.stat("Items drawn", int(len(merged)))
    rep.stat("Items coded by a human", int(len(complete)))

    n_boot = min(2000, int(CFG.run.n_bootstrap))
    results = {}
    for variable, human_col, auto_col in (
        ("Element role", "human_role", "auto_role"),
        ("Label status", "human_label_status", "auto_label_status"),
    ):
        human = complete[human_col].astype(str).str.strip().str.lower()
        auto = complete[auto_col].astype(str).str.strip().str.lower()
        kappa, observed, n = cohens_kappa(human, auto)
        lo, hi = bootstrap_kappa(human, auto, n_boot, args.seed)
        results[variable] = {"kappa": kappa, "observed": observed, "n": n,
                             "ci": (lo, hi)}
        con.kv(f"{variable}: raw agreement", f"{100 * observed:.1f}%")
        con.kv(f"{variable}: Cohen's kappa", f"{kappa:.3f} [{lo:.3f}, {hi:.3f}]")

        rep.test(
            f"kappa_{auto_col}",
            TestResult(
                name=f"Cohen's kappa, {variable.lower()}",
                statistic_name="agreement", statistic=observed,
                p_value=float("nan"), df=None,
                effect_name="Cohen's kappa", effect_size=kappa,
                ci_low=lo, ci_high=hi, n=n,
                groups={"raw_agreement_pct": 100 * observed,
                        "n_categories": int(human.nunique())},
                note=f"CI from {n_boot:,} bootstrap replicates of the coded sample. "
                     f"Chance-corrected; raw agreement alone overstates reliability.",
            ),
            hypothesis=f"The automated rules reproduce human coding of {variable.lower()}.",
            interpretation=interpret_kappa(kappa),
        )

    con.step(3, 4, "Confusion matrices")
    for variable, human_col, auto_col, name in (
        ("Element role", "human_role", "auto_role", "confusion_role"),
        ("Label status", "human_label_status", "auto_label_status", "confusion_status"),
    ):
        matrix = pd.crosstab(
            complete[human_col].astype(str).str.strip().str.lower(),
            complete[auto_col].astype(str).str.strip().str.lower(),
        )
        rows = [[index] + [int(v) for v in row] for index, row in
                zip(matrix.index, matrix.values)]
        rep.table(Table(
            name, ["human \\ auto"] + list(matrix.columns), rows,
            caption=f"Confusion matrix for {variable.lower()}: rows are human "
                    f"codes, columns are automated codes.",
            note="Off-diagonal cells are where the rules and the human disagree. "
                 "Read them before revising the codebook: a concentrated "
                 "disagreement is a rule to fix, a scattered one is coder noise.",
        ))
        con.table(["human \\ auto"] + [str(c) for c in matrix.columns],
                  [[str(index)] + [str(v) for v in row]
                   for index, row in zip(matrix.index, matrix.values)])

    con.step(4, 4, "Verdict")
    target = float(CFG.analysis.codebook.target_kappa)
    below = [v for v, r in results.items() if not np.isnan(r["kappa"]) and r["kappa"] < target]
    if below:
        con.warn(f"Below the {target:.2f} target: {', '.join(below)}")
        con.note("Read the confusion matrices, revise taxonomy.py, re-run phase 2, "
                 "and re-code. Do not proceed to phase 5 with unreliable codes.")
        rep.limitation(
            f"Agreement for {', '.join(below)} fell below the pre-set target of "
            f"{target:.2f}. Results using those categories must be interpreted "
            f"with that in mind and the shortfall reported in the paper."
        )
    else:
        con.ok(f"All variables meet the {target:.2f} target")
        rep.note(f"All coded variables met the pre-set reliability target of {target:.2f}.")

    rep.note("Kappa is chance-corrected. Raw agreement is reported alongside it "
             "because the gap between the two is itself informative.")
    written = rep.write(verbose_console=con)
    con.done(written)
    con.info("Next:  python phase4_validation/01_validate_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
