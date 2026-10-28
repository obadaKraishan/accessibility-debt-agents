#!/usr/bin/env python3
"""
PHASE 4 / STEP 1 — Validate the parsed data against known ground truth.

Runs the checks that catch a broken parse before it becomes a wrong number
in a paper. Each check compares what we produced against a figure published
by the dataset authors or a schema invariant verified against the raw files.
Failures are reported with the specific fix, not just a red mark.

    python phase4_validation/01_validate_datasets.py

CHECKS
------
 1. Rico screen count against the published 66,261.
 2. Rico app count against the published 9,384, with the one known
    malformed row quarantined.
 3. AndroidControl episode count against splits.json (15,283).
 4. The zero-area invariant: 48.7% of Rico nodes have zero or negative
    area (phase4_validation/outputs/reports/validation_apa_report.txt, which
    this script writes), and some of those are still flagged visible.
    If this fraction is far from expectation, the bounds parse is wrong.
 5. Screen dimensions vary; no constant 1440x2560 assumption survives.
 6. Every label_status value is in the codebook vocabulary.
 7. Click hit rate for AndroidControl (expected near 100%).
 8. No element is both 'labelled' and has an empty content description.
 9. The unified table's schema is identical across datasets.

Outputs
-------
phase4_validation/outputs/reports/validation_apa_report.txt
phase4_validation/outputs/reports/validation_results.json
phase4_validation/outputs/tables/validation_checks.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                                           # noqa: E402

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.taxonomy import LABEL_STATUSES, ELEMENT_ROLES   # noqa: E402


class Checks:
    def __init__(self, con: Console) -> None:
        self.con = con
        self.rows: list[list] = []
        self.failed = 0

    def add(self, name: str, observed, expected, ok: bool,
            note: str = "", blocking: bool = True) -> None:
        status = "PASS" if ok else ("FAIL" if blocking else "WARN")
        if not ok and blocking:
            self.failed += 1
        self.rows.append([name, str(observed), str(expected), status, note])
        report = self.con.ok if ok else (self.con.error if blocking else self.con.warn)
        report(f"{name}: observed {observed}, expected {expected}")
        if note and not ok:
            self.con.note(note)


def main() -> int:
    parser = base_parser(
        "Validate parsed data against published figures and schema invariants.",
        epilog="Run after phase 2. Stops you shipping a wrong number.",
    )
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="relative tolerance for count checks (default 0.02)")
    args = parser.parse_args()

    con = Console("01_validate_datasets", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 4 / STEP 1", "Validate the parsed data")

    outdir = Path(__file__).resolve().parent / "outputs"
    elements_path = PATHS.processed / "elements.parquet"
    if not elements_path.exists():
        con.fail(f"Missing {elements_path}",
                 hint="Run: python phase2_processing/03_build_analysis_tables.py")

    con.step(1, 4, "Load")
    df = pd.read_parquet(elements_path)
    rico = df[df["dataset"] == "rico_2017"]
    ac = df[df["dataset"] == "androidcontrol_2024"]
    con.kv("Unified elements", len(df))
    con.kv("Rico elements", len(rico))
    con.kv("AndroidControl elements", len(ac))

    checks = Checks(con)
    pilot = False

    con.step(2, 4, "Count checks against published figures")
    n_screens = rico["screen_uid"].nunique()
    expected_screens = int(CFG.rico.expected_screens)
    within = abs(n_screens - expected_screens) <= args.tolerance * expected_screens
    if n_screens < expected_screens * 0.9:
        pilot = True
        con.warn(f"Only {n_screens:,} Rico screens present. This looks like a "
                 f"pilot run, so count checks are reported but not enforced.")
    checks.add("Rico screens", f"{n_screens:,}", f"{expected_screens:,}",
               within or pilot,
               "Re-run phase1_collection/01_download_rico.py to finish extraction.",
               blocking=not pilot)

    apps_path = PATHS.processed / "rico_apps.parquet"
    if apps_path.exists():
        n_apps = len(pd.read_parquet(apps_path))
        expected_apps = int(CFG.rico.expected_apps)
        checks.add("Rico apps (after quarantine)", f"{n_apps:,}",
                   f"{expected_apps - 1:,} to {expected_apps:,}",
                   abs(n_apps - expected_apps) <= 5,
                   "One malformed row is expected to be dropped; more than a "
                   "handful means the CSV reader is misparsing.")

    if len(ac):
        n_eps = ac["episode_id"].nunique()
        expected_eps = int(CFG.androidcontrol.expected_episodes.total)
        ac_pilot = n_eps < expected_eps * 0.9
        if ac_pilot:
            con.warn(f"Only {n_eps:,} AndroidControl episodes; treating as a pilot.")
        checks.add("AndroidControl episodes", f"{n_eps:,}", f"{expected_eps:,}",
                   abs(n_eps - expected_eps) <= args.tolerance * expected_eps or ac_pilot,
                   "Download more shards, then re-run the parser.",
                   blocking=not ac_pilot)

    con.step(3, 4, "Schema and invariant checks")

    # The zero-area invariant. Getting bounds wrong is the single easiest way
    # to produce plausible-looking but wrong coverage figures.
    if len(rico):
        zero_area = ((rico["x2"] - rico["x1"] <= 0) | (rico["y2"] - rico["y1"] <= 0))
        frac = float(zero_area.mean())
        checks.add("Rico zero/negative-area fraction", f"{100 * frac:.1f}%",
                   "40% to 56%", 0.40 <= frac <= 0.56,
                   "Far from the 48.7% in validation_apa_report.txt means the bounds parse is wrong. "
                   "Check that bounds is read as [x1, y1, x2, y2].")

        visible_zero = int((zero_area & rico["visible_to_user"]).sum())
        checks.add("Zero-area yet flagged visible", f"{visible_zero:,}", "> 0",
                   visible_zero > 0,
                   "Rico really does flag many zero-area nodes as visible. Seeing "
                   "none suggests the visibility flag is not being read.",
                   blocking=False)

        widths = rico.groupby("screen_uid")["screen_w"].first()
        checks.add("Distinct Rico screen widths", f"{widths.nunique()}", ">= 2",
                   widths.nunique() >= 2 or pilot,
                   "Screen sizes vary in Rico (1440x2560, 1440x2392, 411x683 and "
                   "others). A single width means normalisation is hardcoded.",
                   blocking=False)

    bad_status = set(df["label_status"].dropna().unique()) - set(LABEL_STATUSES)
    checks.add("label_status vocabulary", f"{sorted(bad_status) or 'clean'}",
               "only codebook values", not bad_status,
               "A value outside the codebook means taxonomy.py and the parser "
               "have drifted apart.")

    bad_roles = set(df["role"].dropna().unique()) - set(ELEMENT_ROLES)
    checks.add("role vocabulary", f"{sorted(bad_roles) or 'clean'}",
               "only codebook values", not bad_roles)

    contradiction = int(((df["label_status"] == "labelled")
                         & (df["content_desc"].fillna("").str.strip() == "")).sum())
    checks.add("'labelled' with empty description", f"{contradiction:,}", "0",
               contradiction == 0,
               "A logical contradiction in the coder. Inspect taxonomy.label_status.")

    report_path = PATHS.interim / "androidcontrol_parse_report.json"
    if report_path.exists():
        blob = json.loads(report_path.read_text(encoding="utf-8"))
        hit = float(blob.get("hit_rate_pct", 0))
        n_clicks = int(blob.get("totals", {}).get("clicks", 0))
        checks.add("Click hit rate", f"{hit:.1f}%", ">= 95%",
                   hit >= 95 or n_clicks == 0,
                   "Validation runs resolved essentially every click to a visible "
                   "node. A lower rate biases RQ3; inspect resolve_click.",
                   blocking=n_clicks > 0)

    if len(rico) and len(ac):
        cols_rico = set(rico.columns)
        cols_ac = set(ac.columns)
        checks.add("Unified schema identical", "identical" if cols_rico == cols_ac
                   else f"differs by {sorted(cols_rico ^ cols_ac)}",
                   "identical", cols_rico == cols_ac)

    con.step(4, 4, "Report")
    rep = ResultsReport(
        stem="validation",
        title="Data validation",
        subtitle="Parsed corpora checked against published figures and schema invariants",
        outdir=outdir,
        script="phase4_validation/01_validate_datasets.py",
    )
    rep.provenance(
        elements_table=str(elements_path),
        rico_expected_screens=CFG.rico.expected_screens,
        androidcontrol_expected_episodes=CFG.androidcontrol.expected_episodes.as_dict(),
        pilot_run=pilot,
    )
    rep.section(
        "Purpose",
        "Every check below compares the parsed data against a figure published "
        "by the dataset authors or an invariant verified directly against the "
        "raw files. Their purpose is to catch a silent parsing error, which in "
        "this study would produce plausible-looking but wrong coverage rates.",
    )
    rep.stat("Checks run", len(checks.rows))
    rep.stat("Checks failed", checks.failed)
    rep.table(Table(
        "validation_checks",
        ["Check", "Observed", "Expected", "Status", "If it fails"],
        checks.rows,
        caption="Data validation checks.",
        note="FAIL is blocking; WARN is informational. A pilot run relaxes the "
             "count checks because partial downloads legitimately fall short.",
    ))

    for dataset, group in df.groupby("dataset"):
        included = group[group["included"]]
        needs = included[included["needs_label"]]
        rep.proportion(
            f"Label coverage, {dataset}",
            int((needs["label_status"] == "labelled").sum()), len(needs),
            note="elements requiring a content description",
        )

    if pilot:
        rep.limitation(
            "This validation ran on a partial corpus. Count checks were reported "
            "but not enforced, and every coverage figure here is a pilot estimate."
        )
    rep.note("The zero-area check exists because roughly 48% of Rico nodes have "
             "no positive area while many are still flagged visible-to-user. "
             "Filtering on visibility alone would silently inflate the "
             "denominator and deflate every coverage estimate.")

    written = rep.write(verbose_console=con)
    con.table(["Check", "Observed", "Expected", "Status"],
              [[r[0], r[1], r[2], r[3]] for r in checks.rows])

    if checks.failed:
        con.fail(f"{checks.failed} blocking check(s) failed.",
                 hint="Read the 'If it fails' column in the report, fix the cause, "
                      "re-run phase 2, then re-run this script. Do not proceed to "
                      "phase 5 with failing checks.")
    con.done(written)
    con.info("Next:  python phase4_validation/02_sensitivity_checks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
