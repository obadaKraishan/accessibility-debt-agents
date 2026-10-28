#!/usr/bin/env python3
"""
PHASE 0 / STEP 2 — Initialise the project and prove the reporting pipeline works.

Two jobs:

  1. Create every directory the pipeline writes to, and copy .env.example to
     .env if you have not made one.
  2. Run the whole statistics-and-reporting stack end to end on a small
     SYNTHETIC dataset, so you can read a finished _apa_report.txt before any
     real data has been downloaded. If the format is not what you want for the
     manuscript, this is the cheapest moment to change it.

    python phase0_setup/02_init_project.py

Outputs
-------
outputs/manuscript/demo/reports/demo_apa_report.txt   <- READ THIS ONE
outputs/manuscript/demo/reports/demo_results.json
outputs/manuscript/demo/reports/demo_results.csv
outputs/manuscript/demo/tables/demo_coverage_by_role.csv
outputs/manuscript/project_manifest.json

The synthetic numbers are clearly marked as fake in the report header. They
exist only to exercise the code path.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                        # noqa: E402

from a11ydebt.cli import base_parser                      # noqa: E402
from a11ydebt.config import CFG, PATHS, PROJECT_ROOT      # noqa: E402
from a11ydebt.console import Console                      # noqa: E402
from a11ydebt.report import ResultsReport, Table          # noqa: E402
from a11ydebt.stats import (                              # noqa: E402
    chi_square_test, permutation_test, two_proportion_test,
)
from a11ydebt.taxonomy import (                           # noqa: E402
    ELEMENT_ROLES, LABEL_STATUSES, Role, classify_role, label_status,
)

DIRECTORIES = [
    "data/raw", "data/interim", "data/processed",
    "outputs/logs", "outputs/checkpoints", "outputs/manuscript",
    "phase5_analysis/rq1_label_coverage/outputs/tables",
    "phase5_analysis/rq1_label_coverage/outputs/figures",
    "phase5_analysis/rq1_label_coverage/outputs/reports",
    "phase5_analysis/rq2_temporal_change/outputs/tables",
    "phase5_analysis/rq2_temporal_change/outputs/figures",
    "phase5_analysis/rq2_temporal_change/outputs/reports",
    "phase5_analysis/rq3_trajectory_bias/outputs/tables",
    "phase5_analysis/rq3_trajectory_bias/outputs/figures",
    "phase5_analysis/rq3_trajectory_bias/outputs/reports",
    "phase5_analysis/rq4_difficulty_model/outputs/tables",
    "phase5_analysis/rq4_difficulty_model/outputs/figures",
    "phase5_analysis/rq4_difficulty_model/outputs/reports",
    "phase6_synthesis/outputs",
]


def make_directories(con: Console) -> None:
    created = 0
    for rel in DIRECTORIES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created += 1
        (path / ".gitkeep").touch(exist_ok=True)
    PATHS.ensure()
    con.kv("Directories checked", len(DIRECTORIES))
    con.kv("Directories created", created)
    con.ok("Directory tree is in place")


def make_dotenv(con: Console) -> None:
    env = PROJECT_ROOT / ".env"
    example = PROJECT_ROOT / ".env.example"
    if env.exists():
        con.ok(".env already exists; leaving it alone")
    elif example.exists():
        shutil.copy(example, env)
        con.ok("Created .env from .env.example")
        con.note("Edit it if you want data on a different disk.")
    else:
        con.warn(".env.example is missing; defaults will be used")


def taxonomy_self_test(con: Console) -> bool:
    """A handful of assertions that catch a broken codebook immediately."""
    cases = [
        # (class_name, content_desc, text, clickable, expected_role, expected_status)
        ("android.widget.ImageButton", "", "", True, Role.BUTTON, "unlabelled"),
        ("android.widget.ImageButton", "Share", "", True, Role.BUTTON, "labelled"),
        ("android.widget.TextView", "", "Sign in", False, Role.TEXT, "named_by_text"),
        ("android.widget.ImageView", "", "", False, Role.IMAGE, "unlabelled"),
        ("android.widget.LinearLayout", "", "", False, Role.CONTAINER, "not_applicable"),
        ("androidx.appcompat.widget.AppCompatImageView", "", "", True, Role.IMAGE, "unlabelled"),
        ("com.google.android.material.floatingactionbutton.FloatingActionButton",
         "", "", True, Role.BUTTON, "unlabelled"),
        ("android.widget.EditText", "", "", True, Role.INPUT, "unlabelled"),
        ("android.webkit.WebView", "", "", True, Role.WEB, "not_applicable"),
    ]
    rows, ok = [], True
    for class_name, desc, text, clickable, want_role, want_status in cases:
        role = classify_role(class_name)
        status = label_status(role, desc, text, clickable)
        passed = (role == want_role) and (status == want_status)
        ok &= passed
        rows.append([
            class_name.rsplit(".", 1)[-1], role, status,
            "ok" if passed else f"EXPECTED {want_role}/{want_status}",
        ])
    con.table(["Class", "Role", "Label status", "Check"], rows)
    return ok


def synthetic_frame(seed: int) -> "np.ndarray":
    """Fake elements with a deliberate, known coverage gap between datasets."""
    rng = np.random.default_rng(seed)
    records = []
    specs = [
        # dataset,      n,     P(labelled | needs a label)
        ("rico_2017", 12000, 0.15),
        ("androidcontrol_2024", 9000, 0.56),
    ]
    roles = [Role.IMAGE, Role.BUTTON, Role.TEXT, Role.INPUT, Role.CONTAINER]
    weights = [0.22, 0.13, 0.30, 0.08, 0.27]
    for dataset, n, p_label in specs:
        chosen = rng.choice(roles, size=n, p=weights)
        for role in chosen:
            clickable = role in {Role.BUTTON, Role.INPUT} or rng.random() < 0.25
            text = "Label" if (role == Role.TEXT and rng.random() < 0.9) else ""
            needs = role in {Role.IMAGE, Role.BUTTON, Role.INPUT} and not text
            desc = "described" if (needs and rng.random() < p_label) else ""
            records.append((dataset, role, desc, text, clickable))
    return records


def demo_report(con: Console, seed: int) -> dict:
    """Exercise every reporting feature so the txt format can be reviewed now."""
    outdir = PATHS.manuscript / "demo"
    records = synthetic_frame(seed)

    rep = ResultsReport(
        stem="demo",
        title="Reporting pipeline self-test",
        subtitle="SYNTHETIC DATA — these numbers are fabricated and mean nothing",
        outdir=outdir,
        script="phase0_setup/02_init_project.py",
    )
    rep.provenance(
        datasets={"synthetic": f"{len(records):,} generated elements"},
        seed=seed,
        warning="SYNTHETIC. Generated to verify formatting, not to report findings.",
    )

    rep.section(
        "Purpose",
        "This report was produced by the phase 0 self-test. It proves that the "
        "statistics module, the effect-size machinery and the report writer all "
        "work end to end on this machine. Read it to confirm that the layout, "
        "the level of statistical detail and the APA formatting are what you "
        "want before real results are generated. Every number below is fake.",
    )

    # ---- descriptives
    rep.section("Sample and inclusion")
    rep.stat("Synthetic elements generated", len(records))
    rep.stat("Datasets compared", 2)

    by_dataset: dict[str, list] = {}
    for dataset, role, desc, text, clickable in records:
        by_dataset.setdefault(dataset, []).append((role, desc, text, clickable))

    counts = {}
    for dataset, rows in by_dataset.items():
        statuses = [label_status(r, d, t, c) for r, d, t, c in rows]
        needs = sum(1 for s in statuses if s in {"labelled", "unlabelled"})
        labelled = sum(1 for s in statuses if s == "labelled")
        counts[dataset] = {"needs": needs, "labelled": labelled, "statuses": statuses}
        rep.proportion(
            f"Label coverage, {dataset}", labelled, needs,
            note="of elements that require a content description",
        )

    # ---- table
    table_rows = []
    for dataset, rows in by_dataset.items():
        for role in ELEMENT_ROLES:
            subset = [(r, d, t, c) for r, d, t, c in rows if r == role]
            if not subset:
                continue
            statuses = [label_status(r, d, t, c) for r, d, t, c in subset]
            needs = sum(1 for s in statuses if s in {"labelled", "unlabelled"})
            lab = sum(1 for s in statuses if s == "labelled")
            table_rows.append([
                dataset, role, len(subset), needs, lab,
                round(100 * lab / needs, 2) if needs else "n/a",
            ])
    rep.table(Table(
        "coverage_by_role",
        ["Dataset", "Role", "Elements", "Needs label", "Labelled", "Coverage %"],
        table_rows,
        caption="Label coverage by element role and dataset (synthetic).",
        note="Elements exempt from labelling (containers, text with visible text, "
             "web views) are counted under Elements but not under Needs label.",
    ))

    # ---- tests
    rep.section("Inferential results")
    a, b = "rico_2017", "androidcontrol_2024"
    rep.test(
        "coverage_gap",
        two_proportion_test(
            counts[b]["labelled"], counts[b]["needs"],
            counts[a]["labelled"], counts[a]["needs"],
            label1=b, label2=a,
        ),
        hypothesis="H1. Label coverage is higher in the newer corpus than in the older one.",
    )

    role_labels = [r for r in ELEMENT_ROLES if any(row[1] == r for row in table_rows)]
    contingency = []
    for role in role_labels:
        lab = sum(int(row[4]) for row in table_rows if row[1] == role)
        unlab = sum(int(row[3]) - int(row[4]) for row in table_rows if row[1] == role)
        contingency.append([lab, unlab])
    rep.test(
        "role_dependence",
        chi_square_test(contingency, row_labels=role_labels,
                        col_labels=["labelled", "unlabelled"]),
        hypothesis="H2. Label status is not independent of element role.",
    )

    rng = np.random.default_rng(seed)
    touched = rng.binomial(1, 0.42, size=1800)
    background = rng.binomial(1, 0.31, size=9000)
    rep.test(
        "trajectory_bias",
        permutation_test(
            touched, background, n_perm=2000,
            name="Permutation test (touched vs background)",
            effect_name="difference in label rate",
        ),
        hypothesis="H3. Elements touched by task trajectories are labelled at a "
                   "higher rate than the background population.",
    )

    rep.narrative(
        "This paragraph shows where hand-written prose lands in the draft. "
        "Anything passed to rep.narrative() appears above the auto-composed "
        "sentences, so you can lead with your own framing and let the machinery "
        "supply the statistics."
    )
    rep.note("Wilson score intervals are used for all proportions.")
    rep.note("Permutation p-values use the (r+1)/(m+1) convention and are never zero.")
    rep.limitation("Synthetic data. No inference about real corpora is possible here.")

    return rep.write(verbose_console=con)


def main() -> int:
    parser = base_parser(
        "Create the project's directories and self-test the reporting pipeline.",
        epilog="Safe to re-run at any time. Never touches downloaded data.",
    )
    args = parser.parse_args()

    con = Console("02_init_project", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 0 / STEP 2", "Initialise the project and self-test the reporting stack")

    con.step(1, 4, "Directory tree")
    make_directories(con)

    con.step(2, 4, "Environment file")
    make_dotenv(con)

    con.step(3, 4, "Taxonomy self-test")
    con.note("Nine hand-checked cases through classify_role() and label_status().")
    if not taxonomy_self_test(con):
        con.fail(
            "The element taxonomy did not behave as specified.",
            hint="src/a11ydebt/taxonomy.py has been edited in a way that breaks the "
                 "codebook. Revert it, or update both the rules and the expected "
                 "cases in this script together.",
        )
    con.ok("All taxonomy cases behave as documented in phase3_codebook/CODEBOOK.md")

    con.step(4, 4, "Reporting pipeline self-test (synthetic data)")
    written = demo_report(con, seed=args.seed)

    manifest = {
        "initialised_at": datetime.now().isoformat(timespec="seconds"),
        "project": CFG.as_dict().get("project", {}),
        "paths": {k: str(v) for k, v in vars(PATHS).items()},
        "roles": list(ELEMENT_ROLES),
        "label_statuses": list(LABEL_STATUSES),
        "seed": args.seed,
    }
    manifest_path = PATHS.manuscript / "project_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    written["project_manifest.json"] = manifest_path

    con.done(written)
    con.info("Open the demo report and check the format:")
    con.info(f"  {PATHS.manuscript / 'demo' / 'reports' / 'demo_apa_report.txt'}")
    con.info("")
    con.info("Next:  python phase1_collection/01_download_rico.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
