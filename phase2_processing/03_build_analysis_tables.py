#!/usr/bin/env python3
"""
PHASE 2 / STEP 3 — Build the unified analysis tables.

Joins the two parsed corpora into one element table with an identical schema,
attaches Rico app metadata (category, rating, install range), and builds the
touched-versus-background table that RQ3 needs. Everything downstream reads
only the files this script writes, so the RQ scripts never touch raw data.

    python phase2_processing/03_build_analysis_tables.py

CROSS-DATASET FAIRNESS
----------------------
Only fields both corpora can supply survive into elements.parquet. Rico's
`ancestors` and AndroidControl's `hint_text`, `editable` and `labeled_by_id`
are dropped from the unified table and kept in the per-dataset files for
sensitivity analysis. Any comparison between 2017 and 2024 therefore rests
on identical information, which is what makes RQ2 defensible.

Outputs
-------
data/processed/elements.parquet        unified, one row per element
data/processed/screens.parquet         unified, one row per screen
data/processed/rq3_touched.parquet     touched vs background, AndroidControl
data/interim/build_tables_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                                           # noqa: E402

from a11ydebt.checkpoint import atomic_write_json             # noqa: E402
from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import PATHS                             # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.taxonomy import CANONICAL_NODE_FIELDS           # noqa: E402

UNIFIED_COLUMNS = [
    "dataset", "screen_uid", "node_uid", "depth",
    "class_name", "content_desc", "text", "resource_id",
    "clickable", "long_clickable", "enabled", "focusable", "visible_to_user",
    "x1", "y1", "x2", "y2", "screen_w", "screen_h",
    "role", "label_status", "needs_label", "icon_like", "is_fab", "included",
    "app_package", "app_category", "app_rating", "app_downloads",
    "episode_id", "step_index", "was_touched",
]


def require(path: Path, con: Console, hint: str) -> None:
    if not path.exists():
        con.fail(f"Missing input: {path}", hint=hint)


def main() -> int:
    parser = base_parser(
        "Join the parsed corpora into the unified analysis tables.",
        epilog="Fast; reads only parquet written by the two parsers.",
    )
    args = parser.parse_args()

    con = Console("03_build_analysis_tables", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 2 / STEP 3", "Build the unified analysis tables")

    rico_nodes_path = PATHS.processed / "rico_nodes.parquet"
    ac_nodes_path = PATHS.processed / "androidcontrol_nodes.parquet"
    require(rico_nodes_path, con, "Run: python phase2_processing/01_parse_rico.py")

    con.step(1, 5, "Load Rico elements")
    rico = pd.read_parquet(rico_nodes_path)
    con.kv("Rico elements", len(rico))

    apps_path = PATHS.processed / "rico_apps.parquet"
    if apps_path.exists():
        apps = pd.read_parquet(apps_path)
        rico = rico.merge(
            apps[["app_package", "category", "avg_rating", "downloads_range"]],
            on="app_package", how="left",
        ).rename(columns={"category": "app_category", "avg_rating": "app_rating",
                          "downloads_range": "app_downloads"})
        matched = rico["app_category"].notna().sum()
        con.kv("Elements with app metadata", int(matched))
        con.kv("Coverage", f"{100 * matched / max(1, len(rico)):.1f}%")
    else:
        rico["app_category"] = pd.NA
        rico["app_rating"] = pd.NA
        rico["app_downloads"] = pd.NA
        con.warn("rico_apps.parquet missing; app-level analysis in RQ1 will be empty")

    rico["episode_id"] = pd.NA
    rico["step_index"] = pd.NA
    rico["was_touched"] = False

    con.step(2, 5, "Load AndroidControl elements")
    if ac_nodes_path.exists():
        ac = pd.read_parquet(ac_nodes_path)
        con.kv("AndroidControl elements", len(ac))
        ac["app_package"] = ""
        ac["app_category"] = pd.NA
        ac["app_rating"] = pd.NA
        ac["app_downloads"] = pd.NA
        con.note("AndroidControl ships no Play Store metadata, so app category, "
                 "rating and installs exist for Rico only. RQ1's category "
                 "breakdown is therefore a Rico-only analysis; say so in the paper.")
    else:
        ac = pd.DataFrame(columns=rico.columns)
        con.warn("androidcontrol_nodes.parquet not found. Continuing with Rico only.")
        con.note("Run: python phase2_processing/02_parse_androidcontrol.py")

    con.step(3, 5, "Unify")
    for frame in (rico, ac):
        for column in UNIFIED_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
    elements = pd.concat([rico[UNIFIED_COLUMNS], ac[UNIFIED_COLUMNS]], ignore_index=True)

    for column in ("clickable", "long_clickable", "enabled", "focusable",
                   "visible_to_user", "icon_like", "is_fab", "included", "was_touched",
                   "needs_label"):
        elements[column] = elements[column].fillna(False).astype(bool)

    elements_path = PATHS.processed / "elements.parquet"
    elements.to_parquet(elements_path, compression="zstd", index=False)
    con.kv("Unified elements", len(elements))
    con.table(
        ["Dataset", "Elements", "Included", "Needs label", "Labelled", "Coverage %"],
        [
            [
                name,
                f"{len(g):,}",
                f"{int(g['included'].sum()):,}",
                f"{int(((g['included']) & g['needs_label']).sum()):,}",
                f"{int(((g['included']) & g['needs_label'] & (g['label_status'] == 'labelled')).sum()):,}",
                (f"{100 * ((g['included']) & g['needs_label'] & (g['label_status'] == 'labelled')).sum() / max(1, ((g['included']) & g['needs_label']).sum()):.2f}"),
            ]
            for name, g in elements.groupby("dataset")
        ],
    )
    con.note("These are the headline RQ1 numbers. If they look implausible, stop "
             "here and check the codebook before running any analysis.")

    con.step(4, 5, "Unify screens")
    frames = []
    for path in (PATHS.processed / "rico_screens.parquet",
                 PATHS.processed / "androidcontrol_episodes.parquet"):
        if path.exists():
            frames.append(pd.read_parquet(path))
    if frames:
        screens = pd.concat(frames, ignore_index=True)
        screens_path = PATHS.processed / "screens.parquet"
        screens.to_parquet(screens_path, compression="zstd", index=False)
        con.kv("Screen and episode rows", len(screens))
    else:
        screens_path = None
        con.warn("No screen-level tables found")

    con.step(5, 5, "Build the RQ3 touched-versus-background table")
    rq3_path = PATHS.processed / "rq3_touched.parquet"
    if len(ac) and "was_touched" in ac.columns:
        # The background is every included element on a screen that a trajectory
        # actually visited, so the comparison is within-screen and not confounded
        # by which screens agents happen to reach.
        touched_screens = set(ac.loc[ac["was_touched"], "screen_uid"].unique())
        rq3 = ac[ac["screen_uid"].isin(touched_screens) & ac["included"]].copy()
        rq3["is_labelled"] = (rq3["label_status"] == "labelled").astype(int)
        rq3.to_parquet(rq3_path, compression="zstd", index=False)
        con.kv("Screens with at least one touch", len(touched_screens))
        con.kv("Elements on those screens", len(rq3))
        con.kv("Touched elements", int(rq3["was_touched"].sum()))
        n_touch = int(rq3["was_touched"].sum())
        if n_touch < 200:
            con.warn(f"Only {n_touch} touched elements. RQ3 needs more; parse "
                     f"more shards before running it.")
        else:
            con.ok("Enough touched elements for the RQ3 permutation test")
    else:
        con.warn("Skipped: AndroidControl elements are required for RQ3")

    report_path = PATHS.interim / "build_tables_report.json"
    atomic_write_json(report_path, {
        "unified_elements": int(len(elements)),
        "by_dataset": {k: int(v) for k, v in elements["dataset"].value_counts().items()},
        "canonical_fields": list(CANONICAL_NODE_FIELDS),
        "unified_columns": UNIFIED_COLUMNS,
    })

    written = {"elements.parquet": elements_path, "build_tables_report.json": report_path}
    if screens_path:
        written["screens.parquet"] = screens_path
    if rq3_path.exists():
        written["rq3_touched.parquet"] = rq3_path
    con.done(written)
    con.info("Next:  python phase4_validation/01_validate_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
