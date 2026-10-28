#!/usr/bin/env python3
"""
PHASE 3 / STEP 1 — Draw the stratified sample for codebook validation.

Produces two CSVs: one blank sheet for a human to code, and one answer key
holding the automated codes. Coding against the blank sheet keeps the human
blind to the machine's answers, which is the whole point.

    python phase3_codebook/01_draw_validation_sample.py
    python phase3_codebook/01_draw_validation_sample.py --n-screens 80

Sampling is stratified by dataset, element role, and automated label status,
so rare cells (a labelled floating action button, an unlabelled input) are
represented rather than swamped by the common ones. The seed is recorded in
the output so the draw is reproducible.

Outputs
-------
phase3_codebook/outputs/validation_sample_blank.csv     <- code this one
phase3_codebook/outputs/validation_sample_key.csv       <- automated codes
phase3_codebook/outputs/validation_sample_meta.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                                           # noqa: E402

from a11ydebt.checkpoint import atomic_write_json             # noqa: E402
from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402

HUMAN_COLUMNS = ["human_role", "human_label_status", "human_notes"]
CONTEXT_COLUMNS = [
    "dataset", "screen_uid", "node_uid", "class_name",
    "content_desc", "text", "resource_id",
    "clickable", "long_clickable", "visible_to_user",
    "x1", "y1", "x2", "y2", "screen_w", "screen_h",
]


def main() -> int:
    parser = base_parser(
        "Draw a stratified sample of elements for human coding.",
        epilog="Run after phase 2. Produces a blank sheet and a hidden answer key.",
    )
    parser.add_argument("--n-screens", type=int, default=None,
                        help="screens per dataset (default: config codebook.n_screens_per_dataset)")
    parser.add_argument("--per-screen", type=int, default=None,
                        help="max elements per screen (default: config codebook.n_elements_per_screen)")
    args = parser.parse_args()

    con = Console("01_draw_validation_sample", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 3 / STEP 1", "Draw the codebook validation sample")

    elements_path = PATHS.processed / "elements.parquet"
    if not elements_path.exists():
        con.fail(f"Missing {elements_path}",
                 hint="Run: python phase2_processing/03_build_analysis_tables.py")

    n_screens = args.n_screens or int(CFG.analysis.codebook.n_screens_per_dataset)
    per_screen = args.per_screen or int(CFG.analysis.codebook.n_elements_per_screen)
    outdir = Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    con.step(1, 4, "Load elements")
    df = pd.read_parquet(elements_path)
    df = df[df["included"]].copy()
    con.kv("Included elements", len(df))
    con.kv("Datasets", df["dataset"].nunique())

    con.step(2, 4, "Sample screens")
    frames = []
    for dataset, group in df.groupby("dataset"):
        screens = group["screen_uid"].drop_duplicates()
        take = min(n_screens, len(screens))
        chosen = screens.sample(take, random_state=args.seed)
        frames.append(group[group["screen_uid"].isin(set(chosen))])
        con.kv(f"{dataset}: screens sampled", take)
    pool = pd.concat(frames, ignore_index=True)

    con.step(3, 4, "Stratify within screens")
    # Stratify on role x label_status so rare cells survive the draw.
    pool["_stratum"] = pool["dataset"] + "|" + pool["role"] + "|" + pool["label_status"]
    per_stratum = max(1, per_screen)
    sample = (
        pool.groupby(["screen_uid", "_stratum"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=args.seed))
        .reset_index(drop=True)
    )
    con.kv("Sampled elements", len(sample))
    con.table(
        ["Dataset", "Role", "Status", "n"],
        [[d, r, s, f"{n:,}"] for (d, r, s), n in
         sample.groupby(["dataset", "role", "label_status"]).size().items()],
    )

    con.step(4, 4, "Write sheets")
    key = sample[CONTEXT_COLUMNS + ["role", "label_status", "icon_like"]].copy()
    key = key.rename(columns={"role": "auto_role", "label_status": "auto_label_status"})
    key.insert(0, "item_id", range(1, len(key) + 1))

    blank = key[["item_id"] + CONTEXT_COLUMNS].copy()
    for column in HUMAN_COLUMNS:
        blank[column] = ""

    blank_path = outdir / "validation_sample_blank.csv"
    key_path = outdir / "validation_sample_key.csv"
    blank.to_csv(blank_path, index=False)
    key.to_csv(key_path, index=False)

    meta_path = outdir / "validation_sample_meta.json"
    atomic_write_json(meta_path, {
        "seed": args.seed,
        "n_screens_per_dataset": n_screens,
        "max_per_screen_stratum": per_stratum,
        "n_items": int(len(blank)),
        "target_kappa": CFG.analysis.codebook.target_kappa,
        "strata": int(sample["_stratum"].nunique()),
        "instructions": (
            "Code human_role and human_label_status using phase3_codebook/CODEBOOK.md. "
            "Do not open validation_sample_key.csv until coding is finished."
        ),
    })

    con.done({"blank sheet (code this)": blank_path,
              "answer key (do not peek)": key_path,
              "metadata": meta_path})
    con.info("Code the blank sheet against phase3_codebook/CODEBOOK.md, then run:")
    con.info("  python phase3_codebook/02_intercoder_reliability.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
