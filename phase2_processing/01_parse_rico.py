#!/usr/bin/env python3
"""
PHASE 2 / STEP 1 — Parse Rico view hierarchies into the canonical node table.

Walks every extracted Rico screen, flattens its view hierarchy depth-first,
applies the codebook in src/a11ydebt/taxonomy.py, and writes one row per UI
element. This is the table every Rico number in the paper is computed from.

    python phase2_processing/01_parse_rico.py
    python phase2_processing/01_parse_rico.py --limit 2000      # pilot
    python phase2_processing/01_parse_rico.py --restart

Roughly 6.4 million elements across 66,261 screens. Checkpointed every 500
screens, so interrupting and re-running is safe and cheap.

SCHEMA FACTS THIS PARSER RELIES ON (all verified against the real archive)
-------------------------------------------------------------------------
* The root node is at d["activity"]["root"]. Top level has exactly four keys.
* "children" is ABSENT on leaves rather than being an empty list.
* "content-desc" is ALWAYS a list of length one; it is [None] when absent,
  never a bare string.
* "bounds" is [x1, y1, x2, y2] in absolute pixels.
* There is no "scrollable" key; it is split into scrollable-horizontal and
  scrollable-vertical.
* "pressed" is a string ("not_pressed"), not a boolean.
* "pointer" is NOT unique across nodes, so it is never used as an
  identifier; node ids (node_uid) are assigned here by traversal order.
* 48.7% of nodes have zero or negative area, and 1,716 of those are still
  flagged visible-to-user (phase4_validation/outputs/reports/validation_apa_report.txt),
  so area is filtered separately from visibility.
* Screen dimensions vary (1440x2560, 1440x2392, 411x683 and others), so
  normalisation uses each screen's own root bounds, never a constant.

Outputs
-------
data/processed/rico_nodes.parquet      one row per element
data/processed/rico_screens.parquet    one row per screen
data/processed/rico_apps.parquet       cleaned app_details
data/interim/rico_parse_report.json    counts, failures, timings
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow as pa                                          # noqa: E402
import pyarrow.parquet as pq                                  # noqa: E402

from a11ydebt.checkpoint import Checkpoint, atomic_write_json  # noqa: E402
from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.taxonomy import (                               # noqa: E402
    classify_role, is_floating_action_button, is_icon_like,
    in_analysis_population, label_status, needs_label,
)

BATCH_ROWS = 250_000
DATASET = "rico_2017"

NODE_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("screen_uid", pa.string()),
    ("node_uid", pa.int32()), ("parent_uid", pa.int32()), ("depth", pa.int16()),
    ("class_name", pa.string()), ("content_desc", pa.string()), ("text", pa.string()),
    ("resource_id", pa.string()),
    ("clickable", pa.bool_()), ("long_clickable", pa.bool_()),
    ("enabled", pa.bool_()), ("focusable", pa.bool_()),
    ("visible_to_user", pa.bool_()), ("visibility", pa.string()),
    ("x1", pa.int32()), ("y1", pa.int32()), ("x2", pa.int32()), ("y2", pa.int32()),
    ("screen_w", pa.int32()), ("screen_h", pa.int32()),
    ("role", pa.string()), ("role_with_ancestors", pa.string()),
    ("label_status", pa.string()), ("needs_label", pa.bool_()),
    ("icon_like", pa.bool_()), ("is_fab", pa.bool_()), ("included", pa.bool_()),
    ("app_package", pa.string()),
])

SCREEN_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("screen_uid", pa.string()),
    ("app_package", pa.string()), ("activity_name", pa.string()),
    ("screen_w", pa.int32()), ("screen_h", pa.int32()),
    ("n_nodes", pa.int32()), ("n_included", pa.int32()),
    ("n_needs_label", pa.int32()), ("n_labelled", pa.int32()),
    ("n_icon_like", pa.int32()), ("n_icon_like_labelled", pa.int32()),
    ("max_depth", pa.int16()),
])


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

def read_ui_details(path: Path, con: Console) -> dict[str, str]:
    """Map screen id -> app package. Columns: UI Number, App Package Name, ..."""
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            uid = (row.get("UI Number") or "").strip()
            pkg = (row.get("App Package Name") or "").strip()
            if uid:
                mapping[uid] = pkg
    con.kv("Screens in ui_details.csv", len(mapping))
    return mapping


def read_app_details(path: Path, con: Console) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Read app_details.csv, repairing the one known malformed row.

    'net.ayudaporfavor.Love' has an unescaped double quote in its Play Store
    Name, which shifts every later column by one and creates a bogus 28th
    category value. Rows whose Category is not a plausible category string
    are quarantined rather than silently poisoning the category analysis.
    """
    rows: list[dict[str, Any]] = []
    bad: list[str] = []
    known_bad = set(CFG.rico.known_bad_app_rows)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            pkg = (row.get("App Package Name") or "").strip()
            category = (row.get("Category") or "").strip()
            if pkg in known_bad or not category or any(ch.isdigit() for ch in category[:3]):
                bad.append(pkg or "<blank>")
                continue

            def as_float(value: str | None) -> float | None:
                try:
                    return float(str(value).strip())
                except (TypeError, ValueError):
                    return None

            def as_int(value: str | None) -> int | None:
                try:
                    return int(str(value).replace(",", "").strip())
                except (TypeError, ValueError):
                    return None

            rows.append({
                "app_package": pkg,
                "play_store_name": (row.get("Play Store Name") or "").strip(),
                "category": category,
                "avg_rating": as_float(row.get("Average Rating")),
                "n_ratings": as_int(row.get("Number of Ratings")),
                # This column is a padded string range like "  100,000 - 500,000  ".
                "downloads_range": (row.get("Number of Downloads") or "").strip(),
                "date_updated": (row.get("Date Updated") or "").strip(),
            })
    con.kv("Apps parsed", len(rows))
    if bad:
        con.warn(f"Quarantined {len(bad)} malformed app row(s): {', '.join(bad[:5])}")
        con.note("This is the documented unescaped-quote defect in app_details.csv.")
    return rows, bad


# --------------------------------------------------------------------------
# hierarchy walking
# --------------------------------------------------------------------------

def _content_desc(node: dict) -> str:
    """content-desc is always a list of length 1; [None] means absent."""
    value = node.get("content-desc")
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null"} else text


def _bounds(node: dict) -> tuple[int, int, int, int]:
    b = node.get("bounds") or [0, 0, 0, 0]
    try:
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    except (TypeError, ValueError, IndexError):
        return 0, 0, 0, 0


def walk_screen(
    blob: dict, screen_uid: str, app_package: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flatten one screen depth-first into canonical node rows."""
    root = (blob.get("activity") or {}).get("root")
    if not isinstance(root, dict):
        raise ValueError("no activity.root")

    sw = max(1, _bounds(root)[2] - _bounds(root)[0])
    sh = max(1, _bounds(root)[3] - _bounds(root)[1])
    activity_name = str(blob.get("activity_name") or "")

    rows: list[dict[str, Any]] = []
    counter = 0
    max_depth = 0
    stack: list[tuple[dict, int, int]] = [(root, -1, 0)]

    while stack:
        node, parent_uid, depth = stack.pop()
        node_uid = counter
        counter += 1
        max_depth = max(max_depth, depth)

        class_name = str(node.get("class") or "")
        ancestors = node.get("ancestors") or []
        content_desc = _content_desc(node)
        text = str(node.get("text") or "").strip()
        clickable = bool(node.get("clickable"))
        long_clickable = bool(node.get("long-clickable"))
        visible = bool(node.get("visible-to-user"))
        enabled = bool(node.get("enabled"))
        x1, y1, x2, y2 = _bounds(node)

        role = classify_role(class_name)
        rows.append({
            "dataset": DATASET, "screen_uid": screen_uid,
            "node_uid": node_uid, "parent_uid": parent_uid, "depth": depth,
            "class_name": class_name, "content_desc": content_desc, "text": text,
            "resource_id": str(node.get("resource-id") or ""),
            "clickable": clickable, "long_clickable": long_clickable,
            "enabled": enabled, "focusable": bool(node.get("focusable")),
            "visible_to_user": visible,
            "visibility": str(node.get("visibility") or ""),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "screen_w": sw, "screen_h": sh,
            "role": role,
            "role_with_ancestors": classify_role(class_name, ancestors),
            "label_status": label_status(role, content_desc, text, clickable, long_clickable),
            # Explicit, so coverage denominators never include exempt elements.
            # A container that happens to carry a description is 'labelled' but
            # was never required to have one; counting it would make container
            # coverage 100% by construction.
            "needs_label": needs_label(role, clickable, long_clickable, text),
            "icon_like": is_icon_like(role, text, clickable, long_clickable),
            "is_fab": is_floating_action_button(class_name),
            "included": in_analysis_population(visible, x1, y1, x2, y2, enabled),
            "app_package": app_package,
        })

        # 'children' is absent on leaves rather than an empty list.
        children = node.get("children") or []
        for child in reversed(children):
            if isinstance(child, dict):
                stack.append((child, node_uid, depth + 1))

    included = [r for r in rows if r["included"]]
    needs = [r for r in included if r["needs_label"]]
    icons = [r for r in included if r["icon_like"]]
    summary = {
        "dataset": DATASET, "screen_uid": screen_uid, "app_package": app_package,
        "activity_name": activity_name, "screen_w": sw, "screen_h": sh,
        "n_nodes": len(rows), "n_included": len(included),
        "n_needs_label": len(needs),
        "n_labelled": sum(1 for r in needs if r["label_status"] == "labelled"),
        "n_icon_like": len(icons),
        "n_icon_like_labelled": sum(1 for r in icons if r["label_status"] == "labelled"),
        "max_depth": max_depth,
    }
    return rows, summary


def iter_screen_files(directory: Path, limit: int | None) -> Iterator[Path]:
    count = 0
    for path in sorted(directory.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        if path.name.startswith("._"):
            continue
        yield path
        count += 1
        if limit and count >= limit:
            return


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = base_parser(
        "Parse Rico view hierarchies into the canonical element table.",
        epilog="Checkpointed every 500 screens; safe to interrupt and re-run.",
    )
    args = parser.parse_args()

    con = Console("01_parse_rico", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 2 / STEP 1", "Parse Rico view hierarchies")

    screens_dir = PATHS.rico_screens
    if not screens_dir.exists() or not any(screens_dir.glob("*.json")):
        con.fail(
            f"No Rico screens found in {screens_dir}",
            hint="Run:  python phase1_collection/01_download_rico.py",
        )

    con.step(1, 4, "Read metadata")
    ui_details = read_ui_details(PATHS.rico_raw / "ui_details.csv", con)
    apps, bad_apps = read_app_details(PATHS.rico_raw / "app_details.csv", con)
    apps_path = PATHS.processed / "rico_apps.parquet"
    pq.write_table(pa.Table.from_pylist(apps), apps_path)
    con.ok(f"Wrote {apps_path.name}")

    con.step(2, 4, "Enumerate screens")
    files = list(iter_screen_files(screens_dir, args.limit))
    con.kv("Screen files found", len(files))
    if args.limit:
        con.warn(f"Pilot run: limited to {args.limit:,} screens. Figures from this "
                 f"run are estimates, not the census.")

    ck = Checkpoint("parse_rico", PATHS.checkpoints,
                    flush_every=int(CFG.run.checkpoint_every), restart=args.restart)
    nodes_path = PATHS.processed / "rico_nodes.parquet"
    screens_path = PATHS.processed / "rico_screens.parquet"
    if args.restart:
        nodes_path.unlink(missing_ok=True)
        screens_path.unlink(missing_ok=True)
    if ck.resumed and nodes_path.exists():
        con.warn(f"A checkpoint exists ({ck.n_done:,} screens done) but parquet "
                 f"output cannot be appended to safely across runs.")
        con.note("Re-running with --restart is the reliable option. Continuing "
                 "would risk duplicate rows, so this script restarts the parquet "
                 "files and re-parses. The checkpoint still saves the archive scan.")
        ck.clear()
        nodes_path.unlink(missing_ok=True)
        screens_path.unlink(missing_ok=True)

    con.step(3, 4, "Parse hierarchies")
    con.note("One row per UI element. Expect roughly 6.4 million rows at full scale.")

    node_writer = pq.ParquetWriter(nodes_path, NODE_SCHEMA, compression="zstd")
    screen_writer = pq.ParquetWriter(screens_path, SCREEN_SCHEMA, compression="zstd")

    buffer: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    totals = {"screens": 0, "nodes": 0, "included": 0, "needs": 0, "labelled": 0,
              "icons": 0, "icons_labelled": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    t0 = time.time()
    progress_every = int(CFG.run.progress_every)

    try:
        for i, path in enumerate(files, start=1):
            screen_uid = path.stem
            try:
                blob = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                rows, summary = walk_screen(blob, screen_uid, ui_details.get(screen_uid, ""))
            except Exception as exc:
                totals["failed"] += 1
                failures.append({"screen_uid": screen_uid, "error": f"{type(exc).__name__}: {exc}"})
                continue

            buffer.extend(rows)
            screen_rows.append(summary)
            totals["screens"] += 1
            totals["nodes"] += summary["n_nodes"]
            totals["included"] += summary["n_included"]
            totals["needs"] += summary["n_needs_label"]
            totals["labelled"] += summary["n_labelled"]
            totals["icons"] += summary["n_icon_like"]
            totals["icons_labelled"] += summary["n_icon_like_labelled"]
            ck.mark(screen_uid)

            if len(buffer) >= BATCH_ROWS:
                node_writer.write_table(pa.Table.from_pylist(buffer, schema=NODE_SCHEMA))
                buffer.clear()
            if i % progress_every == 0 or i == len(files):
                con.progress(i, len(files), every=1,
                             label=f"{totals['nodes']:,} elements")
    finally:
        if buffer:
            node_writer.write_table(pa.Table.from_pylist(buffer, schema=NODE_SCHEMA))
        if screen_rows:
            screen_writer.write_table(pa.Table.from_pylist(screen_rows, schema=SCREEN_SCHEMA))
        node_writer.close()
        screen_writer.close()
        ck.close()

    con.step(4, 4, "Summary")
    elapsed = time.time() - t0
    con.kv("Screens parsed", totals["screens"])
    con.kv("Screens failed", totals["failed"])
    con.kv("Elements extracted", totals["nodes"])
    con.kv("Elements in analysis population", totals["included"])
    con.kv("Elements requiring a label", totals["needs"])
    con.kv("...of which labelled", totals["labelled"])
    if totals["needs"]:
        con.kv("Label coverage", f"{100 * totals['labelled'] / totals['needs']:.2f}%")
    con.kv("Icon-like elements", totals["icons"])
    if totals["icons"]:
        con.kv("Icon-like coverage",
               f"{100 * totals['icons_labelled'] / totals['icons']:.2f}%")
    con.kv("Elapsed", f"{elapsed:.1f}s")

    if totals["failed"]:
        con.warn(f"{totals['failed']} screen(s) failed to parse; details in the report.")

    report_path = PATHS.interim / "rico_parse_report.json"
    atomic_write_json(report_path, {
        "dataset": DATASET,
        "totals": totals,
        "elapsed_seconds": round(elapsed, 1),
        "limit": args.limit,
        "quarantined_app_rows": bad_apps,
        "failures": failures[:200],
        "inclusion_rule": CFG.analysis.inclusion.as_dict(),
    })

    con.done({
        "rico_nodes.parquet": nodes_path,
        "rico_screens.parquet": screens_path,
        "rico_apps.parquet": apps_path,
        "rico_parse_report.json": report_path,
    })
    con.info("Next:  python phase2_processing/02_parse_androidcontrol.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
