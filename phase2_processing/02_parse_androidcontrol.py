#!/usr/bin/env python3
"""
PHASE 2 / STEP 2 — Parse AndroidControl accessibility trees and trajectories.

Reads the TFRecord shards, decodes every AndroidAccessibilityForest, applies
the same codebook used for Rico, and resolves each click action to the
element it hit. Produces three tables: elements, steps, and episodes.

    python phase2_processing/02_parse_androidcontrol.py
    python phase2_processing/02_parse_androidcontrol.py --limit 500     # pilot
    python phase2_processing/02_parse_androidcontrol.py --verify-tf     # cross-check

Roughly 23 million elements at full scale. Checkpointed per episode.

WHY THE CLICK RESOLUTION MATTERS
--------------------------------
AndroidControl actions carry only coordinates. The complete key set across
every action type is {action_type, x, y, text, direction, app_name}: there is
no element, node, unique_id, target, bounds or index field. So "which element
did the trajectory touch", the question RQ3 turns on, is not in the data and
must be inferred by point-in-bounds hit testing.

The resolution rule lives in src/a11ydebt/tfrecord.py:resolve_click. On the
parsed episodes, 99.92% of clicks and long presses resolved to an element. Of
those, 43.71% were reassigned to the nearest clickable ancestor and 18.42%
stayed on a node that is not clickable, because none of its ancestors was
(outputs/manuscript/manuscript_numbers.txt, ReviewerChecks section). This script records the hit rate every run, so a
drop is visible immediately rather than silently biasing RQ3.

DECODING TRAPS THIS PARSER HANDLES
----------------------------------
* unique_id restarts per window, so nodes are keyed on (window_index,
  unique_id) rather than unique_id alone.
* List position is not unique_id; observed node order includes 0,1,2,3,4,5,7,6.
  child_ids are resolved through an explicit id map.
* text_selection fields are -1 when unset and read as a huge unsigned value
  if treated as unsigned. This parser does not use them.
* len(actions) == len(screenshots) - 1, so the last observation has no action.

Outputs
-------
data/processed/androidcontrol_nodes.parquet      one row per element per step
data/processed/androidcontrol_steps.parquet      one row per action
data/processed/androidcontrol_episodes.parquet   one row per episode
data/interim/androidcontrol_parse_report.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

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
from a11ydebt.tfrecord import (                               # noqa: E402
    TruncatedShardError, iter_records, parse_example, parse_forest, resolve_click,
)

BATCH_ROWS = 250_000
DATASET = "androidcontrol_2024"

#: A real shard is exactly "android_control-NNNNN-of-NNNNN" and nothing more.
SHARD_NAME = re.compile(r"android_control-\d{5}-of-\d{5}")

NODE_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("episode_id", pa.int64()), ("step_index", pa.int32()),
    ("screen_uid", pa.string()), ("window_index", pa.int32()), ("window_layer", pa.int32()),
    ("node_uid", pa.int32()), ("depth", pa.int32()),
    ("class_name", pa.string()), ("content_desc", pa.string()), ("text", pa.string()),
    ("resource_id", pa.string()), ("hint_text", pa.string()),
    ("clickable", pa.bool_()), ("long_clickable", pa.bool_()),
    ("enabled", pa.bool_()), ("focusable", pa.bool_()),
    ("visible_to_user", pa.bool_()), ("editable", pa.bool_()),
    ("x1", pa.int32()), ("y1", pa.int32()), ("x2", pa.int32()), ("y2", pa.int32()),
    ("screen_w", pa.int32()), ("screen_h", pa.int32()),
    ("role", pa.string()), ("label_status", pa.string()), ("needs_label", pa.bool_()),
    ("icon_like", pa.bool_()), ("is_fab", pa.bool_()), ("included", pa.bool_()),
    ("labeled_by_id", pa.int32()),
    ("was_touched", pa.bool_()),
])

STEP_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("episode_id", pa.int64()), ("step_index", pa.int32()),
    ("action_type", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
    ("direction", pa.string()), ("app_name", pa.string()),
    ("has_text", pa.bool_()), ("step_instruction", pa.string()),
    ("resolved", pa.bool_()), ("resolved_window", pa.int32()),
    ("resolved_node_uid", pa.int32()), ("resolved_role", pa.string()),
    ("resolved_label_status", pa.string()), ("resolved_clickable", pa.bool_()),
    ("used_ancestor_fallback", pa.bool_()),
])

EPISODE_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("episode_id", pa.int64()), ("split", pa.string()),
    ("goal", pa.string()), ("n_observations", pa.int32()), ("n_actions", pa.int32()),
    ("n_elements", pa.int32()), ("n_included", pa.int32()),
    ("n_needs_label", pa.int32()), ("n_labelled", pa.int32()),
    ("n_icon_like", pa.int32()), ("n_icon_like_labelled", pa.int32()),
    ("n_clicks", pa.int32()), ("n_clicks_resolved", pa.int32()),
    ("shard", pa.string()),
])


class _ShardIterator:
    """Wraps the record generator so a truncated shard warns instead of raising.

    A shard that ends mid-record yields everything it could read and sets
    `.truncated`, which the caller reports. Silently analysing a partial
    corpus would be worse than either alternative.
    """

    def __init__(self, generator) -> None:
        self._generator = generator
        self.truncated = False

    def __iter__(self):
        while True:
            try:
                yield next(self._generator)
            except StopIteration:
                return
            except TruncatedShardError:
                self.truncated = True
                return


def _iterate_shard(generator) -> _ShardIterator:
    return _ShardIterator(generator)


def load_splits(directory: Path, con: Console) -> dict[int, str]:
    path = directory / "splits.json"
    mapping: dict[int, str] = {}
    if path.exists():
        for split, ids in json.loads(path.read_text(encoding="utf-8")).items():
            for episode_id in ids:
                mapping[int(episode_id)] = split
        con.kv("Episodes with a split label", len(mapping))
    else:
        con.warn("splits.json not found; every episode will be labelled 'unknown'")
    return mapping


def parse_episode(
    example: dict[str, list],
    split: str,
    shard_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Decode one episode into element rows, step rows and an episode summary."""
    episode_id = int(example.get("episode_id", [0])[0])
    goal = (example.get("goal") or [b""])[0].decode("utf-8", "replace")
    trees = example.get("accessibility_trees") or []
    widths = example.get("screenshot_widths") or []
    heights = example.get("screenshot_heights") or []
    actions_raw = example.get("actions") or []
    instructions = example.get("step_instructions") or []

    node_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    touched: set[tuple[int, int, int]] = set()   # (step, window_index, unique_id)
    n_clicks = n_resolved = 0

    forests = []
    for step_index, blob in enumerate(trees):
        try:
            forests.append(parse_forest(blob))
        except Exception:
            forests.append([])

    # --- resolve actions first, so elements can be flagged as touched
    for step_index, blob in enumerate(actions_raw):
        try:
            action = json.loads(blob.decode("utf-8", "replace"))
        except Exception:
            action = {"action_type": "unparsable"}
        action_type = str(action.get("action_type", ""))
        x = action.get("x")
        y = action.get("y")
        row = {
            "dataset": DATASET, "episode_id": episode_id, "step_index": step_index,
            "action_type": action_type,
            "x": float(x) if x is not None else None,
            "y": float(y) if y is not None else None,
            "direction": str(action.get("direction") or ""),
            "app_name": str(action.get("app_name") or ""),
            "has_text": bool(action.get("text")),
            "step_instruction": (
                instructions[step_index].decode("utf-8", "replace")
                if step_index < len(instructions) else ""
            ),
            "resolved": False, "resolved_window": -1, "resolved_node_uid": -1,
            "resolved_role": "", "resolved_label_status": "",
            "resolved_clickable": False, "used_ancestor_fallback": False,
        }

        if action_type in {"click", "long_press"} and x is not None and y is not None:
            n_clicks += 1
            windows = forests[step_index] if step_index < len(forests) else []
            node, window = resolve_click(windows, float(x), float(y))
            if node is not None and window is not None:
                n_resolved += 1
                window_index = next(
                    (i for i, w in enumerate(windows) if w is window), -1
                )
                role = classify_role(node.class_name)
                direct, _ = resolve_click(windows, float(x), float(y),
                                          fallback_to_clickable_ancestor=False)
                used_fallback = direct is not node
                row.update({
                    "resolved": True,
                    "resolved_window": window_index,
                    "resolved_node_uid": node.unique_id,
                    "resolved_role": role,
                    "resolved_label_status": label_status(
                        role, node.content_description, node.text,
                        node.is_clickable, node.is_long_clickable),
                    "resolved_clickable": node.is_clickable,
                    # Was `not node.is_clickable`, evaluated on the node
                    # resolve_click returned, i.e. after the fallback had run,
                    # so it marked clicks that ended on a non-clickable node and
                    # never a successful fallback. The parquet files in
                    # data/processed predate this fix and still hold the old
                    # meaning. phase4_validation/03_reviewer_checks.py
                    # reconstructs the flag from the stored nodes and is the
                    # source of record for the fallback share.
                    "used_ancestor_fallback": used_fallback,
                })
                touched.add((step_index, window_index, node.unique_id))
        step_rows.append(row)

    # --- element rows
    for step_index, windows in enumerate(forests):
        sw = int(widths[step_index]) if step_index < len(widths) else 0
        sh = int(heights[step_index]) if step_index < len(heights) else 0
        for window_index, window in enumerate(windows):
            for node in window.nodes:
                role = classify_role(node.class_name)
                status = label_status(
                    role, node.content_description, node.text,
                    node.is_clickable, node.is_long_clickable)
                node_rows.append({
                    "dataset": DATASET, "episode_id": episode_id,
                    "step_index": step_index,
                    "screen_uid": f"{episode_id}_{step_index}",
                    "window_index": window_index, "window_layer": window.layer,
                    "node_uid": node.unique_id, "depth": node.depth,
                    "class_name": node.class_name,
                    "content_desc": node.content_description,
                    "text": node.text,
                    "resource_id": node.view_id_resource_name,
                    "hint_text": node.hint_text,
                    "clickable": node.is_clickable,
                    "long_clickable": node.is_long_clickable,
                    "enabled": node.is_enabled, "focusable": node.is_focusable,
                    "visible_to_user": node.is_visible_to_user,
                    "editable": node.is_editable,
                    "x1": node.left, "y1": node.top, "x2": node.right, "y2": node.bottom,
                    "screen_w": sw, "screen_h": sh,
                    "role": role, "label_status": status,
                    "needs_label": needs_label(role, node.is_clickable,
                                               node.is_long_clickable, node.text),
                    "icon_like": is_icon_like(role, node.text, node.is_clickable,
                                              node.is_long_clickable),
                    "is_fab": is_floating_action_button(node.class_name),
                    "included": in_analysis_population(
                        node.is_visible_to_user, node.left, node.top,
                        node.right, node.bottom, node.is_enabled),
                    "labeled_by_id": node.labeled_by_id,
                    "was_touched": (step_index, window_index, node.unique_id) in touched,
                })

    included = [r for r in node_rows if r["included"]]
    needs = [r for r in included if r["needs_label"]]
    icons = [r for r in included if r["icon_like"]]
    summary = {
        "dataset": DATASET, "episode_id": episode_id, "split": split, "goal": goal,
        "n_observations": len(trees), "n_actions": len(actions_raw),
        "n_elements": len(node_rows), "n_included": len(included),
        "n_needs_label": len(needs),
        "n_labelled": sum(1 for r in needs if r["label_status"] == "labelled"),
        "n_icon_like": len(icons),
        "n_icon_like_labelled": sum(1 for r in icons if r["label_status"] == "labelled"),
        "n_clicks": n_clicks, "n_clicks_resolved": n_resolved,
        "shard": shard_name,
    }
    return node_rows, step_rows, summary


def main() -> int:
    parser = base_parser(
        "Parse AndroidControl accessibility trees and resolve trajectory clicks.",
        epilog="Checkpointed per episode. Uses a built-in TFRecord reader; "
               "TensorFlow is not required.",
    )
    parser.add_argument("--verify-tf", action="store_true",
                        help="cross-check the built-in reader against TensorFlow "
                             "and android_env on the first 20 episodes")
    args = parser.parse_args()

    con = Console("02_parse_androidcontrol", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 2 / STEP 2", "Parse AndroidControl accessibility trees")

    raw_dir = PATHS.androidcontrol_raw
    # Match the shard filename EXACTLY. A glob like "android_control-*-of-*"
    # also matches anything sitting beside a shard: a ".part" from a download
    # in progress, and a ".sha256" checksum record. Reading either would count
    # a non-shard as a real one. Excluding suffixes one at a time is whack-a-
    # mole, so the name is matched against the exact pattern instead, and
    # everything else is reported rather than silently skipped.
    all_matches = sorted(raw_dir.glob("android_control-*-of-*"))
    shards = [p for p in all_matches if SHARD_NAME.fullmatch(p.name)]
    in_progress = [p for p in all_matches if p.name.endswith(".part")]
    other = [p for p in all_matches
             if p not in shards and p not in in_progress]
    if not shards:
        con.fail(
            f"No AndroidControl shards found in {raw_dir}",
            hint="Run:  python phase1_collection/02_download_androidcontrol.py",
        )

    con.step(1, 4, "Inputs")
    con.kv("Complete shards on disk", len(shards))
    con.kv("Of the full set", CFG.androidcontrol.n_shards)
    if in_progress:
        con.warn(f"{len(in_progress)} shard(s) still downloading; skipping them: "
                 + ", ".join(p.name for p in in_progress))
        con.note("Parsing what is finished is safe. Re-run this script after the "
                 "download completes to include the rest.")
    if len(shards) < int(CFG.androidcontrol.n_shards):
        con.warn("Partial shard set. Coverage figures will be pilot estimates; "
                 "record the shard count in the manuscript.")
    splits = load_splits(raw_dir, con)

    if args.verify_tf:
        con.step(2, 4, "Cross-check the built-in reader against TensorFlow")
        try:
            import tensorflow as tf
            from android_env.proto.a11y import android_accessibility_forest_pb2 as fpb
            raw = tf.data.TFRecordDataset([str(shards[0])], compression_type="GZIP")
            mismatches = 0
            for i, record in enumerate(raw.take(20)):
                blob = record.numpy()
                ours = parse_example(blob)
                theirs = tf.train.Example.FromString(blob)
                tf_trees = list(theirs.features.feature["accessibility_trees"].bytes_list.value)
                if len(tf_trees) != len(ours.get("accessibility_trees", [])):
                    mismatches += 1
                    continue
                if tf_trees:
                    ref = fpb.AndroidAccessibilityForest().FromString(tf_trees[0])
                    ours_windows = parse_forest(tf_trees[0])
                    if len(ref.windows) != len(ours_windows):
                        mismatches += 1
            if mismatches:
                con.warn(f"{mismatches}/20 episodes disagreed between readers.")
            else:
                con.ok("Built-in reader agrees with TensorFlow and android_env on 20 episodes")
        except ImportError as exc:
            con.warn(f"Cross-check skipped: {exc}")
    else:
        con.step(2, 4, "Reader")
        con.ok("Using the built-in TFRecord and protobuf reader (no TensorFlow needed)")
        con.note("Add --verify-tf to cross-check it against TensorFlow when available.")

    ck = Checkpoint("parse_androidcontrol", PATHS.checkpoints,
                    flush_every=50, restart=True)  # parquet cannot be appended across runs
    nodes_path = PATHS.processed / "androidcontrol_nodes.parquet"
    steps_path = PATHS.processed / "androidcontrol_steps.parquet"
    episodes_path = PATHS.processed / "androidcontrol_episodes.parquet"

    con.step(3, 4, "Decode episodes")
    node_writer = pq.ParquetWriter(nodes_path, NODE_SCHEMA, compression="zstd")
    step_writer = pq.ParquetWriter(steps_path, STEP_SCHEMA, compression="zstd")
    episode_writer = pq.ParquetWriter(episodes_path, EPISODE_SCHEMA, compression="zstd")

    node_buf: list[dict[str, Any]] = []
    step_buf: list[dict[str, Any]] = []
    episode_buf: list[dict[str, Any]] = []
    totals = {"episodes": 0, "elements": 0, "included": 0, "needs": 0, "labelled": 0,
              "icons": 0, "icons_labelled": 0, "clicks": 0, "clicks_resolved": 0,
              "failed": 0}
    failures: list[dict[str, str]] = []
    truncated_shards: list[str] = []
    t0 = time.time()

    try:
        for shard in shards:
            con.info(f"shard {shard.name}")
            shard_start = totals["episodes"]
            try:
                shard_records = iter_records(shard, CFG.androidcontrol.schema.compression)
                blobs = _iterate_shard(shard_records)
            except TruncatedShardError as exc:
                con.warn(str(exc))
                continue
            for blob in blobs:
                if args.limit and totals["episodes"] >= args.limit:
                    break
                try:
                    example = parse_example(blob)
                    episode_id = int(example.get("episode_id", [0])[0])
                    nodes, steps, summary = parse_episode(
                        example, splits.get(episode_id, "unknown"), shard.name)
                except Exception as exc:
                    totals["failed"] += 1
                    failures.append({"shard": shard.name,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    continue

                node_buf.extend(nodes)
                step_buf.extend(steps)
                episode_buf.append(summary)
                totals["episodes"] += 1
                totals["elements"] += summary["n_elements"]
                totals["included"] += summary["n_included"]
                totals["needs"] += summary["n_needs_label"]
                totals["labelled"] += summary["n_labelled"]
                totals["icons"] += summary["n_icon_like"]
                totals["icons_labelled"] += summary["n_icon_like_labelled"]
                totals["clicks"] += summary["n_clicks"]
                totals["clicks_resolved"] += summary["n_clicks_resolved"]
                ck.mark(summary["episode_id"])

                if len(node_buf) >= BATCH_ROWS:
                    node_writer.write_table(pa.Table.from_pylist(node_buf, schema=NODE_SCHEMA))
                    node_buf.clear()
                if totals["episodes"] % 100 == 0:
                    con.info(f"  {totals['episodes']:,} episodes, "
                             f"{totals['elements']:,} elements, "
                             f"{totals['clicks_resolved']:,}/{totals['clicks']:,} clicks resolved")
            if blobs.truncated:
                con.warn(f"{shard.name} ended mid-record; recovered "
                         f"{totals['episodes'] - shard_start:,} episodes from it.")
                con.note("This is normally an interrupted download. Re-run "
                         "phase1_collection/02_download_androidcontrol.py to resume, "
                         "then re-run this parser.")
                truncated_shards.append(shard.name)
            if args.limit and totals["episodes"] >= args.limit:
                break
    finally:
        if node_buf:
            node_writer.write_table(pa.Table.from_pylist(node_buf, schema=NODE_SCHEMA))
        if step_buf:
            step_writer.write_table(pa.Table.from_pylist(step_buf, schema=STEP_SCHEMA))
        if episode_buf:
            episode_writer.write_table(pa.Table.from_pylist(episode_buf, schema=EPISODE_SCHEMA))
        node_writer.close()
        step_writer.close()
        episode_writer.close()
        ck.close()

    con.step(4, 4, "Summary")
    elapsed = time.time() - t0
    con.kv("Episodes parsed", totals["episodes"])
    con.kv("Episodes failed", totals["failed"])
    con.kv("Elements extracted", totals["elements"])
    con.kv("Elements in analysis population", totals["included"])
    con.kv("Elements requiring a label", totals["needs"])
    con.kv("...of which labelled", totals["labelled"])
    if totals["needs"]:
        con.kv("Label coverage", f"{100 * totals['labelled'] / totals['needs']:.2f}%")
    con.kv("Click actions", totals["clicks"])
    hit_rate = 100 * totals["clicks_resolved"] / totals["clicks"] if totals["clicks"] else 0.0
    con.kv("Clicks resolved to an element", f"{totals['clicks_resolved']:,} ({hit_rate:.2f}%)")
    if totals["clicks"] and hit_rate < 95:
        con.warn("Hit rate below 95%. Validation runs found ~100% of clicks landing "
                 "inside a visible node, so check the resolution rule before RQ3.")
    else:
        con.ok("Hit rate is in the expected range for RQ3")
    con.kv("Elapsed", f"{elapsed:.1f}s")

    report_path = PATHS.interim / "androidcontrol_parse_report.json"
    atomic_write_json(report_path, {
        "dataset": DATASET, "totals": totals, "hit_rate_pct": round(hit_rate, 2),
        "shards_used": [s.name for s in shards],
        "shards_skipped_incomplete": [s.name for s in in_progress],
        "non_shard_files_ignored": [s.name for s in other],
        "elapsed_seconds": round(elapsed, 1), "limit": args.limit,
        "failures": failures[:200],
        "truncated_shards": truncated_shards,
        "hit_test_rule": CFG.analysis.hit_test.as_dict(),
    })

    con.done({
        "androidcontrol_nodes.parquet": nodes_path,
        "androidcontrol_steps.parquet": steps_path,
        "androidcontrol_episodes.parquet": episodes_path,
        "androidcontrol_parse_report.json": report_path,
    })
    con.info("Next:  python phase2_processing/03_build_analysis_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
