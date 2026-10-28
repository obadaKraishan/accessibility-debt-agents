#!/usr/bin/env python3
"""
PHASE 1 / STEP 2 — Download AndroidControl.

Fetches the AndroidControl TFRecord shards, plus splits.json and
test_subsplits.json, from the public Google Research bucket. Plain
anonymous HTTPS works; gsutil is not needed and neither is a Google
account, despite what the dataset README implies.

Full set: 20 shards, 2.35 to 2.68 GB each, 46.5 GiB total.
Pilot:    --shards 3 gives roughly 2,300 episodes and about 7 GB.

    python phase1_collection/02_download_androidcontrol.py
    python phase1_collection/02_download_androidcontrol.py --shards 3
    python phase1_collection/02_download_androidcontrol.py --splits-only

Licence: Apache-2.0. Unlike Rico, this one you may redistribute.

Outputs
-------
data/raw/androidcontrol/android_control-000NN-of-00020   (one per shard)
data/raw/androidcontrol/splits.json
data/raw/androidcontrol/test_subsplits.json
data/raw/androidcontrol/download_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console, human_bytes             # noqa: E402
from a11ydebt.download import download, load_manifest, save_manifest  # noqa: E402


def report_splits(con: Console, dest_dir: Path) -> dict:
    """
    Parse and sanity-check the split files.

    Two facts worth printing every run, because both are easy to get wrong:
    the paper reports 13,604 training episodes while splits.json contains
    13,603 (trust the file), and the four test subsplits OVERLAP, so their
    sizes must never be summed.
    """
    splits_path = dest_dir / "splits.json"
    subsplits_path = dest_dir / "test_subsplits.json"
    summary: dict = {}

    if splits_path.exists():
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        rows = [[name, f"{len(ids):,}"] for name, ids in splits.items()]
        total = sum(len(v) for v in splits.values())
        rows.append(["TOTAL", f"{total:,}"])
        con.table(["Split", "Episodes"], rows)
        summary["splits"] = {k: len(v) for k, v in splits.items()}
        summary["total_episodes"] = total

        expected = CFG.androidcontrol.expected_episodes.as_dict()
        if total == expected.get("total"):
            con.ok(f"Episode total matches the published figure ({total:,})")
        else:
            con.warn(f"Episode total is {total:,}; config expects "
                     f"{expected.get('total'):,}")
        con.note("The paper's table says train = 13,604 while splits.json holds "
                 "13,603. Trust the file and say so in the manuscript.")

    if subsplits_path.exists():
        subs = json.loads(subsplits_path.read_text(encoding="utf-8"))
        rows = [[name, f"{len(ids):,}"] for name, ids in subs.items()]
        con.table(["Test subsplit", "Episodes"], rows)
        summary["test_subsplits"] = {k: len(v) for k, v in subs.items()}
        sets = {k: set(v) for k, v in subs.items()}
        names = list(sets)
        overlaps = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = len(sets[a] & sets[b])
                if shared:
                    overlaps.append([a, b, f"{shared:,}"])
        if overlaps:
            con.warn("The test subsplits OVERLAP. Never sum metrics across them.")
            con.table(["Subsplit A", "Subsplit B", "Shared episodes"], overlaps)
            summary["subsplit_overlaps"] = overlaps
    return summary


def main() -> int:
    parser = base_parser(
        "Download the AndroidControl dataset (Li et al., NeurIPS 2024 D&B).",
        epilog="Resumable. Use --shards N for a pilot; each shard is about 2.5 GB.",
    )
    parser.add_argument("--shards", type=int, default=None,
                        help="how many shards to fetch (default: config.yaml "
                             "androidcontrol.n_shards_to_download)")
    parser.add_argument("--splits-only", action="store_true",
                        help="fetch only splits.json and test_subsplits.json (about 120 KB)")
    parser.add_argument("--no-checksum", action="store_true",
                        help="skip sha256 (faster, but the manifest is then less useful)")
    args = parser.parse_args()

    con = Console("02_download_androidcontrol", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 1 / STEP 2", "Download AndroidControl")

    dest_dir = PATHS.androidcontrol_raw
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "download_manifest.json"
    manifest = load_manifest(manifest_path)

    con.section("Licence")
    con.ok("Apache-2.0. Redistribution is permitted, unlike Rico.")

    aux = list(CFG.androidcontrol.aux_files)
    n_shards = 0 if args.splits_only else int(
        args.shards if args.shards is not None else CFG.androidcontrol.n_shards_to_download
    )
    n_shards = max(0, min(n_shards, int(CFG.androidcontrol.n_shards)))
    total_steps = len(aux) + n_shards + 1

    for i, name in enumerate(aux, start=1):
        con.step(i, total_steps, f"Download {name}")
        url = CFG.androidcontrol.aux_url(name)
        con.kv("URL", url)
        manifest[name] = download(url, dest_dir / name, con, checksum=True)
        save_manifest(manifest_path, manifest)

    if n_shards:
        con.section("Shards")
        con.note(f"Fetching {n_shards} of {CFG.androidcontrol.n_shards} shards "
                 f"(about {n_shards * 2.5:.0f} GB).")
        if n_shards < int(CFG.androidcontrol.n_shards):
            con.warn("This is a partial download. Any coverage figure computed "
                     "from it is a pilot estimate, not the census. Record the "
                     "shard count in the manuscript.")

    for j in range(n_shards):
        step = len(aux) + j + 1
        url = CFG.androidcontrol.shard_url(j)
        name = url.rsplit("/", 1)[-1]
        con.step(step, total_steps, f"Download shard {j + 1}/{n_shards}: {name}")
        try:
            manifest[name] = download(url, dest_dir / name, con,
                                      checksum=not args.no_checksum)
        except Exception as exc:
            save_manifest(manifest_path, manifest)
            con.fail(f"Shard {name} failed: {exc}",
                     hint="Re-run to resume. Reduce --shards if disk space is short.")
        save_manifest(manifest_path, manifest)

    con.step(total_steps, total_steps, "Inspect splits")
    summary = report_splits(con, dest_dir)
    manifest["splits_summary"] = summary
    save_manifest(manifest_path, manifest)

    con.section("Summary")
    shard_files = [p for p in sorted(dest_dir.glob("android_control-*-of-*"))
                   if p.suffix != ".part"]
    total_bytes = sum(p.stat().st_size for p in shard_files)
    con.kv("Complete shards on disk", len(shard_files))
    con.kv("Shard bytes", human_bytes(total_bytes))
    con.kv("Episodes in splits.json", summary.get("total_episodes", "n/a"))

    con.done({"download_manifest.json": manifest_path})
    con.info("Next:  python phase1_collection/03_download_rico_semantics.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
