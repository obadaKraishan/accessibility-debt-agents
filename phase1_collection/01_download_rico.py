#!/usr/bin/env python3
"""
PHASE 1 / STEP 1 — Download Rico.

Fetches the Rico packages this study needs from the University of Illinois
public bucket, resuming any partial download, verifying byte counts against
the sizes recorded in config.yaml, and computing a sha256 for each file so
the data availability statement can name a hash rather than just a URL.

The 6 GB screens archive is optionally extracted here too. Extraction is the
slow part (66,261 JSON files plus 66,261 JPEGs), so it is checkpointed and
safe to interrupt.

    python phase1_collection/01_download_rico.py
    python phase1_collection/01_download_rico.py --no-extract
    python phase1_collection/01_download_rico.py --metadata-only   # 4 MB, instant

LICENCE WARNING
---------------
Rico is distributed under a University of Illinois research-use agreement,
not an open licence, and its screenshots may contain third-party copyrighted
material. Do not redistribute the raw data, and do not commit it. See
docs/DATA_LICENSES.md. This script prints the terms URL before downloading.

Outputs
-------
data/raw/rico/ui_details.csv
data/raw/rico/app_details.csv
data/raw/rico/unique_uis.tar.gz          (6.03 GiB)
data/raw/rico/combined/<id>.json        (66,261 view hierarchies)
data/raw/rico/combined/<id>.jpg         (only with --with-screenshots)
data/raw/rico/download_manifest.json     sizes, checksums, timestamps
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a11ydebt.checkpoint import Checkpoint                    # noqa: E402
from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console, human_bytes             # noqa: E402
from a11ydebt.download import download, load_manifest, save_manifest  # noqa: E402

TERMS_URL = "https://www.interactionmining.org/archive/rico/copyright.txt"


def extract_archive(archive: Path, target: Path, con: Console, restart: bool,
                    json_only: bool = True) -> int:
    """
    Extract the view hierarchies (and optionally the screenshots) from
    unique_uis.tar.gz.

    By default only the .json hierarchies are written. NO analysis in this
    project reads a Rico screenshot: the parsers use bounds, and the
    RICO-Semantics validation rescales normalised boxes against the root
    bounds rather than against an image. Extracting the JPEGs as well would
    write 66,261 extra small files for nothing, which is slow on an external
    or network volume and costs several gigabytes. Pass --with-screenshots
    if you want them for figures or manual inspection.

    The archive is a single flat directory. Members arrive in random order,
    not sorted by id, so progress is reported against the member count rather
    than the id range. Any member whose basename starts with '._' is a macOS
    AppleDouble artifact and is skipped (they appear in traces.tar.gz; guarding
    here costs nothing and prevents a confusing double count).
    """
    target.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint("extract_rico", PATHS.checkpoints, restart=restart)
    if ck.resumed:
        con.info(f"Resuming extraction; {ck.n_done:,} members already written")

    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for i, member in enumerate(tar, start=1):
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name.startswith("._") or not name:
                continue
            if json_only and not name.endswith(".json"):
                continue
            if ck.is_done(name):
                continue
            out = target / name
            if out.exists() and out.stat().st_size == member.size:
                ck.mark(name)
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            out.write_bytes(src.read())
            ck.mark(name)
            written += 1
            if written % 2000 == 0:
                con.info(f"  extracted {written:,} new members "
                         f"({ck.n_done:,} total, {i:,} scanned)")
    ck.close()
    con.ok(f"Extraction complete: {ck.n_done:,} members in {target}")
    return ck.n_done


def main() -> int:
    parser = base_parser(
        "Download the Rico dataset (Deka et al., UIST 2017).",
        epilog="Resumable. Re-running skips anything already present and verified.",
    )
    parser.add_argument("--no-extract", action="store_true",
                        help="download the archive but do not extract it")
    parser.add_argument("--metadata-only", action="store_true",
                        help="fetch only ui_details.csv and app_details.csv (about 4 MB)")
    parser.add_argument("--with-semantics", action="store_true",
                        help="also fetch semantic_annotations.zip (150 MB), used in validation")
    parser.add_argument("--with-screenshots", action="store_true",
                        help="also extract the 66,261 JPEG screenshots. No analysis "
                             "reads them, so this is off by default: it writes twice "
                             "as many files and several extra GB for nothing.")
    args = parser.parse_args()

    con = Console("01_download_rico", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 1 / STEP 1", "Download Rico")

    con.section("Terms of use")
    con.warn("Rico is NOT openly licensed.")
    con.note("It is released under a University of Illinois research-use agreement.")
    con.note("Screenshots may contain third-party copyrighted material.")
    con.note("Do not redistribute the raw data and do not commit it to git.")
    con.note(f"Full terms: {TERMS_URL}")
    con.note("This repository's .gitignore already excludes data/.")

    dest_dir = PATHS.rico_raw
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "download_manifest.json"
    manifest = load_manifest(manifest_path)

    wanted = ["ui_details", "app_details"]
    if not args.metadata_only:
        wanted.append("unique_uis")
        if args.with_semantics:
            wanted.append("semantic_annotations")

    files_cfg = CFG.rico.files.as_dict()
    total_steps = len(wanted) + (0 if (args.no_extract or args.metadata_only) else 1)

    for i, key in enumerate(wanted, start=1):
        spec = files_cfg[key]
        name, expected = spec["name"], spec.get("bytes")
        con.step(i, total_steps, f"Download {name}")
        url = CFG.rico.download_url(name)
        con.kv("URL", url)
        try:
            manifest[key] = download(url, dest_dir / name, con, expected_bytes=expected)
        except Exception as exc:
            save_manifest(manifest_path, manifest)
            con.fail(
                f"Could not download {name}: {exc}",
                hint="Check your connection and re-run; the download resumes from "
                     "where it stopped. If the host is down, try again later.",
            )
        save_manifest(manifest_path, manifest)

    if not (args.no_extract or args.metadata_only):
        archive = dest_dir / files_cfg["unique_uis"]["name"]
        con.step(total_steps, total_steps, "Extract screens and view hierarchies")
        if args.with_screenshots:
            con.note("66,261 JSON hierarchies and 66,261 JPEG screenshots.")
            con.warn("Screenshots are not read by any analysis in this project. "
                     "You are writing about 6 GB and 66,261 extra files for "
                     "inspection only.")
        else:
            con.note("66,261 JSON view hierarchies. Screenshots are skipped because "
                     "no analysis reads them; pass --with-screenshots to include them.")
        con.note("This takes a while and is checkpointed; interrupting is safe.")
        n = extract_archive(archive, PATHS.rico_screens, con, restart=args.restart,
                            json_only=not args.with_screenshots)
        manifest["extraction"] = {"members": n, "target": str(PATHS.rico_screens)}
        save_manifest(manifest_path, manifest)

        expected = CFG.rico.expected_screens
        n_json = len(list(PATHS.rico_screens.glob("*.json")))
        con.kv("JSON hierarchies found", n_json)
        con.kv("Expected", expected)
        if n_json != expected:
            con.warn(f"Found {n_json:,} hierarchies, expected {expected:,}. "
                     "Re-run to resume extraction if it was interrupted.")
        else:
            con.ok("Screen count matches the published figure")

    con.section("Summary")
    total_bytes = sum(v.get("bytes", 0) for v in manifest.values() if isinstance(v, dict))
    con.kv("Files in manifest", len([k for k, v in manifest.items() if isinstance(v, dict)]))
    con.kv("Total downloaded", human_bytes(total_bytes))

    con.done({"download_manifest.json": manifest_path,
              "screens directory": PATHS.rico_screens})
    con.info("Next:  python phase1_collection/02_download_androidcontrol.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
