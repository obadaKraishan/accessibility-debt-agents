#!/usr/bin/env python3
"""
PHASE 1 / STEP 3 — Download RICO-Semantics.

RICO-Semantics is a set of human-drawn bounding boxes with icon-meaning
labels over 42,434 of Rico's 66,261 screens. The entire dataset lives in
the GitHub repository, so this is a single ~164 MB archive download.

WHAT IT IS FOR HERE
-------------------
RICO-Semantics does NOT contain content descriptions. It is not a second
source of developer-authored labels. It is used in phase 4 for one job:
given a Rico element that our codebook marks as unlabelled, RICO-Semantics
tells us whether a human annotator recognised a meaningful icon there. That
converts "no content description" into "no content description where a
human could name the function", which is a far stronger claim.

Coordinates in RICO-Semantics are normalised to [0, 1]; Rico bounds are
absolute pixels. The parser rescales using each screen's root bounds.

    python phase1_collection/03_download_rico_semantics.py

Licence: CC BY-SA 4.0 (note the ShareAlike obligation on derived data).

Outputs
-------
data/raw/rico_semantics/rico_semantics.zip
data/raw/rico_semantics/data/semantics/{train,val,test}.json
data/raw/rico_semantics/download_manifest.json
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console, human_bytes             # noqa: E402
from a11ydebt.download import download, load_manifest, save_manifest  # noqa: E402


def main() -> int:
    parser = base_parser(
        "Download RICO-Semantics (Sunkara et al., 2022).",
        epilog="Small and fast. Used only for codebook validation in phase 4.",
    )
    args = parser.parse_args()

    con = Console("03_download_rico_semantics", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 1 / STEP 3", "Download RICO-Semantics")

    dest_dir = PATHS.rico_semantics_raw
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "download_manifest.json"
    manifest = load_manifest(manifest_path)
    archive = dest_dir / "rico_semantics.zip"

    con.section("Licence")
    con.ok("CC BY-SA 4.0. Note the ShareAlike obligation on anything derived from it.")
    con.note("Rico's own terms still apply to the underlying screenshots.")

    con.step(1, 3, "Download the repository archive")
    url = CFG.rico_semantics.archive_url
    con.kv("URL", url)
    manifest["archive"] = download(url, archive, con)
    save_manifest(manifest_path, manifest)

    con.step(2, 3, "Extract the semantics split files")
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if "/data/semantics/" in member and member.endswith(".json"):
                target = dest_dir / "data" / "semantics" / Path(member).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                extracted.append(target)
                con.ok(f"{target.name} ({human_bytes(target.stat().st_size)})")
    if not extracted:
        con.fail("No semantics files found inside the archive.",
                 hint="The repository layout may have changed. Check "
                      f"{CFG.rico_semantics.repo_url} and update config.yaml.")

    con.step(3, 3, "Verify coverage")
    screens, elements, labels = set(), 0, set()
    for path in extracted:
        entries = json.loads(path.read_text(encoding="utf-8"))
        for entry in entries:
            screens.add(str(entry["screen_id"]))
            for el in entry.get("screen_elements", []):
                elements += 1
                labels.add(el.get("label", ""))
    con.kv("Screens covered", len(screens))
    con.kv("Annotated elements", elements)
    con.kv("Distinct labels", len(labels))
    expected = CFG.rico_semantics.expected_screens
    con.kv("Expected screens", expected)
    if len(screens) == expected:
        con.ok("Coverage matches the published figure")
    else:
        con.warn(f"Coverage is {len(screens):,}, config expects {expected:,}")
    con.note(f"That is {100 * len(screens) / CFG.rico.expected_screens:.1f}% of "
             f"Rico's {CFG.rico.expected_screens:,} screens. Validation in phase 4 "
             f"is restricted to this subset, which must be stated in the paper.")

    manifest["coverage"] = {
        "screens": len(screens), "elements": elements, "labels": len(labels),
    }
    save_manifest(manifest_path, manifest)

    con.done({"download_manifest.json": manifest_path,
              "semantics files": dest_dir / "data" / "semantics"})
    con.info("Next:  python phase2_processing/01_parse_rico.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
