#!/usr/bin/env python3
"""
PHASE 0 / STEP 1 — Verify the environment before anything downloads.

Run this first. It checks, and prints a verdict for, every dependency the
pipeline needs: Python version, packages, the AndroidControl protobuf
modules, free disk space, write permissions, and whether all three dataset
hosts are reachable. Nothing is downloaded and nothing is written except a
small JSON summary.

    python phase0_setup/01_verify_environment.py
    python phase0_setup/01_verify_environment.py --skip-network

Exit code is 0 when the project can run, 1 when something blocking is
missing. Warnings do not fail the run.

Outputs
-------
outputs/logs/01_verify_environment_<timestamp>.log
outputs/manuscript/environment_check.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a11ydebt.cli import base_parser                      # noqa: E402
from a11ydebt.console import Console, human_bytes         # noqa: E402

MIN_PYTHON = (3, 10)

REQUIRED_PACKAGES = [
    ("numpy", "array maths"),
    ("pandas", "tables"),
    ("scipy", "statistical tests"),
    ("yaml", "config.yaml parsing (package name: PyYAML)"),
    ("requests", "downloads"),
    ("pyarrow", "parquet interim tables"),
]

OPTIONAL_PACKAGES = [
    ("statsmodels", "logistic regression in RQ4"),
    ("matplotlib", "figures"),
    ("tqdm", "download bars"),
    ("tensorflow", "reading AndroidControl TFRecord shards"),
]

# Rico 6.03 GiB + AndroidControl 46.5 GiB + parsed tables, with headroom.
DISK_REQUIRED_GB = 80
DISK_PILOT_GB = 20

HOSTS = [
    ("Rico", "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv"),
    ("AndroidControl", "https://storage.googleapis.com/gresearch/android_control/splits.json"),
    ("RICO-Semantics", "https://github.com/google-research-datasets/rico_semantics"),
]


def check_python(con: Console, results: dict) -> bool:
    version = sys.version_info
    ok = version[:2] >= MIN_PYTHON
    con.kv("Python version", f"{version.major}.{version.minor}.{version.micro}")
    results["python"] = {"version": f"{version.major}.{version.minor}.{version.micro}", "ok": ok}
    if ok:
        con.ok(f"Python {version.major}.{version.minor} meets the minimum of "
               f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    else:
        con.error(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
                  f"(android-env 1.3.0 will not install below it)")
    return ok


def check_packages(con: Console, results: dict) -> bool:
    rows, blocking = [], False
    for name, purpose in REQUIRED_PACKAGES:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            rows.append([name, version, "required", "ok", purpose])
            results["packages"][name] = {"version": version, "ok": True, "required": True}
        except ImportError:
            rows.append([name, "-", "required", "MISSING", purpose])
            results["packages"][name] = {"version": None, "ok": False, "required": True}
            blocking = True

    for name, purpose in OPTIONAL_PACKAGES:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            rows.append([name, version, "optional", "ok", purpose])
            results["packages"][name] = {"version": version, "ok": True, "required": False}
        except ImportError:
            rows.append([name, "-", "optional", "missing", purpose])
            results["packages"][name] = {"version": None, "ok": False, "required": False}

    con.table(["Package", "Version", "Kind", "Status", "Used for"], rows)
    return not blocking


def check_androidcontrol_protos(con: Console, results: dict) -> bool:
    """
    The dataset README tells you to clone google-deepmind/android_env and
    pip install it. That is unnecessary: the PyPI wheel android-env==1.3.0
    already ships the pre-generated *_pb2 modules.
    """
    try:
        from android_env.proto.a11y import android_accessibility_forest_pb2 as forest_pb2
        message = forest_pb2.AndroidAccessibilityForest()
        _ = message.SerializeToString()
        con.ok("android_env accessibility protos import and round-trip cleanly")
        results["android_env_protos"] = {"ok": True}
        return True
    except ImportError as exc:
        con.warn(f"android_env protos unavailable ({exc.__class__.__name__}: {exc})")
        con.note("Fix with:  pip install android-env==1.3.0")
        con.note("You can still run every Rico step without this.")
        results["android_env_protos"] = {"ok": False, "error": str(exc)}
        return False
    except Exception as exc:  # protobuf runtime mismatch
        con.warn(f"android_env protos present but not usable: {exc}")
        con.note("Usually a protobuf version clash. Try:  pip install -U protobuf")
        results["android_env_protos"] = {"ok": False, "error": str(exc)}
        return False


def check_disk(con: Console, results: dict, data_dir: Path) -> bool:
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_dir)
    free_gb = usage.free / (1024 ** 3)
    con.kv("Data directory", str(data_dir))
    con.kv("Free space", f"{free_gb:,.1f} GB")
    results["disk"] = {"path": str(data_dir), "free_gb": round(free_gb, 1)}

    if free_gb >= DISK_REQUIRED_GB:
        con.ok(f"Enough room for the full download ({DISK_REQUIRED_GB} GB needed)")
        return True
    if free_gb >= DISK_PILOT_GB:
        con.warn(f"Only {free_gb:,.1f} GB free. Enough for Rico plus a few "
                 f"AndroidControl shards, not the full 46.5 GB.")
        con.note("Reduce androidcontrol.n_shards_to_download in config.yaml, "
                 "or set A11YDEBT_DATA_DIR in .env to a bigger disk.")
        return True
    con.error(f"Only {free_gb:,.1f} GB free; at least {DISK_PILOT_GB} GB is needed "
              f"even for a pilot run.")
    return False


def check_writable(con: Console, results: dict, paths: list[Path]) -> bool:
    ok = True
    for path in paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            con.ok(f"writable: {path}")
        except OSError as exc:
            con.error(f"not writable: {path} ({exc})")
            ok = False
    results["writable"] = ok
    return ok


def check_network(con: Console, results: dict) -> bool:
    try:
        import requests
    except ImportError:
        con.warn("requests is not installed; skipping reachability checks")
        return True

    rows, all_ok = [], True
    for name, url in HOSTS:
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            size = resp.headers.get("Content-Length")
            status = "reachable" if resp.status_code < 400 else f"HTTP {resp.status_code}"
            ok = resp.status_code < 400
            rows.append([name, status, human_bytes(int(size)) if size else "-"])
            results["hosts"][name] = {"url": url, "status": resp.status_code, "ok": ok}
            all_ok &= ok
        except Exception as exc:
            rows.append([name, f"FAILED ({exc.__class__.__name__})", "-"])
            results["hosts"][name] = {"url": url, "status": None, "ok": False,
                                      "error": str(exc)}
            all_ok = False
    con.table(["Host", "Status", "Probe size"], rows)
    return all_ok


def main() -> int:
    parser = base_parser(
        "Verify that this machine can run the a11ydebt pipeline.",
        epilog="Run this before phase 1. It downloads nothing.",
    )
    parser.add_argument("--skip-network", action="store_true",
                        help="do not probe the dataset hosts")
    args = parser.parse_args()

    # config import is deferred so a missing PyYAML produces a clear message
    # here rather than a traceback at import time.
    try:
        from a11ydebt.config import PATHS, CFG
    except Exception as exc:
        print(f"\n  Could not load project config: {exc}")
        print("  Fix with:  pip install -r requirements.txt\n")
        return 1

    con = Console("01_verify_environment", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 0 / STEP 1", "Verify the environment before anything downloads")

    results: dict = {"packages": {}, "hosts": {}}
    blocking: list[str] = []

    con.step(1, 6, "Python interpreter")
    if not check_python(con, results):
        blocking.append("Python version")

    con.step(2, 6, "Required and optional packages")
    if not check_packages(con, results):
        blocking.append("required packages")

    con.step(3, 6, "AndroidControl protobuf modules")
    check_androidcontrol_protos(con, results)  # warning only

    con.step(4, 6, "Disk space")
    if not check_disk(con, results, PATHS.data):
        blocking.append("disk space")

    con.step(5, 6, "Write permissions")
    if not check_writable(con, results, [PATHS.raw, PATHS.interim, PATHS.processed,
                                         PATHS.logs, PATHS.checkpoints, PATHS.manuscript]):
        blocking.append("write permissions")

    con.step(6, 6, "Dataset host reachability")
    if args.skip_network:
        con.note("skipped (--skip-network)")
    elif not check_network(con, results):
        con.warn("At least one host was unreachable. Downloads may fail.")

    con.section("Configuration in force")
    con.kv("Project", CFG.as_dict().get("project", {}).get("name", "?"))
    con.kv("Random seed", CFG.run.seed, raw=True)
    con.kv("Confidence level", CFG.run.confidence)
    con.kv("Permutations", CFG.run.n_permutations)
    con.kv("AndroidControl shards to download", CFG.androidcontrol.n_shards_to_download)

    out_path = PATHS.manuscript / "environment_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results["blocking"] = blocking
    results["verdict"] = "ready" if not blocking else "blocked"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    if blocking:
        con.fail(
            "Blocking problems: " + ", ".join(blocking),
            hint="Install dependencies with:  pip install -r requirements.txt\n"
                 "             Then re-run:  python phase0_setup/01_verify_environment.py",
        )
    con.done({"environment_check.json": out_path})
    con.info("Next:  python phase0_setup/02_init_project.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
