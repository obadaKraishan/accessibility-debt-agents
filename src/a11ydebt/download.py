"""
Resumable HTTP downloads with integrity checking.

Both dataset hosts (storage.googleapis.com for Rico and AndroidControl)
support HTTP Range requests, so an interrupted 6 GB or 46 GB download picks
up where it stopped instead of starting again. Every file is checksummed
after download and the checksum is recorded, because the manuscript's data
availability statement should name a hash, not just a URL.

    from a11ydebt.download import download
    info = download(url, dest, con, expected_bytes=6471262799)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from .console import Console, human_bytes, human_time

__all__ = ["download", "sha256_file", "remote_size", "load_manifest",
           "save_manifest", "cached_sha256", "record_sha256"]

CHUNK = 1024 * 1024  # 1 MiB
TIMEOUT = (15, 120)  # connect, read


def remote_size(url: str) -> int | None:
    try:
        resp = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        size = resp.headers.get("Content-Length")
        return int(size) if size else None
    except Exception:
        return None


def _sidecar(path: Path) -> Path:
    """Where the recorded checksum for a downloaded file lives."""
    return path.with_name(path.name + ".sha256")


def cached_sha256(path: Path, size: int) -> str | None:
    """
    Return a previously computed checksum for `path`, if one is on record and
    still applies.

    Hashing is expensive: a 6 GB archive takes about eight minutes and a 2.5 GB
    shard about two. Re-hashing every completed file on every resume would cost
    half an hour before a single new byte was fetched, which is exactly the
    situation a resumable downloader exists to avoid. The checksum is written
    beside the file once, and reused as long as the byte count still matches.
    The recorded size is part of the sidecar, so a file that changed on disk is
    re-hashed rather than trusted.
    """
    side = _sidecar(path)
    if not side.exists():
        return None
    try:
        blob = json.loads(side.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if int(blob.get("bytes", -1)) != int(size):
        return None
    digest = str(blob.get("sha256", ""))
    return digest if len(digest) == 64 else None


def record_sha256(path: Path, digest: str, size: int) -> None:
    try:
        _sidecar(path).write_text(
            json.dumps({"sha256": digest, "bytes": int(size),
                        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}),
            encoding="utf-8",
        )
    except OSError:
        pass          # a missing sidecar only costs time, never correctness


def sha256_file(path: str | Path, con: Console | None = None) -> str:
    """Checksum a large file without loading it into memory."""
    path = Path(path)
    total = path.stat().st_size
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK * 8):
            digest.update(chunk)
            read += len(chunk)
            if con and total > 200 * 1024 * 1024:
                con.progress(read, total, every=1, label="sha256")
    return digest.hexdigest()


def download(
    url: str,
    dest: str | Path,
    con: Console,
    expected_bytes: int | None = None,
    checksum: bool = True,
    max_retries: int = 5,
) -> dict[str, Any]:
    """
    Download `url` to `dest`, resuming a partial file if one is present.

    Returns a dict describing the result, suitable for a download manifest.
    Raises on unrecoverable failure so the caller can decide what to do.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    size = expected_bytes or remote_size(url)
    if size:
        con.kv("Remote size", human_bytes(size))

    # Already finished?
    if dest.exists() and (size is None or dest.stat().st_size == size):
        con.ok(f"Already downloaded: {dest.name} ({human_bytes(dest.stat().st_size)})")
        result: dict[str, Any] = {
            "url": url, "path": str(dest), "bytes": dest.stat().st_size,
            "status": "cached", "downloaded_at": None,
        }
        if checksum:
            known = cached_sha256(dest, dest.stat().st_size)
            if known:
                con.note(f"sha256 already on record: {known[:32]}...")
                result["sha256"] = known
                result["checksum_source"] = "recorded"
            else:
                con.info("Computing sha256 (once; recorded for future runs) ...")
                digest = sha256_file(dest, con)
                record_sha256(dest, digest, dest.stat().st_size)
                result["sha256"] = digest
                result["checksum_source"] = "computed"
        return result

    start_at = part.stat().st_size if part.exists() else 0
    if start_at:
        con.info(f"Resuming from {human_bytes(start_at)}")

    attempt = 0
    t0 = time.time()
    while attempt < max_retries:
        attempt += 1
        headers = {"Range": f"bytes={start_at}-"} if start_at else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
                if start_at and resp.status_code == 200:
                    # Server ignored the Range header; start over.
                    con.warn("Server ignored the range request; restarting the file")
                    start_at = 0
                    part.unlink(missing_ok=True)
                elif start_at and resp.status_code != 206:
                    resp.raise_for_status()
                else:
                    resp.raise_for_status()

                total = size or (int(resp.headers.get("Content-Length", 0)) + start_at)
                mode = "ab" if start_at else "wb"
                written = start_at
                with part.open(mode) as fh:
                    for chunk in resp.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        if total:
                            con.progress(written, total, every=1,
                                         label=f"{human_bytes(written)}")
            break

        except (requests.RequestException, OSError) as exc:
            start_at = part.stat().st_size if part.exists() else 0
            wait = min(60, 2 ** attempt)
            con.warn(f"Attempt {attempt}/{max_retries} failed ({exc.__class__.__name__}: "
                     f"{exc}). Retrying in {wait}s from {human_bytes(start_at)}.")
            if attempt >= max_retries:
                raise
            time.sleep(wait)

    final_size = part.stat().st_size
    if size and final_size != size:
        raise IOError(
            f"{dest.name}: expected {size:,} bytes but got {final_size:,}. "
            f"The partial file was kept at {part}; re-run to resume."
        )
    part.replace(dest)
    elapsed = time.time() - t0
    rate = (final_size - 0) / elapsed if elapsed > 0 else 0
    con.ok(f"Downloaded {dest.name} — {human_bytes(final_size)} in "
           f"{human_time(elapsed)} ({human_bytes(rate)}/s)")

    result = {
        "url": url, "path": str(dest), "bytes": final_size,
        "status": "downloaded", "seconds": round(elapsed, 1),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if checksum:
        con.info("Computing sha256 (recorded for the data availability statement) ...")
        digest = sha256_file(dest, con)
        record_sha256(dest, digest, final_size)
        result["sha256"] = digest
        result["checksum_source"] = "computed"
        con.kv("sha256", digest[:32] + "...")
    return result


# --------------------------------------------------------------------------
# download manifest
# --------------------------------------------------------------------------

def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
