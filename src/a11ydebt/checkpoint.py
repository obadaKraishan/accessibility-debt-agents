"""
Resumable progress for long-running steps.

Parsing 66,261 Rico screens or 20 AndroidControl shards takes long enough
that you will interrupt it at least once. Every long step in this project
writes a checkpoint so that re-running the same script continues where it
stopped instead of starting over.

    from a11ydebt.checkpoint import Checkpoint
    ck = Checkpoint("parse_rico", PATHS.checkpoints)
    for screen_id in ids:
        if ck.is_done(screen_id):
            continue
        ...
        ck.mark(screen_id)          # buffered, flushed periodically
    ck.close()

The checkpoint file is plain JSON so you can inspect or hand-edit it.
Delete it (or pass --restart to any script) to force a clean run.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

__all__ = ["Checkpoint", "atomic_write_json"]


def atomic_write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Write JSON without ever leaving a half-written file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


class Checkpoint:
    """A resumable set of completed unit ids, plus free-form state."""

    def __init__(
        self,
        name: str,
        directory: str | Path,
        flush_every: int = 500,
        restart: bool = False,
    ) -> None:
        self.name = name
        self.path = Path(directory) / f"{name}.checkpoint.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self._pending = 0
        self._done: set[str] = set()
        self.state: dict[str, Any] = {}

        if restart:
            self.path.unlink(missing_ok=True)
        elif self.path.exists():
            try:
                blob = json.loads(self.path.read_text(encoding="utf-8"))
                self._done = set(blob.get("done", []))
                self.state = blob.get("state", {})
            except (json.JSONDecodeError, OSError):
                # A corrupt checkpoint must never block a run.
                self.path.unlink(missing_ok=True)

    # -- queries -----------------------------------------------------------

    @property
    def n_done(self) -> int:
        return len(self._done)

    @property
    def resumed(self) -> bool:
        return len(self._done) > 0

    def is_done(self, unit_id: Any) -> bool:
        return str(unit_id) in self._done

    def pending(self, unit_ids: Iterable[Any]) -> list[str]:
        return [str(u) for u in unit_ids if str(u) not in self._done]

    # -- mutation ----------------------------------------------------------

    def mark(self, unit_id: Any) -> None:
        self._done.add(str(unit_id))
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def set_state(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        self.flush()

    def flush(self) -> None:
        atomic_write_json(
            self.path,
            {
                "name": self.name,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_done": len(self._done),
                "state": self.state,
                "done": sorted(self._done),
            },
        )
        self._pending = 0

    def close(self) -> None:
        self.flush()

    def clear(self) -> None:
        self._done.clear()
        self.state.clear()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
