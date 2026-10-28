"""
Console output and logging for the a11ydebt pipeline.

Every script in this project prints the same way, so that when you run a
script you can see exactly which step it is on, how far along it is, and
where its outputs went. Colour is used when the terminal supports it and
is disabled automatically when output is piped to a file or when the
NO_COLOR environment variable is set.

Typical use inside a script:

    from a11ydebt.console import Console
    con = Console("01_parse_rico", log_dir=PATHS.logs)
    con.banner("PHASE 2 / STEP 1", "Parse Rico view hierarchies")
    con.step(1, 4, "Scanning archive")
    con.ok("Found 66,261 screens")
    con.kv("Screens parsed", 66261)
    con.warn("5 nodes missing rel-bounds")
    con.done({"nodes.parquet": path})

Nothing here depends on the rest of the project, so this module can be
imported by any script without side effects.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["Console", "supports_colour", "human_bytes", "human_time"]


# --------------------------------------------------------------------------
# colour handling
# --------------------------------------------------------------------------

def supports_colour() -> bool:
    """True when it is safe to emit ANSI colour codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


class _Palette:
    """ANSI codes, blanked out when colour is unavailable."""

    def __init__(self, enabled: bool) -> None:
        def code(value: str) -> str:
            return value if enabled else ""

        self.reset = code("\033[0m")
        self.bold = code("\033[1m")
        self.dim = code("\033[2m")
        self.blue = code("\033[38;5;61m")
        self.green = code("\033[38;5;29m")
        self.amber = code("\033[38;5;136m")
        self.red = code("\033[38;5;131m")
        self.grey = code("\033[38;5;245m")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def human_bytes(n: float) -> str:
    """1234567 -> '1.18 MB'."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1024.0
    return f"{n:,.2f} TB"


def human_time(seconds: float) -> str:
    """93.4 -> '1m 33s'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


# --------------------------------------------------------------------------
# console
# --------------------------------------------------------------------------

class Console:
    """Prints to the terminal and mirrors everything to a log file."""

    def __init__(
        self,
        name: str,
        log_dir: str | Path | None = None,
        width: int | None = None,
        quiet: bool = False,
    ) -> None:
        self.name = name
        self.quiet = quiet
        self._t0 = time.time()
        self._c = _Palette(supports_colour())
        self.width = width or min(shutil.get_terminal_size((88, 24)).columns, 88)

        self.logger = logging.getLogger(f"a11ydebt.{name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        self.log_path: Path | None = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = log_dir / f"{name}_{stamp}.log"
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
            )
            self.logger.addHandler(handler)

    # -- low level ---------------------------------------------------------

    def _emit(self, text: str, level: int = logging.INFO, raw: str | None = None) -> None:
        if not self.quiet:
            print(text, flush=True)
        self.logger.log(level, raw if raw is not None else _strip_ansi(text))

    # -- structure ---------------------------------------------------------

    def banner(self, title: str, subtitle: str | None = None) -> None:
        c = self._c
        rule = "=" * self.width
        self._emit(f"\n{c.blue}{rule}{c.reset}")
        self._emit(f"{c.bold}{c.blue}  {title}{c.reset}")
        if subtitle:
            self._emit(f"{c.grey}  {subtitle}{c.reset}")
        self._emit(f"{c.blue}{rule}{c.reset}")
        self._emit(f"{c.grey}  started {datetime.now():%Y-%m-%d %H:%M:%S}{c.reset}")
        if self.log_path:
            self._emit(f"{c.grey}  log     {self.log_path}{c.reset}")
        self._emit("")

    def section(self, title: str) -> None:
        c = self._c
        self._emit(f"\n{c.bold}{title}{c.reset}")
        self._emit(f"{c.dim}{'-' * min(len(title), self.width)}{c.reset}")

    def step(self, index: int, total: int, message: str) -> None:
        c = self._c
        self._emit(f"\n{c.blue}[{index}/{total}]{c.reset} {c.bold}{message}{c.reset}")

    # -- messages ----------------------------------------------------------

    def info(self, message: str) -> None:
        self._emit(f"      {message}")

    def ok(self, message: str) -> None:
        self._emit(f"  {self._c.green}OK{self._c.reset}  {message}")

    def warn(self, message: str) -> None:
        self._emit(f"  {self._c.amber}!!{self._c.reset}  {message}", logging.WARNING)

    def error(self, message: str) -> None:
        self._emit(f"  {self._c.red}XX{self._c.reset}  {message}", logging.ERROR)

    def note(self, message: str) -> None:
        self._emit(f"      {self._c.grey}{message}{self._c.reset}")

    def kv(self, label: str, value: Any, unit: str = "", raw: bool = False) -> None:
        if raw:
            shown = str(value)
        elif isinstance(value, float):
            shown = f"{value:,.4f}".rstrip("0").rstrip(".")
        elif isinstance(value, int):
            shown = f"{value:,}"
        else:
            shown = str(value)
        pad = "." * max(2, 34 - len(label))
        self._emit(f"      {label} {self._c.dim}{pad}{self._c.reset} {shown}{unit}")

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
        rows = [[("" if v is None else str(v)) for v in r] for r in rows]
        widths = [len(h) for h in headers]
        for r in rows:
            for i, cell in enumerate(r):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))
        c = self._c
        head = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        self._emit(f"      {c.dim}{head}{c.reset}")
        self._emit(f"      {c.dim}{'  '.join('-' * w for w in widths)}{c.reset}")
        for r in rows:
            line = "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))
            self._emit(f"      {line}")

    # -- progress ----------------------------------------------------------

    def progress(self, done: int, total: int, every: int = 1, label: str = "") -> None:
        """Single-line progress. Call inside a loop; it throttles itself."""
        if total <= 0:
            return
        if done != total and every > 1 and done % every != 0:
            return
        frac = done / total
        filled = int(frac * 28)
        bar = "#" * filled + "." * (28 - filled)
        elapsed = time.time() - self._t0
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
        msg = (
            f"      [{bar}] {frac * 100:5.1f}%  {done:,}/{total:,}"
            f"  elapsed {human_time(elapsed)}  eta {human_time(eta)}"
        )
        if label:
            msg += f"  {label}"
        if not self.quiet:
            print(msg[: self.width + 40], end="\r", flush=True)
        if done >= total:
            if not self.quiet:
                print()
            self.logger.info(_strip_ansi(msg.strip()))

    @contextmanager
    def timed(self, label: str):
        start = time.time()
        self.info(f"{label} ...")
        try:
            yield
        finally:
            self.ok(f"{label} finished in {human_time(time.time() - start)}")

    # -- closing -----------------------------------------------------------

    def done(self, artifacts: Mapping[str, Any] | None = None) -> None:
        c = self._c
        self._emit(f"\n{c.green}{'=' * self.width}{c.reset}")
        self._emit(f"{c.bold}{c.green}  COMPLETE{c.reset}  {self.name}")
        self._emit(f"{c.grey}  wall clock {human_time(time.time() - self._t0)}{c.reset}")
        if artifacts:
            self._emit(f"{c.grey}  files written:{c.reset}")
            for label, path in artifacts.items():
                size = ""
                p = Path(str(path))
                if p.exists() and p.is_file():
                    size = f"  ({human_bytes(p.stat().st_size)})"
                self._emit(f"    - {label}: {path}{size}")
        self._emit(f"{c.green}{'=' * self.width}{c.reset}\n")

    def fail(self, message: str, hint: str | None = None, code: int = 1) -> None:
        """Print a readable failure and exit. Never raises a bare traceback at the user."""
        c = self._c
        self._emit(f"\n{c.red}{'=' * self.width}{c.reset}")
        self._emit(f"{c.bold}{c.red}  FAILED{c.reset}  {self.name}", logging.ERROR)
        self._emit(f"  {message}", logging.ERROR)
        if hint:
            self._emit(f"\n  {c.amber}How to fix:{c.reset} {hint}", logging.ERROR)
        self._emit(f"{c.red}{'=' * self.width}{c.reset}\n")
        sys.exit(code)


def _strip_ansi(text: str) -> str:
    out, i = [], 0
    while i < len(text):
        if text[i] == "\033":
            while i < len(text) and text[i] != "m":
                i += 1
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
