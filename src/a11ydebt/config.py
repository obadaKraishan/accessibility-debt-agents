"""
Project configuration and canonical paths.

Everything that could differ between machines lives in config.yaml and
.env. Nothing in this project hardcodes a path outside this module, so a
collaborator who clones the repo only has to copy .env.example to .env and
optionally point A11YDEBT_DATA_DIR at a bigger disk.

    from a11ydebt.config import CFG, PATHS
    print(PATHS.rico_raw)
    print(CFG.rico.download_url("unique_uis.tar.gz"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

__all__ = ["CFG", "PATHS", "PROJECT_ROOT", "Paths", "Config", "reload"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# .env loading (no external dependency required)
# --------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    outputs: Path
    logs: Path
    checkpoints: Path
    manuscript: Path

    # dataset-specific raw locations
    rico_raw: Path
    rico_screens: Path          # extracted combined/ directory
    androidcontrol_raw: Path
    rico_semantics_raw: Path

    def ensure(self) -> None:
        for value in vars(self).values():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)

    def analysis_dir(self, rq: str) -> Path:
        """phase5_analysis/<rq>/outputs — rq is e.g. 'rq1_label_coverage'."""
        return self.root / "phase5_analysis" / rq / "outputs"


def _build_paths() -> Paths:
    data = Path(os.environ.get("A11YDEBT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
    outputs = Path(os.environ.get("A11YDEBT_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
    raw = data / "raw"
    return Paths(
        root=PROJECT_ROOT,
        data=data,
        raw=raw,
        interim=data / "interim",
        processed=data / "processed",
        outputs=outputs,
        logs=outputs / "logs",
        checkpoints=outputs / "checkpoints",
        manuscript=outputs / "manuscript",
        rico_raw=raw / "rico",
        rico_screens=raw / "rico" / "combined",
        androidcontrol_raw=raw / "androidcontrol",
        rico_semantics_raw=raw / "rico_semantics",
    )


PATHS = _build_paths()


# --------------------------------------------------------------------------
# config.yaml
# --------------------------------------------------------------------------

class _Section:
    """Attribute access over a nested dict, so CFG.rico.base_url works."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data or {}

    def __getattr__(self, item: str) -> Any:
        if item not in self._data:
            raise AttributeError(
                f"config.yaml has no key '{item}' in this section. "
                f"Available: {sorted(self._data)}"
            )
        value = self._data[item]
        return _Section(value) if isinstance(value, dict) else value

    def get(self, item: str, default: Any = None) -> Any:
        return self._data.get(item, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Section({sorted(self._data)})"


class RicoSection(_Section):
    def download_url(self, filename: str) -> str:
        return f"{self._data['base_url'].rstrip('/')}/{filename}"


class AndroidControlSection(_Section):
    def shard_url(self, index: int) -> str:
        n = int(self._data["n_shards"])
        return f"{self._data['base_url'].rstrip('/')}/android_control-{index:05d}-of-{n:05d}"

    def aux_url(self, filename: str) -> str:
        return f"{self._data['base_url'].rstrip('/')}/{filename}"


@dataclass
class Config:
    _raw: dict[str, Any] = field(default_factory=dict)

    @property
    def rico(self) -> RicoSection:
        return RicoSection(self._raw.get("rico", {}))

    @property
    def androidcontrol(self) -> AndroidControlSection:
        return AndroidControlSection(self._raw.get("androidcontrol", {}))

    @property
    def rico_semantics(self) -> _Section:
        return _Section(self._raw.get("rico_semantics", {}))

    @property
    def analysis(self) -> _Section:
        return _Section(self._raw.get("analysis", {}))

    @property
    def run(self) -> _Section:
        return _Section(self._raw.get("run", {}))

    def as_dict(self) -> dict[str, Any]:
        return dict(self._raw)


def _load_config() -> Config:
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. It ships with the repository; "
            "restore it from git before running any script."
        )
    if yaml is None:
        raise ImportError("PyYAML is required. Run: pip install -r requirements.txt")
    return Config(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


CFG = _load_config()


def reload() -> Config:
    """Re-read config.yaml (useful in notebooks after editing it)."""
    global CFG
    CFG = _load_config()
    return CFG
