"""
a11ydebt — Trained on What Screen Readers Cannot Read: Accessibility Debt in the GUI Corpora Behind Computer-Use Agents.

Shared library for the study of content-description coverage in the GUI
corpora used to train and evaluate computer-use agents (Rico, AndroidControl).

Modules
-------
config      Canonical paths and config.yaml / .env loading.
console     Uniform terminal output and per-run log files.
checkpoint  Resumable progress for long parses and downloads.
taxonomy    The operational codebook: element roles and label status.
stats       Tests that always return an effect size and a CI.
report      Writes the APA 7 / IEEE txt report, plus JSON and CSV.
"""

__version__ = "0.1.0"
__author__ = "Obada Kraishan"

from .config import CFG, PATHS, PROJECT_ROOT  # noqa: F401
from .console import Console                   # noqa: F401
from .checkpoint import Checkpoint             # noqa: F401
from .report import ResultsReport, Table       # noqa: F401

__all__ = [
    "CFG", "PATHS", "PROJECT_ROOT",
    "Console", "Checkpoint", "ResultsReport", "Table",
    "__version__",
]
