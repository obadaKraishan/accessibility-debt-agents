"""
The reporting engine: one call writes TXT, CSV, and JSON for a research question.

The TXT file is the point of this module. It is a complete, self-contained
statistical report you can read straight into a Methods and Results
section: sample sizes, inclusion counts, descriptive tables, every test
with its statistic, exact p, effect size and confidence interval, an
auto-composed APA 7 narrative draft, a compact IEEE-style numeric summary,
and the provenance needed to reproduce the run.

    from a11ydebt.report import ResultsReport, Table

    rep = ResultsReport("rq1", "Label coverage in GUI corpora", outdir)
    rep.provenance(datasets={"rico": "v0.1"}, seed=20260908)
    rep.section("Sample and inclusion")
    rep.stat("Screens parsed", 66261)
    rep.proportion("Clickable nodes with a content description", 250, 2000)
    rep.table(Table("coverage_by_role", ["Role", "n", "Labelled", "%"], rows,
                    caption="Label coverage by element role."))
    rep.test("h1_clickable_gap", result, hypothesis="...", interpretation="...")
    rep.limitation("Rico was collected in 2017 and has not been updated.")
    paths = rep.write()

Outputs, for stem 'rq1':
    rq1_apa_report.txt      <- the human-readable statistical report
    rq1_results.json        <- everything, machine-readable
    rq1_results.csv         <- one row per reported statistic
    tables/rq1_<name>.csv   <- one file per table
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .stats import (
    TestResult, fmt_ci, fmt_p, fmt_pct, fmt_stat, proportion,
    interpret_h, interpret_v, interpret_delta, interpret_or,
)

__all__ = ["ResultsReport", "Table", "git_commit", "environment_info"]

WIDTH = 80
RULE = "=" * WIDTH
THIN = "-" * WIDTH


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------

def git_commit(root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root or Path.cwd()), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            commit = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(root or Path.cwd()), capture_output=True, text=True, timeout=5,
            )
            return commit + (" (dirty)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "not a git repository"


def environment_info() -> dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
    }
    for mod in ("numpy", "scipy", "pandas", "statsmodels", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    return info


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

@dataclass
class Table:
    """A reportable table. Rendered as fixed-width text and written as CSV."""

    name: str
    headers: Sequence[str]
    rows: Sequence[Sequence[Any]]
    caption: str = ""
    note: str = ""
    number: int | None = None

    def render(self) -> str:
        cells = [[_cell(v) for v in row] for row in self.rows]
        widths = [len(str(h)) for h in self.headers]
        for row in cells:
            for i, value in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(value))
        widths = [min(w, 32) for w in widths]

        def line(values: Sequence[str]) -> str:
            return "  ".join(str(v)[: widths[i]].ljust(widths[i]) for i, v in enumerate(values))

        parts = []
        title = f"Table {self.number}. " if self.number else ""
        if self.caption:
            parts.append(f"{title}{self.caption}")
        elif title:
            parts.append(title.strip())
        parts.append(line([str(h) for h in self.headers]))
        parts.append("  ".join("-" * w for w in widths))
        parts.extend(line(row) for row in cells)
        if self.note:
            parts.append("")
            parts.append(f"Note. {self.note}")
        return "\n".join(parts)

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.headers)
            writer.writerows(self.rows)
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "number": self.number, "caption": self.caption,
            "note": self.note, "headers": list(self.headers),
            "rows": [list(r) for r in self.rows],
        }


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.3e}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass
class _Entry:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class ResultsReport:
    """Accumulates results, then writes TXT + JSON + CSV in one call."""

    def __init__(
        self,
        stem: str,
        title: str,
        outdir: str | Path,
        subtitle: str = "",
        script: str = "",
    ) -> None:
        self.stem = stem
        self.title = title
        self.subtitle = subtitle
        self.script = script or (sys.argv[0] if sys.argv else "")
        self.outdir = Path(outdir)
        self.created = datetime.now()

        self._entries: list[_Entry] = []
        self._tests: dict[str, dict[str, Any]] = {}
        self._tables: list[Table] = []
        self._figures: list[dict[str, str]] = []
        self._limitations: list[str] = []
        self._notes: list[str] = []
        self._narrative: list[str] = []
        self._provenance: dict[str, Any] = {
            "environment": environment_info(),
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        }
        self._table_counter = 0

    # -- provenance --------------------------------------------------------

    def provenance(self, **kwargs: Any) -> "ResultsReport":
        """Record datasets, revisions, seeds, filters. Appears at the top of the TXT."""
        self._provenance.update(kwargs)
        return self

    # -- structure ---------------------------------------------------------

    def section(self, title: str, body: str = "") -> "ResultsReport":
        self._entries.append(_Entry("section", {"title": title, "body": body}))
        return self

    def text(self, body: str) -> "ResultsReport":
        self._entries.append(_Entry("text", {"body": body}))
        return self

    def stat(self, label: str, value: Any, unit: str = "", note: str = "") -> "ResultsReport":
        self._entries.append(_Entry("stat", {
            "label": label, "value": value, "unit": unit, "note": note,
        }))
        return self

    def proportion(
        self, label: str, k: int, n: int, note: str = "", conf: float = 0.95
    ) -> dict[str, Any]:
        """Record a proportion with its Wilson CI. Returns the computed dict."""
        p = proportion(k, n, conf)
        self._entries.append(_Entry("proportion", {"label": label, "note": note, **p}))
        return p

    def table(self, table: Table) -> Table:
        self._table_counter += 1
        table.number = self._table_counter
        self._tables.append(table)
        self._entries.append(_Entry("table", {"name": table.name}))
        return table

    def figure(self, name: str, path: str | Path, caption: str = "") -> "ResultsReport":
        self._figures.append({"name": name, "path": str(path), "caption": caption})
        return self

    def test(
        self,
        key: str,
        result: TestResult,
        hypothesis: str = "",
        interpretation: str = "",
    ) -> TestResult:
        """Record a statistical test. Interpretation is auto-filled if omitted."""
        if not interpretation:
            interpretation = _auto_interpret(result)
        payload = result.to_dict()
        payload["hypothesis"] = hypothesis
        payload["interpretation"] = interpretation
        self._tests[key] = payload
        self._entries.append(_Entry("test", {"key": key}))
        return result

    def narrative(self, paragraph: str) -> "ResultsReport":
        """Add a hand-written paragraph to the results-section draft."""
        self._narrative.append(paragraph.strip())
        return self

    def note(self, text: str) -> "ResultsReport":
        self._notes.append(text.strip())
        return self

    def limitation(self, text: str) -> "ResultsReport":
        self._limitations.append(text.strip())
        return self

    # -- rendering ---------------------------------------------------------

    def _header(self) -> str:
        lines = [
            RULE,
            f"  {self.title.upper()}",
        ]
        if self.subtitle:
            lines.append(f"  {self.subtitle}")
        lines += [
            RULE,
            "",
            f"Report id      : {self.stem}",
            f"Generated      : {self.created:%Y-%m-%d %H:%M:%S}",
            f"Script         : {self.script}",
            f"Git commit     : {self._provenance.get('git_commit', 'n/a')}",
        ]
        env = self._provenance.get("environment", {})
        if env:
            versions = ", ".join(f"{k} {v}" for k, v in env.items() if k != "platform")
            lines.append(f"Environment    : {env.get('platform', '')} | {versions}")
        for key, value in self._provenance.items():
            if key in {"environment", "git_commit"}:
                continue
            if isinstance(value, dict):
                lines.append(f"{key[:15].ljust(15)}:")
                for sub_k, sub_v in value.items():
                    lines.append(f"                 - {sub_k}: {sub_v}")
            else:
                lines.append(f"{key[:15].ljust(15)}: {value}")
        lines.append("")
        return "\n".join(lines)

    def _body(self) -> str:
        out: list[str] = []
        tables = {t.name: t for t in self._tables}
        section_no = 0

        for entry in self._entries:
            if entry.kind == "section":
                section_no += 1
                out += ["", THIN, f"{section_no}. {entry.payload['title'].upper()}", THIN]
                if entry.payload.get("body"):
                    out += ["", _wrap(entry.payload["body"])]
            elif entry.kind == "text":
                out += ["", _wrap(entry.payload["body"])]
            elif entry.kind == "stat":
                label = entry.payload["label"]
                value = entry.payload["value"]
                shown = f"{value:,}" if isinstance(value, int) else (
                    f"{value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
                )
                line = f"  {label.ljust(48, '.')} {shown}{entry.payload.get('unit', '')}"
                out.append(line)
                if entry.payload.get("note"):
                    out.append(f"      {entry.payload['note']}")
            elif entry.kind == "proportion":
                p = entry.payload
                out.append(
                    f"  {p['label'].ljust(48, '.')} {p['formatted']}"
                )
                out.append(
                    f"      95% CI [{p['ci_low_pct']:.2f}%, {p['ci_high_pct']:.2f}%]"
                    + (f"  {p['note']}" if p.get("note") else "")
                )
            elif entry.kind == "table":
                table = tables.get(entry.payload["name"])
                if table:
                    out += ["", table.render(), ""]
            elif entry.kind == "test":
                out += ["", _render_test(self._tests[entry.payload["key"]])]

        return "\n".join(out)

    def _results_draft(self) -> str:
        out = ["", THIN, "RESULTS SECTION DRAFT (APA 7)", THIN, ""]
        out.append(_wrap(
            "Auto-composed from the recorded statistics. Treat it as a first "
            "draft: check every claim against the tables above, and rewrite "
            "the prose in your own voice before it goes near the manuscript."
        ))
        out.append("")
        if self._narrative:
            for para in self._narrative:
                out += [_wrap(para), ""]
        for key, t in self._tests.items():
            sentence = []
            if t.get("hypothesis"):
                sentence.append(t["hypothesis"].rstrip("."))
                sentence.append(". ")
            sentence.append(f"A {t['name'].lower()} was conducted, {t['apa']}. ")
            interp = t["interpretation"]
            if interp.startswith("see "):
                sentence.append(
                    "The magnitude should be read from the estimate and its "
                    "confidence interval rather than a conventional benchmark."
                )
            else:
                sentence.append(f"The effect is {interp}.")
            out += [_wrap("".join(sentence)), ""]
        if not self._tests and not self._narrative:
            out.append("(No tests recorded.)")
        return "\n".join(out)

    def _ieee_summary(self) -> str:
        rows = []
        for key, t in self._tests.items():
            rows.append([
                key,
                t["statistic_name"],
                fmt_stat(t["statistic"]),
                "" if t["df"] is None else fmt_stat(t["df"], 0),
                fmt_p(t["p_value"]).replace("p ", "").replace("= ", ""),
                t["effect_name"],
                fmt_stat(t["effect_size"]),
                fmt_ci(t["ci_low"], t["ci_high"]),
                (f"{t.get('effect2_name','')} = {fmt_stat(t['effect2_size'])}"
                 if t.get("effect2_name") and t.get("effect2_size") is not None else ""),
                f"{t['n']:,}",
            ])
        table = Table(
            "ieee_summary",
            ["Test", "Stat", "Value", "df", "p", "Effect", "Size", "95% CI",
             "Std. effect", "N"],
            rows,
            caption="Compact numeric summary of all tests (IEEE style).",
            note="p reported exactly; '<.001' where smaller. Effect sizes carry 95% CIs.",
        )
        return "\n".join(["", THIN, "IEEE-STYLE NUMERIC SUMMARY", THIN, "", table.render()])

    def _closing(self) -> str:
        out = ["", THIN, "NOTES, ASSUMPTIONS AND LIMITATIONS", THIN, ""]
        if self._notes:
            out.append("Notes")
            for i, n in enumerate(self._notes, 1):
                out.append(_wrap(f"  {i}. {n}", indent="     "))
            out.append("")
        if self._limitations:
            out.append("Limitations")
            for i, l in enumerate(self._limitations, 1):
                out.append(_wrap(f"  {i}. {l}", indent="     "))
            out.append("")
        if self._figures:
            out.append("Figures")
            for f in self._figures:
                out.append(f"  - {f['name']}: {f['path']}")
                if f["caption"]:
                    out.append(_wrap(f"    {f['caption']}", indent="    "))
            out.append("")
        out += [RULE, f"  END OF REPORT — {self.stem}", RULE, ""]
        return "\n".join(out)

    # -- writing -----------------------------------------------------------

    def write(self, verbose_console: Any = None) -> dict[str, Path]:
        """Write all output files. Returns {label: path}."""
        reports = self.outdir / "reports"
        tables_dir = self.outdir / "tables"
        reports.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        txt_path = reports / f"{self.stem}_apa_report.txt"
        json_path = reports / f"{self.stem}_results.json"
        csv_path = reports / f"{self.stem}_results.csv"

        txt_path.write_text(
            self._header() + self._body() + "\n" + self._results_draft()
            + self._ieee_summary() + "\n" + self._closing(),
            encoding="utf-8",
        )

        payload = {
            "stem": self.stem,
            "title": self.title,
            "subtitle": self.subtitle,
            "generated_at": self.created.isoformat(timespec="seconds"),
            "script": self.script,
            "provenance": self._provenance,
            "descriptives": [
                e.payload for e in self._entries if e.kind in {"stat", "proportion"}
            ],
            "tests": self._tests,
            "tables": [t.to_dict() for t in self._tables],
            "figures": self._figures,
            "notes": self._notes,
            "limitations": self._limitations,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "kind", "key", "label", "value_or_statistic", "n", "df", "p_value",
                "effect_name", "effect_size", "ci_low", "ci_high", "interpretation",
            ])
            for e in self._entries:
                if e.kind == "stat":
                    writer.writerow(["descriptive", "", e.payload["label"],
                                     e.payload["value"], "", "", "", "", "", "", "", ""])
                elif e.kind == "proportion":
                    p = e.payload
                    writer.writerow(["proportion", "", p["label"], p["proportion"],
                                     p["n"], "", "", "", "", p["ci_low"], p["ci_high"], ""])
            for key, t in self._tests.items():
                writer.writerow([
                    "test", key, t["name"], t["statistic"], t["n"], t["df"], t["p_value"],
                    t["effect_name"], t["effect_size"], t["ci_low"], t["ci_high"],
                    t["interpretation"],
                ])

        written = {
            "APA/IEEE report (txt)": txt_path,
            "All results (json)": json_path,
            "Statistics (csv)": csv_path,
        }
        for table in self._tables:
            path = tables_dir / f"{self.stem}_{table.name}.csv"
            table.write_csv(path)
            written[f"Table {table.number}: {table.name} (csv)"] = path

        if verbose_console is not None:
            verbose_console.ok(f"Report written: {txt_path.name}")
        return written


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

def _auto_interpret(result: TestResult) -> str:
    # Prefer the standardised effect when one is reported alongside a raw one.
    if result.effect2_name and result.effect2_size is not None:
        second = result.effect2_name.lower()
        if "cohen's h" in second:
            return f"{interpret_h(result.effect2_size)} (by Cohen's h)"
        if "cramer" in second:
            return f"{interpret_v(result.effect2_size)} (by Cramer's V)"
    name = result.effect_name.lower()
    if "cohen's h" in name:
        return interpret_h(result.effect_size)
    if "cramer" in name:
        return interpret_v(result.effect_size)
    if "delta" in name:
        return interpret_delta(result.effect_size)
    if "odds" in name:
        return interpret_or(result.effect_size)
    return "see estimate and interval"


def _render_test(t: dict[str, Any]) -> str:
    lines = []
    if t.get("hypothesis"):
        lines.append(_wrap(t["hypothesis"]))
    lines.append(f"  {t['name']}")
    lines.append(f"    {t['apa']}")
    lines.append(f"    Interpretation: {t['interpretation']}")
    groups = t.get("groups") or {}
    for label, value in groups.items():
        if isinstance(value, dict) and "formatted" in value:
            lines.append(
                f"    {label}: {value['formatted']}  "
                f"95% CI [{value['ci_low_pct']:.2f}%, {value['ci_high_pct']:.2f}%]"
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            lines.append(f"    {label}: {value:,.4f}".rstrip("0").rstrip("."))
    if t.get("note"):
        lines.append(_wrap(f"    Note. {t['note']}", indent="          "))
    return "\n".join(lines)


def _wrap(text: str, width: int = WIDTH, indent: str = "") -> str:
    import textwrap
    out = []
    for para in str(text).split("\n"):
        if not para.strip():
            out.append("")
            continue
        lead = indent or (" " * (len(para) - len(para.lstrip())))
        out.append(textwrap.fill(
            para.strip(), width=width,
            initial_indent=para[: len(para) - len(para.lstrip())],
            subsequent_indent=lead,
        ))
    return "\n".join(out)
