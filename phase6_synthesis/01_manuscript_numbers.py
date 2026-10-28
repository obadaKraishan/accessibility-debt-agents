#!/usr/bin/env python3
"""
PHASE 6 / STEP 1 — Assemble every number the manuscript will cite.

Reads the results JSON from validation, sensitivity, the reviewer checks,
inter-rater reliability and all four research questions, and writes one file
holding every statistic and table in the paper with the script and report
that produced it. When a reviewer asks where a number came from, this file
answers in one line.

    python phase6_synthesis/01_manuscript_numbers.py

It also runs consistency checks across reports, because the most embarrassing
error in a computational paper is the same quantity appearing with two
different values in the abstract and the results section.

Outputs
-------
outputs/manuscript/manuscript_numbers.txt    <- keep this open while writing
outputs/manuscript/manuscript_numbers.json
outputs/manuscript/manuscript_numbers.csv
outputs/manuscript/figure_index.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a11ydebt.checkpoint import atomic_write_json             # noqa: E402
from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import PATHS, PROJECT_ROOT               # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.report import Table                             # noqa: E402
from a11ydebt.stats import fmt_p, fmt_stat                    # noqa: E402

SOURCES = [
    ("Validation", "phase4_validation/outputs/reports/validation_results.json"),
    ("Sensitivity", "phase4_validation/outputs/reports/sensitivity_results.json"),
    ("ReviewerChecks", "phase4_validation/outputs/reports/reviewer_checks_results.json"),
    ("Reliability", "phase3_codebook/outputs/reports/codebook_irr_results.json"),
    ("RQ1", "phase5_analysis/rq1_label_coverage/outputs/reports/rq1_results.json"),
    ("RQ2", "phase5_analysis/rq2_temporal_change/outputs/reports/rq2_results.json"),
    ("RQ3", "phase5_analysis/rq3_trajectory_bias/outputs/reports/rq3_results.json"),
    ("RQ4", "phase5_analysis/rq4_difficulty_model/outputs/reports/rq4_results.json"),
]

WIDTH = 80


def main() -> int:
    parser = base_parser("Assemble every number the manuscript will cite.")
    args = parser.parse_args()

    con = Console("01_manuscript_numbers", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 6 / STEP 1", "Assemble the manuscript numbers")

    con.step(1, 3, "Collect reports")
    collected, missing = {}, []
    for name, rel in SOURCES:
        path = PROJECT_ROOT / rel
        if path.exists():
            collected[name] = json.loads(path.read_text(encoding="utf-8"))
            con.ok(f"{name}: {path.relative_to(PROJECT_ROOT)}")
        else:
            missing.append(name)
            con.warn(f"{name}: not found ({rel})")
    if not collected:
        con.fail("No results found.",
                 hint="Run the phase 4 and phase 5 scripts first.")
    if missing:
        con.note(f"Missing: {', '.join(missing)}. The assembled file will be "
                 f"incomplete until those run.")

    con.step(2, 3, "Cross-check shared quantities")
    # The same proportion often appears in more than one report. If two reports
    # disagree, one of them is stale.
    seen: dict[str, list[tuple[str, float]]] = {}
    for name, blob in collected.items():
        for entry in blob.get("descriptives", []):
            label = str(entry.get("label", "")).strip().lower()
            if "proportion" in entry:
                seen.setdefault(label, []).append((name, float(entry["proportion"])))
    conflicts = []
    for label, values in seen.items():
        distinct = {round(v, 6) for _, v in values}
        if len(values) > 1 and len(distinct) > 1:
            conflicts.append([label, "; ".join(f"{n}={v:.4f}" for n, v in values)])
    if conflicts:
        con.warn(f"{len(conflicts)} quantity/quantities disagree between reports:")
        con.table(["Quantity", "Values"], conflicts)
        con.note("Usually one report is stale. Re-run the older script.")
    else:
        con.ok("No conflicting values across reports")

    con.step(3, 3, "Write")
    lines = ["=" * WIDTH,
             "  MANUSCRIPT NUMBERS",
             "  Every statistic cited in the paper, with its source",
             "=" * WIDTH, ""]
    if missing:
        lines += [f"INCOMPLETE. Missing reports: {', '.join(missing)}", ""]
    if conflicts:
        lines += ["WARNING: conflicting values were detected across reports.", ""]
        for label, values in conflicts:
            lines.append(f"  {label}: {values}")
        lines.append("")

    rows: list[list] = []
    figures: list[list] = []
    for name, blob in collected.items():
        lines += ["-" * WIDTH, f"{name}: {blob.get('title', '')}", "-" * WIDTH]
        lines.append(f"  source : {blob.get('script', '')}")
        lines.append(f"  written: {blob.get('generated_at', '')}")
        lines.append("")

        descriptives = blob.get("descriptives", [])
        if descriptives:
            lines.append("  Descriptives")
            for entry in descriptives:
                label = entry.get("label", "")
                if "formatted" in entry:
                    value = (f"{entry['formatted']}  95% CI "
                             f"[{entry['ci_low_pct']:.2f}%, {entry['ci_high_pct']:.2f}%]")
                else:
                    raw = entry.get("value")
                    value = f"{raw:,}" if isinstance(raw, int) else str(raw)
                    # A stat's note can carry its interval (e.g. a bootstrap
                    # CI), which the paper may cite, so it is printed too.
                    if entry.get("note"):
                        value += f"  ({entry['note']})"
                lines.append(f"    {str(label).ljust(46, '.')} {value}")
                rows.append([name, "descriptive", label, value, "", "", "", ""])
            lines.append("")

        tests = blob.get("tests", {})
        if tests:
            lines.append("  Tests")
            for key, t in tests.items():
                lines.append(f"    [{key}]")
                lines.append(f"      {t.get('apa', '')}")
                lines.append(f"      interpretation: {t.get('interpretation', '')}")
                if t.get("hypothesis"):
                    lines.append(f"      hypothesis: {t['hypothesis']}")
                rows.append([
                    name, "test", key, fmt_stat(t.get("statistic", float("nan"))),
                    t.get("n", ""), fmt_p(t.get("p_value", float("nan"))),
                    f"{t.get('effect_name', '')} = {fmt_stat(t.get('effect_size', float('nan')))}",
                    f"[{fmt_stat(t.get('ci_low', float('nan')))}, "
                    f"{fmt_stat(t.get('ci_high', float('nan')))}]",
                ])
            lines.append("")

        # Tables too, so a number a paper table cites can be traced here.
        tables = blob.get("tables", [])
        if tables:
            lines.append("  Tables")
            for t in tables:
                rendered = Table(name=t["name"], headers=t["headers"], rows=t["rows"],
                                 caption=t.get("caption", ""), note=t.get("note", ""),
                                 number=t.get("number")).render()
                lines += ["    " + row for row in rendered.splitlines()]
                lines.append("")

        for fig in blob.get("figures", []):
            figures.append([name, fig.get("name", ""), fig.get("path", ""),
                            fig.get("caption", "")])

        limits = blob.get("limitations", [])
        if limits:
            lines.append("  Limitations declared")
            for i, item in enumerate(limits, 1):
                lines.append(f"    {i}. {item}")
            lines.append("")

    lines += ["=" * WIDTH,
              "  Paste these into the manuscript verbatim. If a number in the",
              "  paper is not in this file, it did not come from the pipeline.",
              "=" * WIDTH, ""]

    out_txt = PATHS.manuscript / "manuscript_numbers.txt"
    out_json = PATHS.manuscript / "manuscript_numbers.json"
    out_csv = PATHS.manuscript / "manuscript_numbers.csv"
    fig_csv = PATHS.manuscript / "figure_index.csv"

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(out_json, {"reports": collected, "missing": missing,
                                 "conflicts": conflicts})
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["report", "kind", "key", "statistic", "n", "p",
                         "effect", "ci"])
        writer.writerows(rows)
    with fig_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["report", "figure", "path", "caption"])
        writer.writerows(figures)

    con.kv("Reports collected", len(collected))
    con.kv("Statistics assembled", len(rows))
    con.kv("Figures indexed", len(figures))
    con.done({"manuscript_numbers.txt": out_txt,
              "manuscript_numbers.json": out_json,
              "manuscript_numbers.csv": out_csv,
              "figure_index.csv": fig_csv})
    con.info("Next:  python phase6_synthesis/02_figure_panel.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
