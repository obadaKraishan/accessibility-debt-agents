# Trained on What Screen Readers Cannot Read: Accessibility Debt in the GUI Corpora Behind Computer-Use Agents

**Missing labels in the GUI corpora that train computer-use agents.**

Computer-use agents read the accessibility tree. That tree was built for screen
readers, and it is famously incomplete. This repository measures how incomplete
it is across the two corpora agents are actually trained and evaluated on, and
then asks the question that turns a coverage table into a finding: **do benchmark
tasks disproportionately touch the elements that happen to be labelled?** If they
do, agent evaluation carries an optimism bias, and that bias is an accessibility
artifact.

Target venue: IEEE AISSA 2026.

---

## Research questions

| | Question | Script |
|---|---|---|
| **RQ1** | What is `content-desc` coverage across Rico and AndroidControl, by element type, app category and Play Store rating? | `phase5_analysis/rq1_label_coverage/run_rq1.py` |
| **RQ2** | Has coverage improved between the 2017 Rico snapshot and the 2024-25 AndroidControl apps, and for which element classes? | `phase5_analysis/rq2_temporal_change/run_rq2.py` |
| **RQ3** | Do benchmark task trajectories touch labelled elements at a higher rate than the background population? | `phase5_analysis/rq3_trajectory_bias/run_rq3.py` |
| **RQ4** | Does label density predict task difficulty, and how much of benchmark performance is attributable to it? | `phase5_analysis/rq4_difficulty_model/run_rq4.py` |

---

## Quick start

```bash
git clone <this repo> && cd accessibility-debt-agents
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python phase0_setup/01_verify_environment.py           # checks everything, downloads nothing
python phase0_setup/02_init_project.py                 # self-tests the reporting stack
```

Step 2 writes a **complete example statistical report** to
`outputs/manuscript/demo/reports/demo_apa_report.txt` using synthetic data. Read
it before downloading anything: it shows exactly what every analysis script will
produce, and it is the cheapest moment to change the format.
It also writes `outputs/manuscript/project_manifest.json`, a record of this
machine's paths and settings; that file is regenerated on every run and is not
committed.

Then work through the phases in order. Every script prints what it is doing,
saves progress, and tells you what to run next.

```bash
# Phase 1 — data (Rico 6 GB, AndroidControl up to 46.5 GB; both resumable)
python phase1_collection/01_download_rico.py
python phase1_collection/02_download_androidcontrol.py --shards 20
python phase1_collection/03_download_rico_semantics.py

# Phase 2 — parse
python phase2_processing/01_parse_rico.py
python phase2_processing/02_parse_androidcontrol.py
python phase2_processing/03_build_analysis_tables.py

# Phase 3 — codebook reliability (needs you to hand-code a sample)
python phase3_codebook/01_draw_validation_sample.py
#   ... code phase3_codebook/outputs/validation_sample_blank.csv ...
python phase3_codebook/02_intercoder_reliability.py

# Phase 4 — validate before you trust a single number
python phase4_validation/01_validate_datasets.py
python phase4_validation/02_sensitivity_checks.py

# Phase 5 — the research questions
python phase5_analysis/rq1_label_coverage/run_rq1.py
python phase5_analysis/rq2_temporal_change/run_rq2.py
python phase5_analysis/rq3_trajectory_bias/run_rq3.py
python phase5_analysis/rq4_difficulty_model/run_rq4.py

# Phase 6 — assemble the paper
python phase6_synthesis/01_manuscript_numbers.py
python phase6_synthesis/02_figure_panel.py
```

Or `make all`. Every script also accepts `--limit N` for a fast pilot,
`--restart` to ignore checkpoints, `--quiet`, `--seed`, and `--out`.

### Try it without the full download

```bash
python phase1_collection/01_download_rico.py --metadata-only        # 4 MB
python phase1_collection/02_download_androidcontrol.py --splits-only # 120 KB
python phase1_collection/02_download_androidcontrol.py --shards 2    # ~5 GB
```

---

## What every script produces

Each analysis writes four things into its own `outputs/` directory:

```
outputs/reports/<stem>_apa_report.txt    the statistical report — read this
outputs/reports/<stem>_results.json      everything, machine-readable
outputs/reports/<stem>_results.csv       one row per statistic
outputs/tables/<stem>_<name>.csv         one file per table
outputs/figures/<stem>_<name>.pdf/.png   vector for the manuscript
```

The **`_apa_report.txt`** is the point. It is a self-contained statistical
report you can read straight into a Method and Results section: sample sizes,
inclusion counts, descriptive tables, every test with its statistic, exact *p*,
effect size and confidence interval, an auto-composed APA 7 narrative draft, a
compact IEEE-style numeric summary, and the provenance to reproduce the run
(git commit, package versions, random seed, dataset checksums).

`phase6_synthesis/01_manuscript_numbers.py` then merges every report into
`outputs/manuscript/manuscript_numbers.txt` and cross-checks that no quantity
appears with two different values in two places.

---

## Layout

```
src/a11ydebt/          shared library — config, console, checkpoints, taxonomy,
                       statistics, reporting, figures, TFRecord readers
phase0_setup/          environment verification and self-test          (2 scripts)
phase1_collection/     resumable downloads with checksums              (3 scripts)
phase2_processing/     parsers and the unified analysis tables         (3 scripts)
phase3_codebook/       CODEBOOK.md, validation sample, kappa           (2 scripts)
phase4_validation/     data checks and sensitivity analysis            (2 scripts)
phase5_analysis/       one directory per research question             (4 scripts)
phase6_synthesis/      manuscript numbers and the main figure panel    (2 scripts)
data/                  raw, interim, processed        (never committed)
outputs/               logs, checkpoints, manuscript   (never committed)
docs/                  roadmap, data licences, design notes
```

---

## Data and licences — read before sharing anything

| Dataset | Licence | Redistribution |
|---|---|---|
| **Rico** | University of Illinois research-use agreement | **No.** Not an open licence. Screenshots may contain third-party copyrighted material. |
| **AndroidControl** | Apache-2.0 | Yes |
| **RICO-Semantics** | CC BY-SA 4.0 | Yes, with ShareAlike |

`.gitignore` excludes `data/` entirely. Do not commit raw data, and do not
publish a derived dataset that embeds Rico content without checking the terms.
Full detail in [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md).

The code in this repository is MIT. That covers the code only, not the data.

---

## Three things worth knowing before you start

**AndroidControl actions carry no element reference.** The complete key set
across every action type is `{action_type, x, y, text, direction, app_name}`.
Which element a trajectory touched — the thing RQ3 turns on — is not in the
data and is inferred by point-in-bounds hit testing. The hit rate is reported
on every parse run so a degradation is visible immediately.

**RQ4's outcome is demonstration length, not agent success.** AndroidControl
contains human demonstrations and no agent rollouts, so it has no per-episode
success field. The script says so in its own report, and the design note is in
the roadmap. A reviewer who knows the dataset will check for exactly this.

**TensorFlow is optional.** The dataset README tells you to install TensorFlow
and clone `google-deepmind/android_env`. This project ships its own TFRecord and
protobuf readers (`src/a11ydebt/tfrecord.py`) and uses them by default. Pass
`--verify-tf` to cross-check them against the official path when TensorFlow is
available.

---

## Citation

Accepted for oral presentation at IEEE AISSA 2026.

> Obada Kraishan and Kulsawasd Jitkajornwanich. "Trained on What Screen Readers Cannot Read: Accessibility Debt in the GUI Corpora Behind Computer-Use Agents." In *IEEE AISSA 2026*, 2026.

```bibtex
@inproceedings{kraishan2026trained,
  title     = {Trained on What Screen Readers Cannot Read: Accessibility Debt in the GUI Corpora Behind Computer-Use Agents},
  author    = {Kraishan, Obada and Jitkajornwanich, Kulsawasd},
  booktitle = {IEEE AISSA 2026},
  year      = {2026}
}
```

The same reference is in [`CITATION.cff`](CITATION.cff). Cite the datasets too:

- Deka et al. (2017). *Rico: A Mobile App Dataset for Building Data-Driven Design Applications.* UIST '17. https://doi.org/10.1145/3126594.3126651
- Li et al. (2024). *On the Effects of Data Scale on UI Control Agents.* NeurIPS 2024 Datasets and Benchmarks. arXiv:2406.03679
- Sunkara et al. (2022). *Towards Better Semantic Understanding of Mobile Interfaces.* arXiv:2210.02663
