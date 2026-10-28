#!/usr/bin/env python3
"""
RQ4 — Does label density predict task difficulty, and how much of benchmark
performance is attributable to it?

    python phase5_analysis/rq4_difficulty_model/run_rq4.py

A DESIGN CORRECTION YOU SHOULD READ BEFORE RUNNING THIS
-------------------------------------------------------
The original framing of this question was "does label density predict agent
success". It cannot be answered with AndroidControl alone, because
AndroidControl contains HUMAN DEMONSTRATIONS, not agent rollouts. There is
no per-episode agent success field in the data, and inventing one by
running an agent would cost money, time, and an API budget this study does
not have before the deadline.

So the outcome is reframed to something the data actually supports:

    outcome = steps to completion in the human demonstration
              (a difficulty proxy, not an accuracy measure)

plus a second, ecological analysis at the level of the benchmark's own
generalisation subsplits, where published per-split accuracies can be set
against per-split label density.

State this in the paper plainly. A reviewer who knows AndroidControl will
check for exactly this confusion, and naming it yourself converts a possible
rejection into a demonstration of care. If you later obtain agent rollouts
(the HAL trace archives are one public source), the same script structure
takes an agent-success outcome with a one-line change.

HYPOTHESES
----------
H4.1  Episodes on screens with lower label density take more steps.
H4.2  The association survives controlling for the number of elements on
      screen, which is the obvious confound: busier screens are both harder
      and more likely to contain unlabelled elements.
H4.3  Label density differs across the benchmark's generalisation subsplits,
      which would mean the subsplits differ in accessibility as well as in
      the distribution shift they are designed to test.

MODELS 2 AND 3 (camera-ready, Reviewer 2)
-----------------------------------------
Model 1 is the original H4.2 model, unchanged. Model 2 adds the published
split and clusters standard errors by app. Model 1's element count is the
total across the episode's screens, so it grows with the number of screens
and therefore with the outcome. Model 3 takes both label density and element
count from the episode's first in-app screen, so nothing in it depends on how
the demonstration unfolds; it is the specification the paper reports as the
answer. Specification checks vary one of those choices at a time. Every
result is an association; none says that labelling changes task length.

Outputs
-------
outputs/reports/rq4_apa_report.txt
outputs/tables/rq4_*.csv
outputs/figures/rq4_*.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import pyarrow.parquet as pq                                  # noqa: E402

from a11ydebt.apps import androidcontrol_episode_apps         # noqa: E402
from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.figures import (                                # noqa: E402
    COLUMN, CATEGORICAL, save, use_paper_style,
)
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.stats import (                                  # noqa: E402
    TestResult, bootstrap_ci, chi_square_test, cliffs_delta,
    interpret_delta, permutation_test, proportion,
)

MIN_EPISODES = 100
SPLITS = ("validation", "test")       # published split dummies; train is the reference


def fit_negbin(df: pd.DataFrame, exposure: str, covariates: list[str],
               groups: pd.Series | None = None) -> dict:
    """Negative binomial of steps; returns the exposure's IRR, CI, p and fit facts."""
    import statsmodels.api as sm
    X = sm.add_constant(df[[exposure] + covariates].astype(float))
    kwargs = {}
    if groups is not None:
        kwargs = {"cov_type": "cluster",
                  "cov_kwds": {"groups": pd.factorize(groups)[0]}}
    fit = sm.NegativeBinomial(df["steps"], X).fit(disp=0, maxiter=200, **kwargs)
    coef, se = float(fit.params[exposure]), float(fit.bse[exposure])
    return {
        "irr": float(np.exp(coef)),
        "lo": float(np.exp(coef - 1.96 * se)),
        "hi": float(np.exp(coef + 1.96 * se)),
        "z": coef / se if se else 0.0,
        "p": float(fit.pvalues[exposure]),
        "n": int(len(df)),
        "clusters": int(groups.nunique()) if groups is not None else None,
        "converged": bool(fit.mle_retvals.get("converged", False)),
        "alpha": float(fit.params.get("alpha", float("nan"))),
    }


def main() -> int:
    parser = base_parser("RQ4: label density and task difficulty.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    outdir = Path(args.out) if args.out else here / "outputs"
    con = Console("run_rq4", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("RQ4", "Label density and task difficulty")

    episodes_path = PATHS.processed / "androidcontrol_episodes.parquet"
    if not episodes_path.exists():
        con.fail(f"Missing {episodes_path}",
                 hint="Run phase2_processing/02_parse_androidcontrol.py")

    con.step(1, 6, "Load episodes")
    ep = pd.read_parquet(episodes_path)
    con.kv("Episodes", len(ep))
    if len(ep) < MIN_EPISODES:
        con.fail(
            f"Only {len(ep)} episodes; RQ4 needs at least {MIN_EPISODES}.",
            hint="Download and parse more shards:\n"
                 "             python phase1_collection/02_download_androidcontrol.py --shards 10\n"
                 "             python phase2_processing/02_parse_androidcontrol.py",
        )

    ep = ep[ep["n_needs_label"] > 0].copy()
    ep["label_density"] = ep["n_labelled"] / ep["n_needs_label"]
    ep["steps"] = ep["n_actions"].astype(float)
    ep["elements_per_step"] = ep["n_included"] / ep["n_observations"].clip(lower=1)
    con.kv("Episodes with labellable elements", len(ep))
    con.kv("Median label density", f"{ep['label_density'].median():.3f}")
    con.kv("Median steps", f"{ep['steps'].median():.1f}")

    rep = ResultsReport(
        stem="rq4",
        title="RQ4: Label density and task difficulty",
        subtitle="Demonstration length as the outcome, not agent success",
        outdir=outdir,
        script="phase5_analysis/rq4_difficulty_model/run_rq4.py",
    )
    rep.provenance(episodes_table=str(episodes_path), seed=args.seed,
                   outcome=CFG.analysis.rq4.outcome)
    rep.section(
        "What this analysis can and cannot claim",
        "AndroidControl contains human demonstrations, not agent rollouts. It "
        "carries no per-episode agent success field, so 'does label density "
        "predict agent success' cannot be answered from this dataset. The "
        "outcome used here is the number of steps in the human demonstration, "
        "which indexes task difficulty rather than agent accuracy. Any claim "
        "about agent performance must come from the ecological subsplit "
        "analysis below, or from a separate corpus of agent rollouts.",
    )
    rep.stat("Episodes analysed", int(len(ep)))
    rep.stat("Median steps per episode", float(ep["steps"].median()))
    rep.stat("Median label density", float(ep["label_density"].median()))

    con.step(2, 6, "H4.1 — density and demonstration length")
    rep.section("Label density and demonstration length")
    ep["density_group"] = np.where(
        ep["label_density"] >= ep["label_density"].median(), "higher", "lower")
    high = ep.loc[ep["density_group"] == "higher", "steps"].to_numpy()
    low = ep.loc[ep["density_group"] == "lower", "steps"].to_numpy()

    for name, values in (("lower density", low), ("higher density", high)):
        point, lo, hi = bootstrap_ci(values, np.mean,
                                     n_boot=min(2000, int(CFG.run.n_bootstrap)),
                                     seed=args.seed)
        rep.stat(f"Mean steps, {name}", round(point, 3),
                 note=f"95% CI [{lo:.3f}, {hi:.3f}], n = {len(values):,}")
        con.kv(f"Mean steps ({name})", f"{point:.2f} [{lo:.2f}, {hi:.2f}]")

    if len(high) >= 30 and len(low) >= 30:
        result = permutation_test(
            low, high, n_perm=int(CFG.run.n_permutations), seed=args.seed,
            name="Permutation test, steps by label density group",
            effect_name="difference in mean steps",
        )
        delta = cliffs_delta(low, high)
        result.effect2_name = "Cliff's delta"
        result.effect2_size = delta
        result.note += (f" Cliff's delta = {delta:.3f} ({interpret_delta(delta)}), "
                        f"a rank-based effect size robust to the skew in step counts.")
        rep.test("h4_1_density_steps", result,
                 hypothesis="H4.1. Episodes on screens with lower label density "
                            "take more steps to complete.")

    con.step(3, 6, "H4.2 — controlling for screen complexity")
    rep.section("Controlling for screen complexity")
    rep.text(
        "Busier screens carry both more elements and more unlabelled ones, so "
        "an association between density and difficulty could be complexity in "
        "disguise. This model enters element count alongside density."
    )
    h42_irr = None
    try:
        import statsmodels.api as sm
        model_df = ep[["steps", "label_density", "n_included", "n_observations"]].dropna()
        model_df = model_df[model_df["steps"] > 0]
        if len(model_df) >= 50:
            X = sm.add_constant(pd.DataFrame({
                "label_density": model_df["label_density"],
                "log_elements": np.log1p(model_df["n_included"]),
            }))
            # Step counts are non-negative and over-dispersed, so a negative
            # binomial is the right family; Poisson would understate the SEs.
            fit = sm.NegativeBinomial(model_df["steps"], X).fit(disp=0)
            rows = []
            for term in fit.params.index:
                if term == "alpha":
                    continue
                coef = float(fit.params[term])
                se = float(fit.bse[term])
                rows.append([
                    term, f"{coef:+.4f}", f"{se:.4f}",
                    f"{np.exp(coef):.4f}",
                    f"[{np.exp(coef - 1.96 * se):.4f}, {np.exp(coef + 1.96 * se):.4f}]",
                    f"{float(fit.pvalues[term]):.4f}",
                ])
            rep.table(Table(
                "negbin_model",
                ["Term", "Coefficient", "SE", "Incidence rate ratio", "95% CI", "p"],
                rows,
                caption="Negative binomial regression of demonstration length on "
                        "label density, controlling for screen complexity.",
                note="Incidence rate ratios below 1 mean fewer steps. Step counts "
                     "are over-dispersed counts, so a negative binomial is used "
                     "rather than Poisson, which would understate the standard errors.",
            ))
            con.table(["Term", "IRR", "p"], [[r[0], r[3], r[5]] for r in rows])

            if "label_density" in fit.params.index:
                coef = float(fit.params["label_density"])
                se = float(fit.bse["label_density"])
                h42_irr = float(np.exp(coef))
                rep.test(
                    "h4_2_density_adjusted",
                    TestResult(
                        name="Negative binomial regression (adjusted)",
                        statistic_name="z", statistic=coef / se if se else 0.0,
                        p_value=float(fit.pvalues["label_density"]), df=None,
                        effect_name="incidence rate ratio",
                        effect_size=float(np.exp(coef)),
                        ci_low=float(np.exp(coef - 1.96 * se)),
                        ci_high=float(np.exp(coef + 1.96 * se)),
                        n=int(len(model_df)),
                        groups={"pseudo_r2": float(getattr(fit, "prsquared", float("nan"))),
                                "alpha": float(fit.params.get("alpha", float("nan")))},
                        note="Adjusted for log element count. The IRR is the "
                             "multiplicative change in expected steps per unit "
                             "of label density, i.e. moving from none labelled "
                             "to all labelled.",
                    ),
                    hypothesis="H4.2. The association between label density and "
                               "demonstration length survives adjustment for "
                               "screen complexity.",
                )
        else:
            con.warn(f"Only {len(model_df)} usable episodes; skipping the model")
            rep.note("The adjusted model was skipped for lack of episodes.")
    except ImportError:
        con.warn("statsmodels is not installed; skipping the adjusted model")
        rep.note("statsmodels was unavailable, so H4.2 was not tested. "
                 "Install it with: pip install statsmodels")

    con.step(4, 6, "H4.2 extended — Models 2 and 3, and specification checks")
    nodes_path = PATHS.processed / "androidcontrol_nodes.parquet"
    steps_path = PATHS.processed / "androidcontrol_steps.parquet"
    for path in (nodes_path, steps_path):
        if not path.exists():
            con.fail(f"Missing {path}", hint="Run phase2_processing/02_parse_androidcontrol.py")
    min_first = int(CFG.analysis.rq4.min_first_screen_needs_label)
    splits = [f"split_{s}" for s in SPLITS]

    md = ep[ep["steps"] > 0].copy()
    md["log_elements"] = np.log1p(md["n_included"])
    unexpected = set(md["split"].unique()) - {"train", *SPLITS}
    if unexpected:
        con.fail(f"Unexpected split labels {sorted(unexpected)}.",
                 hint="Check the split column of androidcontrol_episodes.parquet.")
    for s in SPLITS:
        md[f"split_{s}"] = (md["split"] == s).astype(float)
    apps = (androidcontrol_episode_apps(nodes_path, CFG.androidcontrol.system_packages)
            .set_index("episode_id")["app"])
    md["app"] = md["episode_id"].map(apps).fillna("episode:" + md["episode_id"].astype(str))

    # Nodes carry the observation index; observation 0 is the screen shown
    # before the first action. When that action is open_app, observation 0
    # is the screen the app was opened from, so the checks also use the first
    # in-app screen: observation 1 after open_app, otherwise observation 0.
    nodes = pq.read_table(nodes_path, columns=[
        "episode_id", "step_index", "included", "needs_label", "label_status"]).to_pandas()
    nodes = nodes[nodes["included"]]
    nodes["lab"] = nodes["needs_label"] & (nodes["label_status"] == "labelled")
    first_action = (pd.read_parquet(steps_path, columns=["episode_id", "step_index", "action_type"])
                      .query("step_index == 0").set_index("episode_id")["action_type"])
    md["opens_app"] = md["episode_id"].map(first_action).eq("open_app")

    def screen_counts(nodes: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
        key = pd.DataFrame({"episode_id": md["episode_id"].to_numpy(),
                            "step_index": index.astype(nodes["step_index"].dtype)})
        hit = nodes.merge(key, on=["episode_id", "step_index"])
        counts = hit.groupby("episode_id").agg(
            elements=("lab", "size"), need=("needs_label", "sum"), lab=("lab", "sum"))
        return counts.reindex(md["episode_id"].to_numpy()).fillna(0).set_axis(md.index)

    first = screen_counts(nodes, np.zeros(len(md), dtype=int))
    in_app = screen_counts(nodes, np.where(md["opens_app"], 1, 0))
    del nodes
    m3 = md[first["need"] >= min_first].copy()
    m3["first_density"] = first.loc[m3.index, "lab"] / first.loc[m3.index, "need"]
    m3["log_first_elements"] = np.log1p(first.loc[m3.index, "elements"])
    ma = md[in_app["need"] >= min_first].copy()
    ma["app_density"] = in_app.loc[ma.index, "lab"] / in_app.loc[ma.index, "need"]
    ma["log_app_elements"] = np.log1p(in_app.loc[ma.index, "elements"])

    screens_plus_one = int(((md["n_observations"] - md["n_actions"]) == 1).sum())
    length_corr = float(np.corrcoef(md["log_elements"], np.log(md["steps"]))[0, 1])
    con.kv("Episodes with screens visited = steps + 1", f"{screens_plus_one:,} of {len(md):,}")
    con.kv("Correlation, log total elements vs log steps", f"{length_corr:.3f}")
    con.kv(f"Model 3 episodes (>= {min_first} needs-label on first in-app screen)",
           f"{len(ma):,} of {len(md):,}")

    try:
        models = [
            ("Model 1", "all visited screens", "log elements (episode total)", "model-based",
             fit_negbin(md, "label_density", ["log_elements"])),
            ("Model 2", "all visited screens", "log elements (episode total), split",
             "clustered by app",
             fit_negbin(md, "label_density", ["log_elements"] + splits, md["app"])),
            ("Model 3", "first in-app screen", "log elements (same screen), split",
             "clustered by app",
             fit_negbin(ma, "app_density", ["log_app_elements"] + splits, ma["app"])),
        ]
        # Check C became Model 3 at the camera-ready. Check D is the Model 3
        # first specified, which kept the episode-total element count.
        checks = [
            ("Check A", "first screen", "log elements (same screen), split", "clustered by app",
             fit_negbin(m3, "first_density", ["log_first_elements"] + splits, m3["app"])),
            ("Check B", "first in-app screen", "log elements (episode total), split",
             "clustered by app",
             fit_negbin(ma, "app_density", ["log_elements"] + splits, ma["app"])),
            ("Check D", "first screen", "log elements (episode total), split",
             "clustered by app",
             fit_negbin(m3, "first_density", ["log_elements"] + splits, m3["app"])),
        ]
    except ImportError:
        con.fail("statsmodels is not installed.", hint="pip install statsmodels")

    if h42_irr is None or abs(models[0][4]["irr"] - h42_irr) > 1e-9:
        con.fail("Model 1 refit does not reproduce the H4.2 estimate.",
                 hint="The data or the H4.2 specification changed; do not report Models 2 and 3.")

    rep.section(
        "Models 2 and 3: more adjustment, and an exposure fixed before the demonstration",
        "Model 1 is the H4.2 model above. Model 2 adds AndroidControl's published "
        "split (train, validation, test; train is the reference) and clusters "
        "standard errors by app, the app being the dominant resource-id package "
        "of the episode (a11ydebt/apps.py), with episodes that have none as their "
        "own cluster. The element count in Models 1 and 2 is the total across the "
        "episode's screens, which grows with the number of screens and so with the "
        "outcome. Model 3 therefore takes both label density and element count "
        "from the episode's first in-app screen: observation 1 when the first "
        "action is open_app (observation 0 is then the screen the app was opened "
        "from), otherwise observation 0, the screen shown before the first action. "
        f"Episodes enter Model 3 only if that screen has at least {min_first} "
        "elements needing a label. Nothing in Model 3 depends on how the "
        "demonstration unfolds. The number of screens visited is not entered: "
        "every AndroidControl step carries one observation, so screens visited "
        "equals steps plus one and would reproduce the outcome. The specification "
        "checks change one choice at a time: Check A uses observation 0 with its "
        "own element count, Check B the first in-app screen with the episode-total "
        "count, and Check D observation 0 with the episode-total count. These are "
        "associations; none of them says that labelling changes how long a task "
        "takes."
    )
    rep.stat("Model episodes (at least one step)", int(len(md)))
    rep.stat("Episodes where screens visited equals steps plus one", screens_plus_one)
    rep.stat("Correlation of log episode-total elements with log steps", round(length_corr, 3))
    rep.stat("Episodes whose first action is open_app", int(md["opens_app"].sum()))
    rep.stat(f"Model 3 episodes (first in-app screen has >= {min_first} needs-label elements)",
             int(len(ma)))
    rep.stat("Model 3 episodes dropped", int(len(md) - len(ma)),
             note=f"First in-app screen has fewer than {min_first} elements needing a label.")
    rep.stat(f"Check A and D episodes (first screen has >= {min_first} needs-label elements)",
             int(len(m3)))
    for label, rows in (("models", models), ("checks", checks)):
        table_rows = []
        for name, measured, covs, se_kind, r in rows:
            table_rows.append([name, measured, covs, se_kind, f"{r['n']:,}",
                               f"{r['irr']:.3f}", f"[{r['lo']:.3f}, {r['hi']:.3f}]",
                               f"{r['p']:.4f}"])
            con.kv(f"{name}: density IRR",
                   f"{r['irr']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}], N = {r['n']:,}"
                   + ("" if r["converged"] else "  (DID NOT CONVERGE)"))
            if not r["converged"]:
                con.fail(f"{name} did not converge.", hint="Inspect the design matrix.")
            if name == "Model 1":
                continue
            rep.test(
                f"h4_2_{name.lower().replace(' ', '_')}",
                TestResult(
                    name=f"Negative binomial, {name}",
                    statistic_name="z", statistic=r["z"], p_value=r["p"], df=None,
                    effect_name="incidence rate ratio", effect_size=r["irr"],
                    ci_low=r["lo"], ci_high=r["hi"], n=r["n"],
                    groups={"density_on": measured, "covariates": covs,
                            "standard_errors": se_kind, "app_clusters": r["clusters"],
                            "alpha": r["alpha"]},
                    note=f"Density measured on {measured}; covariates: {covs}; "
                         f"standard errors {se_kind}.",
                ),
                hypothesis=f"{name}. Label density is associated with demonstration "
                           f"length under this specification.",
            )
        rep.table(Table(
            f"negbin_{label}",
            ["Model", "Density measured on", "Covariates", "SE", "N",
             "Density IRR", "95% CI", "p"],
            table_rows,
            caption=("Incidence rate ratio for label density in three negative "
                     "binomial models of demonstration length." if label == "models"
                     else "Specification checks: element count and first screen."),
            note="IRR below 1 means more description goes with fewer steps. "
                 "Associations only.",
        ))

    con.step(5, 6, "H4.3 — density across generalisation subsplits")
    rep.section("Label density across the benchmark's subsplits")
    subsplits_path = PATHS.androidcontrol_raw / "test_subsplits.json"
    if subsplits_path.exists():
        subs = json.loads(subsplits_path.read_text(encoding="utf-8"))
        rep.text(
            "AndroidControl's four test subsplits are designed to test different "
            "kinds of distribution shift. They are NOT a partition: they overlap "
            "substantially, so metrics must never be summed across them. If they "
            "also differ in label density, then reported per-split accuracy "
            "differences partly reflect accessibility rather than the shift each "
            "split was built to measure."
        )
        rows, table, labels = [], [], []
        for name, ids in subs.items():
            subset = ep[ep["episode_id"].isin(set(ids))]
            if not len(subset):
                continue
            k = int(subset["n_labelled"].sum())
            n = int(subset["n_needs_label"].sum())
            p = proportion(k, n)
            rows.append([name, f"{len(subset):,}", f"{n:,}", f"{p['pct']:.2f}",
                         f"[{p['ci_low_pct']:.2f}, {p['ci_high_pct']:.2f}]",
                         f"{subset['steps'].mean():.2f}"])
            if n >= 100:
                table.append([k, n - k])
                labels.append(name)
        if rows:
            rep.table(Table(
                "subsplit_density",
                ["Subsplit", "Episodes", "Needs label", "Coverage %", "95% CI",
                 "Mean steps"],
                rows,
                caption="Label coverage and demonstration length by test subsplit.",
                note="The subsplits overlap and are not a partition; episode counts "
                     "here therefore sum to more than the test split.",
            ))
            con.table(["Subsplit", "Episodes", "Coverage %", "Mean steps"],
                      [[r[0], r[1], r[3], r[5]] for r in rows])
        if len(table) >= 2:
            rep.test(
                "h4_3_subsplit_density",
                chi_square_test(table, row_labels=labels,
                                col_labels=["labelled", "unlabelled"]),
                hypothesis="H4.3. Label density differs across the benchmark's "
                           "generalisation subsplits.",
            )
    else:
        con.warn("test_subsplits.json not found; skipping H4.3")
        rep.note("H4.3 was skipped because test_subsplits.json was not downloaded.")

    con.step(6, 6, "Figure and report")
    use_paper_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=COLUMN)
    bins = np.linspace(0, 1, 11)
    ep["density_bin"] = pd.cut(ep["label_density"], bins=bins, include_lowest=True)
    grouped = ep.groupby("density_bin", observed=True)["steps"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 5]
    centres = [interval.mid for interval in grouped.index]
    ax.plot(centres, grouped["mean"], color=CATEGORICAL[0], linewidth=2,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("Label density (share of labellable elements labelled)")
    ax.set_ylabel("Mean demonstration steps")
    ax.set_xlim(0, 1)
    fig_paths = save(fig, outdir / "figures" / "rq4_density_vs_steps")
    plt.close(fig)
    rep.figure("rq4_density_vs_steps", fig_paths[0],
               "Mean demonstration length by label density decile. Bins with "
               "fewer than five episodes are omitted.")

    rep.limitation(
        "The outcome is demonstration length, not agent success. AndroidControl "
        "contains human demonstrations and no agent rollouts, so no claim about "
        "agent accuracy can be made from the episode-level models here."
    )
    rep.limitation(
        "Demonstration length is a coarse difficulty proxy. A long demonstration "
        "may reflect a complex goal rather than a hard interface, and label "
        "density is measured over the whole episode rather than at the moment of "
        "each decision."
    )
    rep.limitation(
        "Models 1 and 2 adjust for the log of the total number of elements across "
        "an episode's screens. That total grows with the number of screens "
        "visited, so it partly measures the outcome, and the association they "
        "show should not be read as a finding. Model 3 takes density and element "
        "count from one screen fixed before the demonstration and is the "
        "specification to report."
    )
    rep.limitation(
        "The subsplit analysis is ecological: it compares aggregate label density "
        "with aggregate published accuracy. An association at split level does "
        "not establish one at episode level."
    )
    rep.note("If agent rollouts become available, the same script takes an "
             "agent-success outcome by swapping the 'steps' column for a binary "
             "success column and the negative binomial for a logistic model.")

    written = rep.write(verbose_console=con)
    written["rq4_density_vs_steps.pdf"] = fig_paths[0]
    con.done(written)
    con.info("Next:  python phase6_synthesis/01_manuscript_numbers.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
