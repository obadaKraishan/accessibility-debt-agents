#!/usr/bin/env python3
"""
PHASE 4 / STEP 3 — Robustness checks requested by the AISSA 2026 reviewers.

Six checks, one report. Each answers a specific reviewer point, named in the
section headers below so the camera-ready revision can cite them directly.

    python phase4_validation/03_reviewer_checks.py

CHECKS
------
 1. Clustered uncertainty for the headline coverage figures and the
    cross-corpus gap. Elements are nested in screens and screens in apps or
    episodes, so element-level intervals are too narrow. Screen-level and
    app-level cluster bootstraps are reported beside the element-level ones.
    (Reviewer 1 weakness 2; Reviewer 2 "statistical dependence".)
 2. Clustered and paired versions of RQ3. The touched-versus-background
    comparison is recomputed with a screen-level bootstrap, a within-screen
    paired difference, and a screen-stratified permutation test.
    (Reviewer 2 "use a paired or clustered analysis for RQ3".)
 3. Hit-test bias toward larger targets. Compares the on-screen area of
    touched and background icon-like elements, reports how often the
    click was resolved by falling back to a clickable ancestor, and
    recomputes RQ3 without the touches the fallback produced.
    (Reviewer 1 weakness 4; Reviewer 2 recommended change on matching.)
 4. Construct validity: content description versus the accessible name.
    For AndroidControl, where the tree carries `hint_text`, recomputes
    coverage treating a non-empty hint as a label. This is the field a
    screen reader falls back to for an EditText, so it bears directly on
    the input-control finding. Rico's hierarchies carry no hint field.
    (Reviewer 2 primary concern.)
 5. Shard selection. Lists the AndroidControl shards used and their
    per-shard coverage, and compares the parsed episodes with the full
    release by split and by episode id, so a reader can see whether the
    omitted shards could plausibly differ.
    (Reviewer 2 "explain the shard selection".)
 6. Table IV denominator. Counts the RQ4 episodes per published split and
    per test subsplit, and confirms that only test episodes carry a
    subsplit label. (Reviewer 2 "clarify Table IV's sample size".)

TWO DATA FACTS THIS SCRIPT WORKS AROUND
---------------------------------------
* AndroidControl's app_name is set only on open_app actions, so app
  identity comes from resource-id packages (a11ydebt/apps.py).
* androidcontrol_steps.used_ancestor_fallback is `not resolved_clickable`,
  written after the fallback ran. Check 3 reconstructs the real fallback
  flag from node depths instead of reading that column.

Outputs
-------
phase4_validation/outputs/reports/reviewer_checks_apa_report.txt
phase4_validation/outputs/reports/reviewer_checks_results.json
phase4_validation/outputs/reports/reviewer_checks_results.csv
phase4_validation/outputs/tables/reviewer_checks_*.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import pyarrow.parquet as pq                                  # noqa: E402

from a11ydebt.apps import androidcontrol_episode_apps         # noqa: E402
from a11ydebt.cli import base_parser                          # noqa: E402
from a11ydebt.config import CFG, PATHS                        # noqa: E402
from a11ydebt.console import Console                          # noqa: E402
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.stats import (                                  # noqa: E402
    TestResult, cliffs_delta, cohens_h, interpret_delta, interpret_h,
    proportion, two_proportion_test,
)

# Resample counts come from config.yaml. At 2,000 permutations the RQ3
# 'all elements' p-value, which sits near .05, moved by more than its
# Monte Carlo error between two runs that differed only in random stream.
N_BOOT = int(CFG.run.n_bootstrap)
N_PERM = int(CFG.run.n_permutations)
RICO, AC = "rico_2017", "androidcontrol_2024"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def cluster_bootstrap_rate(k: np.ndarray, n: np.ndarray, rng: np.random.Generator,
                           n_boot: int = N_BOOT) -> np.ndarray:
    """Bootstrap a pooled rate sum(k)/sum(n) by resampling clusters."""
    m = len(k)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, m, m)
        kk, nn = k[idx].sum(), n[idx].sum()
        out[b] = kk / nn if nn else np.nan
    return out


def pct_ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def screen_counts(d: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Per-screen labelled and total counts for touched and background elements."""
    agg = (d.groupby(["screen_uid", "touch"])["lab"].agg(["sum", "count"])
            .unstack("touch").fillna(0))
    return (agg[("sum", True)].to_numpy(), agg[("count", True)].to_numpy(),
            agg[("sum", False)].to_numpy(), agg[("count", False)].to_numpy())


def stratified_permutation(kt, nt, kb, nb, rng: np.random.Generator,
                           n_perm: int = N_PERM) -> tuple[float, float]:
    """
    Screen-stratified permutation test of the pooled touched-minus-background
    difference. Returns (two-sided p, null centre).

    Shuffling the touched flag within each screen keeps every screen's
    touched count and labelled count fixed, so the number of labelled
    elements landing in the touched set is, screen by screen, a
    hypergeometric draw. Sampling those draws gives the same null
    distribution as shuffling element rows, in seconds rather than minutes.

    The null is not centred on zero. It is centred on the difference that
    screen composition alone produces (touched and background elements sit
    on different screens in different proportions), so the two-sided p
    measures distance from that centre, not from zero.
    """
    kt, nt, kb, nb = (np.asarray(a, dtype=np.int64) for a in (kt, nt, kb, nb))
    k, n = kt + kb, nt + nb
    mixed = (nt > 0) & (nb > 0)
    fixed = int(kt[~mixed].sum())       # all-touched screens give k; untouched give 0
    good, bad, size = k[mixed], (n - k)[mixed], nt[mixed]
    s_null = np.array([fixed + rng.hypergeometric(good, bad, size).sum()
                       for _ in range(n_perm)], dtype=float)
    s_centre = float(fixed + (size * good / (good + bad)).sum())
    s_obs = float(kt.sum())
    t_all, b_all, l_all = nt.sum(), nb.sum(), k.sum()
    centre = s_centre / t_all - (l_all - s_centre) / b_all
    extreme = np.abs(s_null - s_centre) >= abs(s_obs - s_centre) - 1e-9
    return float((extreme.sum() + 1) / (n_perm + 1)), float(centre)


def screen_bootstrap_diff(kt, nt, kb, nb, rng: np.random.Generator,
                          n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Pooled touched-minus-background rate, with a screen-resampling CI."""
    point = kt.sum() / nt.sum() - kb.sum() / nb.sum()
    m = len(kt)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, m, m)
        boot[b] = kt[idx].sum() / max(1, nt[idx].sum()) - kb[idx].sum() / max(1, nb[idx].sum())
    lo, hi = pct_ci(boot)
    return float(point), lo, hi


def clean(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    return out.mask(out.str.lower().isin({"", "none", "null", "nan"}), "")


# --------------------------------------------------------------------------

def main() -> int:
    parser = base_parser("Reviewer-requested robustness checks for the camera-ready.")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    con = Console("03_reviewer_checks", log_dir=PATHS.logs, quiet=args.quiet)
    con.banner("PHASE 4 / STEP 3", "Reviewer-requested robustness checks")

    here = Path(__file__).resolve().parent
    outdir = here / "outputs"
    elements_path = PATHS.processed / "elements.parquet"
    rq3_path = PATHS.processed / "rq3_touched.parquet"
    ac_nodes_path = PATHS.processed / "androidcontrol_nodes.parquet"
    ac_steps_path = PATHS.processed / "androidcontrol_steps.parquet"
    ac_eps_path = PATHS.processed / "androidcontrol_episodes.parquet"
    for p in (elements_path, rq3_path, ac_nodes_path, ac_steps_path, ac_eps_path):
        if not p.exists():
            con.fail(f"Missing {p}", hint="Run phase 2 first.")

    rep = ResultsReport(
        stem="reviewer_checks",
        title="Reviewer-requested robustness checks",
        subtitle="Clustered uncertainty, paired RQ3, hit-test bias, accessible-name "
                 "construct, shard selection, and split denominators",
        outdir=outdir,
        script="phase4_validation/03_reviewer_checks.py",
    )
    rep.provenance(elements_table=str(elements_path), nodes_table=str(ac_nodes_path),
                   seed=args.seed, n_boot=N_BOOT, n_perm=N_PERM,
                   system_packages=list(CFG.androidcontrol.system_packages))

    # ======================================================================
    con.step(1, 6, "Clustered uncertainty for coverage and the gap")
    # ======================================================================
    cols = ["dataset", "screen_uid", "app_package", "episode_id",
            "included", "needs_label", "label_status"]
    el = pq.read_table(elements_path, columns=cols).to_pandas()
    el = el[el["included"] & el["needs_label"]].copy()
    el["lab"] = (el["label_status"] == "labelled").astype(np.int32)
    el.drop(columns=["included", "needs_label", "label_status"], inplace=True)

    # AndroidControl has no app_package in the unified table, and its
    # app_name is set only on open_app actions (an empty string elsewhere),
    # so the app comes from resource-id packages. An episode with no usable
    # resource id becomes its own cluster rather than being dropped, so the
    # app-level bootstrap is centred on the same estimate as the others.
    steps = pq.read_table(ac_steps_path,
                          columns=["episode_id", "step_index", "app_name",
                                   "action_type", "x", "y", "resolved",
                                   "resolved_window", "resolved_node_uid",
                                   "resolved_clickable"]).to_pandas()
    apps = androidcontrol_episode_apps(ac_nodes_path, CFG.androidcontrol.system_packages)
    ep_app = apps.set_index("episode_id")["app"]
    is_ac = el["dataset"] == AC
    ac_ep = el.loc[is_ac, "episode_id"].astype("int64")
    el.loc[is_ac, "app_package"] = ac_ep.map(ep_app).fillna("episode:" + ac_ep.astype(str))

    ac_el = el.loc[is_ac, ["episode_id", "app_package"]]
    from_pkg = ~ac_el["app_package"].str.startswith("episode:")
    opened = (steps.loc[steps["action_type"] == "open_app"]
                   .drop_duplicates("episode_id").set_index("episode_id")["app_name"])
    named = pd.DataFrame({"name": opened}).join(ep_app, how="inner")
    modal = named.groupby("name")["app"].agg(lambda s: s.value_counts().index[0])
    agree = float((named["app"] == named["name"].map(modal)).mean())
    first_named = int((steps.sort_values(["episode_id", "step_index"])
                            .groupby("episode_id")["app_name"].first() != "").sum())
    n_ep_ac = int(ac_el["episode_id"].nunique())
    n_ep_pkg = int(ac_el.loc[from_pkg, "episode_id"].nunique())
    con.kv("AndroidControl episodes with an app from resource ids",
           f"{n_ep_pkg:,} of {n_ep_ac:,}")
    con.kv("Episodes whose first step names the app", f"{first_named:,}")
    con.kv("Agreement with open_app names where both exist", f"{100*agree:.2f}%")

    rep.section(
        "Check 1: clustered uncertainty",
        "Elements are nested within screens and screens within apps (Rico) or "
        "episodes and apps (AndroidControl). The element-level Wilson and "
        "Newcombe intervals in the main text treat elements as independent. "
        "Each figure is recomputed here with a cluster bootstrap that "
        f"resamples whole screens, and again resampling whole apps, {N_BOOT:,} "
        "times. Wider intervals are expected; the question is whether any "
        "conclusion changes. AndroidControl records the app only on open_app "
        "actions, so an episode's app is the dominant package among its "
        "resource ids, ignoring platform packages (status bar, keyboard, "
        "launcher). Episodes with no such resource id form their own cluster."
    )
    rep.stat("AndroidControl episodes with needs-label elements", n_ep_ac)
    rep.stat("AndroidControl episodes whose first step names the app", first_named)
    rep.stat("AndroidControl episodes with an app from resource ids", n_ep_pkg)
    rep.proportion("AndroidControl needs-label elements in an app cluster",
                   int(from_pkg.sum()), int(len(ac_el)))
    rep.stat("AndroidControl apps identified from resource ids",
             int(ac_el.loc[from_pkg, "app_package"].nunique()))
    rep.proportion("Episodes whose package matches the modal package for their open_app name",
                   int((named["app"] == named["name"].map(modal)).sum()), int(len(named)),
                   note="Agreement check on the episodes that also record an app name.")
    del ac_el, from_pkg
    rows = []
    draws: dict[str, dict[str, np.ndarray]] = {}
    for dataset, group in el.groupby("dataset"):
        k_el, n_el = int(group["lab"].sum()), int(len(group))
        p_el = proportion(k_el, n_el)
        draws[dataset] = {}
        for level, key in (("screen", "screen_uid"), ("app", "app_package")):
            g = group[group[key] != ""] if key == "app_package" else group
            agg = g.groupby(key)["lab"].agg(["sum", "count"])
            d = cluster_bootstrap_rate(agg["sum"].to_numpy(), agg["count"].to_numpy(), rng)
            draws[dataset][level] = d
            lo, hi = pct_ci(d)
            rows.append([dataset, level, f"{len(agg):,}", f"{100*k_el/n_el:.2f}",
                         f"[{100*lo:.2f}, {100*hi:.2f}]",
                         f"[{p_el['ci_low_pct']:.2f}, {p_el['ci_high_pct']:.2f}]"])
            con.kv(f"{dataset} {level}-level 95% CI",
                   f"[{100*lo:.2f}, {100*hi:.2f}]  ({len(agg):,} clusters)")
            rep.stat(f"Coverage, {dataset}, {level}-cluster bootstrap 95% CI",
                     f"{100*k_el/n_el:.2f}% [{100*lo:.2f}%, {100*hi:.2f}%], "
                     f"{len(agg):,} {level} clusters")
        rep.proportion(f"Coverage, {dataset} (element-level)", k_el, n_el)
    rep.table(Table(
        "clustered_coverage",
        ["Dataset", "Cluster", "Clusters", "Coverage %", "Cluster-bootstrap 95% CI",
         "Element-level Wilson 95% CI"],
        rows,
        caption="Label coverage with cluster-bootstrap intervals, beside the "
                "element-level intervals reported in the main text.",
    ))

    gap_rows = []
    for level in ("screen", "app"):
        gap = draws[AC][level] - draws[RICO][level]
        lo, hi = pct_ci(gap)
        point = el.loc[el.dataset == AC, "lab"].mean() - el.loc[el.dataset == RICO, "lab"].mean()
        gap_rows.append([level, f"{100*point:+.2f}", f"[{100*lo:.2f}, {100*hi:.2f}]"])
        con.kv(f"Gap, {level}-level 95% CI", f"[{100*lo:.2f}, {100*hi:.2f}] pp")
        rep.test(
            f"gap_cluster_{level}",
            TestResult(
                name=f"Cross-corpus gap, {level}-level cluster bootstrap",
                statistic_name="difference", statistic=float(point),
                p_value=float("nan"), df=None,
                effect_name="risk difference", effect_size=float(point),
                ci_low=lo, ci_high=hi, n=int(len(el)),
                effect2_name="Cohen's h",
                effect2_size=abs(cohens_h(el.loc[el.dataset == AC, "lab"].mean(),
                                          el.loc[el.dataset == RICO, "lab"].mean())),
                note=f"Percentile CI from {N_BOOT:,} resamples of whole {level}s "
                     f"within each corpus, drawn independently.",
            ),
            hypothesis="The cross-corpus coverage gap survives clustering.",
        )
    rep.table(Table("clustered_gap", ["Cluster", "Gap (pp)", "95% CI"], gap_rows,
                    caption="The cross-corpus gap under cluster bootstrapping."))
    del el

    # ======================================================================
    con.step(2, 6, "Clustered and paired RQ3")
    # ======================================================================
    rq3 = pq.read_table(rq3_path, columns=[
        "episode_id", "step_index", "window_index", "node_uid",
        "screen_uid", "was_touched", "needs_label", "label_status",
        "clickable", "icon_like", "role",
        "x1", "y1", "x2", "y2"]).to_pandas()
    rq3["lab"] = (rq3["label_status"] == "labelled").astype(np.int32)
    rq3["touch"] = rq3["was_touched"].astype(bool)
    # The same three populations as run_rq3.py, including its use of the
    # clickable flag alone, so each pooled point estimate below equals the
    # RQ3 row it re-analyses.
    pops = {
        "all requiring a label": rq3["needs_label"].astype(bool),
        "clickable only": rq3["needs_label"].astype(bool) & rq3["clickable"].astype(bool),
        "icon-like only": rq3["icon_like"].astype(bool) & rq3["needs_label"].astype(bool),
    }
    rep.section(
        "Check 2: RQ3 with the screen structure respected",
        "Three re-analyses of the touched-versus-background comparison. "
        "(a) A screen-level cluster bootstrap of the pooled difference. "
        "(b) A paired comparison on screens that contain both a touched and a "
        "background element in the population: the mean of per-screen "
        "differences, with a screen bootstrap. (c) A stratified permutation "
        "test that shuffles the touched flag only within each screen, so the "
        "null preserves which screens were visited and how many elements each "
        "contributed. That null is centred on the difference screen "
        "composition alone would produce, not on zero, because touched and "
        "background elements are spread over the screens in different "
        "proportions. The centre is reported, and the pooled difference splits "
        "into that composition part and a within-screen part; p measures the "
        "within-screen part."
    )
    rq3_rows = []
    for name, mask in pops.items():
        d = rq3[mask]
        kt, nt, kb, nb = screen_counts(d)

        # (a) screen-level cluster bootstrap of the pooled difference
        point, lo_a, hi_a = screen_bootstrap_diff(kt, nt, kb, nb, rng)

        # (b) paired within-screen difference
        both = (nt > 0) & (nb > 0)
        pdiff = kt[both] / nt[both] - kb[both] / nb[both]
        pboot = np.array([pdiff[rng.integers(0, len(pdiff), len(pdiff))].mean()
                          for _ in range(N_BOOT)])
        lo_b, hi_b = pct_ci(pboot)

        # (c) screen-stratified permutation of the touched flag
        p_perm, centre = stratified_permutation(kt, nt, kb, nb, rng)

        rq3_rows.append([name, f"{len(kt):,}", f"{int(both.sum()):,}",
                         f"{100*point:+.2f}", f"[{100*lo_a:.2f}, {100*hi_a:.2f}]",
                         f"{100*pdiff.mean():+.2f}", f"[{100*lo_b:.2f}, {100*hi_b:.2f}]",
                         f"{100*centre:+.2f}", f"{p_perm:.4f}"])
        con.kv(name, f"pooled {100*point:+.2f} pp [{100*lo_a:.2f}, {100*hi_a:.2f}]; "
                     f"paired {100*pdiff.mean():+.2f} pp [{100*lo_b:.2f}, {100*hi_b:.2f}]; "
                     f"composition alone {100*centre:+.2f} pp; "
                     f"stratified perm p = {p_perm:.4f}")
        rep.test(
            f"rq3_clustered_{name.split()[0]}",
            TestResult(
                name=f"RQ3 {name}: screen-level cluster bootstrap",
                statistic_name="difference", statistic=float(point),
                p_value=p_perm, df=None,
                effect_name="risk difference", effect_size=float(point),
                ci_low=lo_a, ci_high=hi_a, n=int(len(d)),
                groups={"screens": int(len(kt)), "paired_screens": int(both.sum()),
                        "paired_mean_diff": float(pdiff.mean()),
                        "paired_ci": [lo_b, hi_b],
                        "composition_only_diff": centre},
                note=f"CI from {N_BOOT:,} screen resamples; p from {N_PERM:,} "
                     f"within-screen permutations of the touched flag, two-sided "
                     f"around the null centre, (r+1)/(m+1) convention.",
            ),
            hypothesis="Touched elements are labelled at a different rate from "
                       "background elements on the same screens.",
        )
        rep.stat(f"RQ3 {name}: paired within-screen difference",
                 f"{100*pdiff.mean():+.2f} pp [{100*lo_b:.2f}, {100*hi_b:.2f}], "
                 f"{int(both.sum()):,} of {len(kt):,} screens",
                 note="Mean of per-screen touched-minus-background rates on screens "
                      "with both; CI from a screen bootstrap.")
        rep.stat(f"RQ3 {name}: difference expected from screen composition alone",
                 f"{100*centre:+.2f} pp",
                 note="The centre of the stratified null: the pooled difference "
                      "if touched elements were labelled at their own screen's "
                      "rate. Observed minus this is the within-screen part.")
        rep.stat(f"RQ3 {name}: within-screen part of the pooled difference",
                 f"{100*(point - centre):+.2f} pp")
    rep.table(Table(
        "rq3_clustered",
        ["Population", "Screens", "Paired screens", "Pooled diff (pp)",
         "Screen-bootstrap 95% CI", "Paired diff (pp)", "Paired 95% CI",
         "Composition alone (pp)", "Stratified perm. p"],
        rq3_rows,
        caption="RQ3 recomputed with screen-level clustering, within-screen "
                "pairing, and a screen-stratified permutation test.",
    ))

    # ======================================================================
    con.step(3, 6, "Hit-test bias toward larger targets")
    # ======================================================================
    # (a) Which resolved clicks used the clickable-ancestor fallback?
    # androidcontrol_steps.used_ancestor_fallback cannot say: the parser set
    # it to `not node.is_clickable` on the node resolve_click returned, after
    # the fallback had run, so it marks clicks that ended on a non-clickable
    # element and never marks a successful fallback. The fallback always
    # climbs to a strictly shallower ancestor and does not check visibility,
    # so a click used it exactly when its resolved node is not the deepest
    # candidate: shallower than the deepest visible, positive-area node that
    # contains the click in the top window, or not a candidate at all.
    # Clicks and long presses, as in the parser's hit rate.
    clicks = steps[steps["action_type"].isin(["click", "long_press"])]
    res = clicks[clicks["resolved"]]
    n_clicks, n_res = int(len(clicks)), int(len(res))
    hit = pq.read_table(ac_nodes_path, columns=[
        "episode_id", "step_index", "window_index", "window_layer", "node_uid",
        "depth", "visible_to_user", "x1", "y1", "x2", "y2"]).to_pandas()
    hit = hit.merge(res[["episode_id", "step_index", "x", "y",
                         "resolved_window", "resolved_node_uid"]],
                    on=["episode_id", "step_index"])
    hit = hit[hit["visible_to_user"] & (hit["x2"] > hit["x1"]) & (hit["y2"] > hit["y1"])
              & (hit["x1"] <= hit["x"]) & (hit["x"] <= hit["x2"])
              & (hit["y1"] <= hit["y"]) & (hit["y"] <= hit["y2"])]
    top = hit.groupby(["episode_id", "step_index"])["window_layer"].transform("max")
    hit = hit[hit["window_layer"] == top]
    deepest = hit.groupby(["episode_id", "step_index"])["depth"].max().rename("deepest")
    chosen = (hit[(hit["window_index"] == hit["resolved_window"])
                  & (hit["node_uid"] == hit["resolved_node_uid"])]
              .set_index(["episode_id", "step_index"])["depth"].rename("chosen"))
    del hit, top
    res = res.set_index(["episode_id", "step_index"]).join(deepest).join(chosen)
    res["fallback"] = res["chosen"].isna() | (res["chosen"] < res["deepest"])
    n_fb = int(res["fallback"].sum())
    n_not_cand = int(res["chosen"].isna().sum())
    n_no_cand = int(res["deepest"].isna().sum())
    n_nonclick = int((~res["resolved_clickable"].astype(bool)).sum())
    n_fb_nonclick = int((res["fallback"] & ~res["resolved_clickable"].astype(bool)).sum())
    con.kv("Clicks and long presses resolved", f"{n_res:,} of {n_clicks:,}")
    con.kv("Resolved via clickable-ancestor fallback",
           f"{n_fb:,} ({100*n_fb/n_res:.2f}%), of which {n_not_cand:,} "
           f"resolved to an ancestor that is not itself a candidate")
    con.kv("Resolved to a non-clickable element", f"{n_nonclick:,} ({100*n_nonclick/n_res:.2f}%)")
    if n_no_cand or n_fb_nonclick:
        con.fail(f"{n_no_cand} resolved clicks have no candidate node and "
                 f"{n_fb_nonclick} fallbacks ended on a non-clickable node; the "
                 f"reconstruction does not match the parser's rule.",
                 hint="Compare src/a11ydebt/tfrecord.py:resolve_click with the "
                      "filter above before trusting any fallback figure.")

    # (b) Are touched icon-like elements larger than background ones?
    icon = rq3[pops["icon-like only"]].copy()
    icon["area"] = ((icon["x2"] - icon["x1"]).clip(lower=0)
                    * (icon["y2"] - icon["y1"]).clip(lower=0)).astype(float)
    a_t = np.log1p(icon.loc[icon["touch"], "area"].to_numpy())
    a_b = np.log1p(icon.loc[~icon["touch"], "area"].to_numpy())
    delta = cliffs_delta(a_t, a_b)
    # Screen-level bootstrap of delta: resample whole screens, keep their elements.
    icon = icon.sort_values("screen_uid", kind="stable")
    s = icon["screen_uid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    lens = np.diff(np.r_[starts, len(s)])
    logarea, touched = np.log1p(icon["area"].to_numpy()), icon["touch"].to_numpy()
    dboot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(starts), len(starts))
        ln, st = lens[pick], starts[pick]
        idx = np.repeat(st - np.r_[0, np.cumsum(ln)[:-1]], ln) + np.arange(ln.sum())
        dboot[b] = cliffs_delta(logarea[idx][touched[idx]], logarea[idx][~touched[idx]])
    d_lo, d_hi = pct_ci(dboot)
    med_t = float(np.median(icon.loc[icon["touch"], "area"]))
    med_b = float(np.median(icon.loc[~icon["touch"], "area"]))
    con.kv("Median area, touched icon-like", f"{med_t:,.0f} px^2")
    con.kv("Median area, background icon-like", f"{med_b:,.0f} px^2")
    con.kv("Cliff's delta (log area, touched vs background)",
           f"{delta:.3f} [{d_lo:.3f}, {d_hi:.3f}] ({interpret_delta(delta)})")

    rep.section(
        "Check 3: does hit testing favour large targets?",
        "The hit-test rule takes the deepest visible node containing the click, "
        "breaking ties on drawing order and then toward the smaller area, so "
        "its first choice is the most specific target. When that node is not "
        "clickable, the rule climbs to the nearest clickable ancestor, which "
        "is larger by construction. Three quantities bear on the resulting "
        "bias: how often the fallback fired, whether touched icon-like "
        "elements are larger than background ones, and whether RQ3 holds "
        "once fallback touches are removed. The steps table's "
        "used_ancestor_fallback column records whether the final element was "
        "non-clickable, not whether the fallback fired, so the fallback is "
        "reconstructed here from node depths: a click used it exactly when "
        "its resolved node is not the deepest visible node containing the "
        "click in the top window."
    )
    rep.proportion("Clicks and long presses resolved to any element", n_res, n_clicks)
    rep.proportion("Resolved clicks that used the clickable-ancestor fallback", n_fb, n_res)
    rep.proportion("Resolved clicks ending on a non-clickable element", n_nonclick, n_res,
                   note="No clickable ancestor existed; this is what the steps "
                        "table's used_ancestor_fallback column records.")
    rep.stat("Median area, touched icon-like elements", med_t, unit="px^2")
    rep.stat("Median area, background icon-like elements", med_b, unit="px^2")
    rep.test(
        "hit_test_area",
        TestResult(name="Area of touched vs background icon-like elements",
                   statistic_name="Cliff's delta", statistic=float(delta),
                   p_value=float("nan"), df=None,
                   effect_name="Cliff's delta", effect_size=float(delta),
                   ci_low=d_lo, ci_high=d_hi,
                   n=int(len(icon)),
                   note=f"Computed on log(1 + area); CI from {N_BOOT:,} screen "
                        f"resamples. Positive means touched elements are larger."),
        hypothesis="Touched elements differ in size from background elements.",
        interpretation=interpret_delta(delta),
    )

    # (c) RQ3 without the touches the fallback produced. Those touched
    # elements are dropped; the background is unchanged. The comparison is
    # therefore lopsided: fallback targets are mostly clickable wrappers of
    # role 'other', and the background keeps every wrapper of that kind. The
    # label rates of fallback and direct touches are the cleaner diagnostic:
    # if fallback touches are labelled less often, the fallback pulls the
    # touched rate down and cannot have produced a positive difference.
    key = ["episode_id", "step_index", "window_index", "node_uid"]
    fb = (res.loc[res["fallback"]].reset_index()
             [["episode_id", "step_index", "resolved_window", "resolved_node_uid"]]
             .set_axis(key, axis=1).assign(via_fallback=True))
    flagged = rq3[key].merge(fb, on=key, how="left")["via_fallback"].notna()
    rq3["via_fallback"] = flagged.to_numpy() & rq3["touch"].to_numpy()
    fb_rows = []
    for name, mask in pops.items():
        n_touch = int((mask & rq3["touch"]).sum())
        n_drop = int((mask & rq3["via_fallback"]).sum())
        direct = mask & rq3["touch"] & ~rq3["via_fallback"]
        via = mask & rq3["via_fallback"]
        k_dir, n_dir = int(rq3.loc[direct, "lab"].sum()), int(direct.sum())
        k_via, n_via = int(rq3.loc[via, "lab"].sum()), int(via.sum())
        kt, nt, kb, nb = screen_counts(rq3[mask & ~rq3["via_fallback"]])
        point, lo, hi = screen_bootstrap_diff(kt, nt, kb, nb, rng)
        fb_rows.append([name, f"{n_touch:,}", f"{n_drop:,}", f"{100*n_drop/n_touch:.2f}",
                        f"{100*k_dir/n_dir:.2f}",
                        f"{100*k_via/n_via:.2f}" if n_via else "",
                        f"{100*point:+.2f}", f"[{100*lo:.2f}, {100*hi:.2f}]"])
        con.kv(f"{name}, fallback touches",
               f"{n_drop:,} of {n_touch:,}; labelled {100*k_via/max(1, n_via):.2f}% "
               f"vs {100*k_dir/n_dir:.2f}% direct; removed: {100*point:+.2f} pp "
               f"[{100*lo:.2f}, {100*hi:.2f}]")
        rep.proportion(f"RQ3 {name}: touched elements reached via fallback", n_drop, n_touch)
        rep.proportion(f"RQ3 {name}: label rate, touches resolved directly", k_dir, n_dir)
        if n_via:
            rep.proportion(f"RQ3 {name}: label rate, touches via fallback", k_via, n_via)
        rep.test(
            f"rq3_no_fallback_{name.split()[0]}",
            TestResult(
                name=f"RQ3 {name}, fallback touches removed: screen-level cluster bootstrap",
                statistic_name="difference", statistic=float(point),
                p_value=float("nan"), df=None,
                effect_name="risk difference", effect_size=float(point),
                ci_low=lo, ci_high=hi, n=int(nt.sum() + nb.sum()),
                groups={"touched_kept": int(nt.sum()), "touched_dropped": n_drop},
                note=f"Touched elements reached through the clickable-ancestor "
                     f"fallback are removed; background unchanged, so elements "
                     f"like the removed ones stay in the background. Read as a "
                     f"bound on direction, not as a cleaner estimate. CI from "
                     f"{N_BOOT:,} screen resamples.",
            ),
            hypothesis="The RQ3 difference does not depend on the fallback.",
        )
    rep.table(Table(
        "rq3_no_fallback",
        ["Population", "Touched", "Via fallback", "Via fallback %",
         "Labelled, direct %", "Labelled, via fallback %",
         "Diff without them (pp)", "Screen-bootstrap 95% CI"],
        fb_rows,
        caption="RQ3 with touches resolved through the clickable-ancestor fallback removed.",
        note="The background is not filtered in the same way, so the last two "
             "columns are lopsided by design; the label-rate columns are the "
             "direct evidence on whether the fallback inflates the result.",
    ))
    del rq3, icon, res

    # ======================================================================
    con.step(4, 6, "Construct validity: hint text as an accessible name")
    # ======================================================================
    nodes = pq.read_table(ac_nodes_path, columns=[
        "included", "needs_label", "label_status", "role",
        "content_desc", "hint_text", "episode_id"]).to_pandas()
    nodes = nodes[nodes["included"] & nodes["needs_label"]].copy()
    nodes["desc"] = clean(nodes["content_desc"]) != ""
    nodes["hint"] = clean(nodes["hint_text"]) != ""
    nodes["lab_f"] = nodes["desc"] | nodes["hint"]
    rep.section(
        "Check 4: content description versus the accessible name",
        "A screen reader's accessible name can come from fields other than the "
        "content description. On Android the most common is the hint text of "
        "an input field. AndroidControl's accessibility forest records "
        "hint_text; Rico's view hierarchies do not. Variant F below recomputes "
        "AndroidControl coverage treating a non-empty hint as a label. The "
        "primary measure, content-description coverage, remains the paper's "
        "construct; this check bounds how far the accessible-name reading "
        "differs from it."
    )
    hint_rows = []
    for role, g in [("all", nodes)] + list(nodes.groupby("role")):
        k0, k1, n = int(g["desc"].sum()), int(g["lab_f"].sum()), int(len(g))
        only_hint = int((g["hint"] & ~g["desc"]).sum())
        hint_rows.append([role, f"{n:,}", f"{100*k0/n:.2f}", f"{100*k1/n:.2f}",
                          f"{100*(k1-k0)/n:+.2f}", f"{only_hint:,}"])
        con.kv(f"{role}: desc-only -> desc-or-hint", f"{100*k0/n:.2f}% -> {100*k1/n:.2f}%  (+{only_hint:,} via hint)")
        rep.proportion(f"AndroidControl coverage, content description only ({role})", k0, n)
        rep.proportion(f"AndroidControl coverage, description or hint ({role})", k1, n)
    rep.table(Table(
        "hint_variant",
        ["Role", "Needs label", "Desc only %", "Desc or hint %", "Change (pp)", "Hint only (n)"],
        hint_rows,
        caption="AndroidControl coverage when a non-empty hint_text counts as a label.",
        note="Rico carries no hint field, so the cross-corpus gap under this "
             "variant uses the primary Rico figure.",
    ))
    k_f, n_f = int(nodes["lab_f"].sum()), int(len(nodes))
    rep.stat("Variant F AndroidControl coverage (description or hint)", round(100*k_f/n_f, 2), unit="%")
    del nodes

    # ======================================================================
    con.step(5, 6, "Shard selection and per-shard coverage")
    # ======================================================================
    eps = pd.read_parquet(ac_eps_path, columns=[
        "episode_id", "split", "shard", "n_needs_label", "n_labelled", "n_actions"])
    parse_report_path = PATHS.interim / "androidcontrol_parse_report.json"
    splits_path = PATHS.androidcontrol_raw / "splits.json"
    subsplits_path = PATHS.androidcontrol_raw / "test_subsplits.json"
    for p in (parse_report_path, splits_path, subsplits_path):
        if not p.exists():
            con.fail(f"Missing {p}",
                     hint="These are written by phase 1 and phase 2 and were kept "
                          "when the raw shards were deleted. Restore them from backup.")
    parse_report = json.loads(parse_report_path.read_text(encoding="utf-8"))
    released = json.loads(splits_path.read_text(encoding="utf-8"))

    # Per-corpus parse totals, so the paper's "elements parsed" row traces to
    # MN. They must add up to the unified elements table.
    rico_report_path = PATHS.interim / "rico_parse_report.json"
    if not rico_report_path.exists():
        con.fail(f"Missing {rico_report_path}",
                 hint="Written by phase2_processing/01_parse_rico.py; restore it from backup.")
    rico_parsed = int(json.loads(rico_report_path.read_text(encoding="utf-8"))["totals"]["nodes"])
    ac_parsed = int(parse_report["totals"]["elements"])
    n_elements = pq.ParquetFile(elements_path).metadata.num_rows
    if rico_parsed + ac_parsed != n_elements:
        con.fail(f"Parse totals {rico_parsed:,} + {ac_parsed:,} do not equal the "
                 f"{n_elements:,} rows of elements.parquet.",
                 hint="The parse reports and the analysis tables are out of step.")
    shard_rows, shard_cov = [], []
    for shard, g in eps.groupby("shard"):
        k, n = int(g["n_labelled"].sum()), int(g["n_needs_label"].sum())
        shard_cov.append(100 * k / n)
        shard_rows.append([shard, f"{len(g):,}", f"{n:,}", f"{100*k/n:.2f}"])
        con.kv(shard, f"{len(g):,} episodes, coverage {100*k/n:.2f}%")
    cov = np.array(shard_cov)

    # Is the parsed half a content-neutral sample of the release? Compare it
    # with the published splits, and check it spans the full episode-id range.
    parsed = set(eps["episode_id"].astype(int))
    all_ids = np.array(sorted({int(i) for ids in released.values() for i in ids}))
    deciles = pd.qcut(all_ids, 10, labels=False)
    decile_share = pd.Series(np.isin(all_ids, list(parsed))).groupby(deciles).mean()
    split_rows = []
    for name in ("train", "validation", "test"):
        ids = {int(i) for i in released[name]}
        split_rows.append([name, f"{len(ids & parsed):,}", f"{len(ids):,}",
                           f"{100*len(ids & parsed)/len(ids):.2f}"])
        con.kv(f"{name}: parsed of released", f"{len(ids & parsed):,} of {len(ids):,}")
    con.kv("Share of each episode-id decile parsed",
           f"{100*decile_share.min():.2f}% to {100*decile_share.max():.2f}%")

    rep.section(
        "Check 5: which AndroidControl shards were used",
        "Phase 2 parsed every complete shard on disk, in filename order. The "
        "parse report lists the shards used and any partial download it "
        "skipped. The release interleaves episode ids across shards, so a "
        "shard is not a selection by app, split or date. The tables below "
        "show per-shard coverage, and compare the parsed episodes with the "
        "published splits and with the full episode-id range."
    )
    rep.stat("AndroidControl shards used", ", ".join(parse_report["shards_used"]))
    rep.stat("AndroidControl shards skipped as incomplete",
             ", ".join(parse_report.get("shards_skipped_incomplete", [])) or "none")
    rep.stat("AndroidControl shards used, count", len(parse_report["shards_used"]))
    rep.stat("Elements parsed, rico_2017", rico_parsed,
             note="data/interim/rico_parse_report.json, totals.nodes")
    rep.stat("Elements parsed, androidcontrol_2024", ac_parsed,
             note="data/interim/androidcontrol_parse_report.json, totals.elements")
    rep.table(Table("shards", ["Shard", "Episodes", "Needs label", "Coverage %"],
                    shard_rows, caption="AndroidControl shards used, with per-shard coverage."))
    rep.stat("Per-shard coverage, min", round(float(cov.min()), 2), unit="%")
    rep.stat("Per-shard coverage, max", round(float(cov.max()), 2), unit="%")
    rep.stat("Per-shard coverage, SD", round(float(cov.std(ddof=1)), 3), unit="pp")
    rep.table(Table("parsed_vs_released",
                    ["Split", "Parsed episodes", "Released episodes", "Parsed %"],
                    split_rows,
                    caption="Parsed AndroidControl episodes against the published splits."))
    for name, got, total, pct in split_rows:
        rep.stat(f"Parsed share of released {name} episodes",
                 f"{pct}% ({got} of {total})")
    rep.stat("Parsed share of each episode-id decile, min",
             round(100 * float(decile_share.min()), 2), unit="%")
    rep.stat("Parsed share of each episode-id decile, max",
             round(100 * float(decile_share.max()), 2), unit="%")

    # ======================================================================
    con.step(6, 6, "Table IV denominator")
    # ======================================================================
    # Table IV comes from run_rq4.py, which keeps episodes with at least one
    # element needing a label; its model then drops episodes with no steps.
    rq4_eps = eps[eps["n_needs_label"] > 0]
    n_model = int((rq4_eps["n_actions"] > 0).sum())
    subs = json.loads(subsplits_path.read_text(encoding="utf-8"))
    test_ids = {int(i) for i in released["test"]}
    in_rq4 = set(rq4_eps["episode_id"].astype(int))
    rows6, memberships, union = [], 0, set()
    for name, ids in subs.items():
        ids = {int(i) for i in ids}
        if not ids <= test_ids:
            con.fail(f"Subsplit {name} contains non-test episodes.",
                     hint="Check test_subsplits.json against splits.json.")
        got = ids & in_rq4
        memberships += len(got)
        union |= got
        rows6.append([name, f"{len(got):,}"])
        con.kv(f"subsplit {name}", f"{len(got):,}")
    split_counts = rq4_eps["split"].value_counts()
    for s, c in split_counts.items():
        con.kv(f"split = {s}", f"{c:,}")
    con.kv("Distinct test episodes in any subsplit", f"{len(union):,}")
    con.kv("Subsplit memberships (Table IV row total)", f"{memberships:,}")
    rep.section(
        "Check 6: why Table IV has fewer episodes than the RQ4 model",
        "AndroidControl publishes its generalisation subsplits (IDD, unseen "
        "category, unseen app, unseen task) only for test episodes; training "
        "and validation episodes carry no subsplit label. Table IV can "
        "therefore use only the test episodes, while the RQ4 regression uses "
        "every episode. The subsplits also overlap, so Table IV's rows sum to "
        "more than the number of distinct test episodes."
    )
    rep.stat("RQ4 episodes (at least one element needing a label)", int(len(rq4_eps)))
    rep.stat("RQ4 model episodes (at least one step)", n_model)
    for s, c in split_counts.items():
        rep.stat(f"RQ4 episodes in published split '{s}'", int(c))
    for name, got in rows6:
        rep.stat(f"RQ4 test episodes in subsplit '{name}'", got)
    rep.stat("Distinct RQ4 test episodes in any subsplit", len(union))
    rep.stat("Sum of subsplit memberships (Table IV rows)", memberships)
    rep.table(Table("splits", ["Split", "Episodes"],
                    [[s, f"{c:,}"] for s, c in split_counts.items()],
                    caption="RQ4 episodes per published split."))
    rep.table(Table("subsplits", ["Subsplit", "Episodes"], rows6,
                    caption="RQ4 test episodes per subsplit; the subsplits overlap.",
                    note=f"{memberships:,} memberships over {len(union):,} distinct "
                         f"test episodes."))

    rep.limitation(
        "Variant F treats any non-empty hint as an accessible name. Hints can "
        "be placeholders ('Enter text'), so this is an upper bound on what a "
        "screen reader would usefully announce, as the primary measure is a "
        "lower bound."
    )
    rep.limitation(
        "The fallback flag is reconstructed from node depths, not read from the "
        "parse, because the parser's used_ancestor_fallback column records a "
        "non-clickable result instead. The reconstruction applies the parser's "
        "own rule to the stored nodes; the script stops if any resolved click "
        "contradicts it."
    )
    rep.limitation(
        "The fallback-free RQ3 rows drop fallback touches from the touched set "
        "only. The background keeps elements of the same kind, mostly clickable "
        "wrappers of role 'other', so those rows bound the direction of any "
        "fallback bias rather than estimate a bias-free difference."
    )
    rep.limitation(
        "AndroidControl app identity comes from resource-id packages. An app "
        "whose views carry no resource ids, or one that spans several packages, "
        "can be split across clusters, which would narrow the app-level "
        "interval slightly."
    )
    written = rep.write(verbose_console=con)
    con.done(written)
    con.info("Next:  python phase6_synthesis/01_manuscript_numbers.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
