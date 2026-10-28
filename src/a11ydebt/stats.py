"""
Statistics with effect sizes and confidence intervals, formatted for APA 7.

Every test in this project returns a `TestResult`, which carries the test
statistic, degrees of freedom, exact p, an effect size, and a confidence
interval for that effect size. Nothing returns a bare p-value, because a
bare p-value cannot be written into a results section.

The formatting helpers follow APA 7 conventions:
  * p is reported to three decimals with no leading zero; below .001 it is
    reported as "p < .001", never as "p = .000".
  * Effect sizes carry a 95% CI.
  * Exact ns are always reported alongside proportions.

IEEE conventions differ mainly in typography rather than substance, so the
report writer emits both an APA narrative and a compact numeric table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    raise ImportError("scipy is required. Run: pip install -r requirements.txt")

__all__ = [
    "TestResult", "fmt_p", "fmt_stat", "fmt_ci", "fmt_pct",
    "wilson_ci", "proportion", "two_proportion_test", "cohens_h",
    "chi_square_test", "cramers_v", "permutation_test",
    "bootstrap_ci", "cliffs_delta", "odds_ratio", "fdr_bh",
    "interpret_h", "interpret_v", "interpret_delta", "interpret_or",
]

DEFAULT_CONF = 0.95
RNG_SEED = 20260908


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------

@dataclass
class TestResult:
    """One statistical test, fully reportable."""

    name: str                       # e.g. "Two-proportion z test"
    statistic_name: str             # "z", "chi2", "OR", "delta"
    statistic: float
    p_value: float
    effect_name: str                # "Cohen's h", "Cramer's V", ...
    effect_size: float
    ci_low: float
    ci_high: float
    n: int
    df: float | None = None
    conf: float = DEFAULT_CONF
    groups: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    # A second effect size reported without a CI. Used where the standard
    # interval belongs to a different quantity than the standardised effect
    # (e.g. a Newcombe CI on the risk difference alongside Cohen's h).
    effect2_name: str = ""
    effect2_size: float | None = None

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def apa(self) -> str:
        """One APA-7 sentence fragment, ready to paste into a results section."""
        parts = [f"{self.statistic_name} "]
        if self.df is not None:
            df = int(self.df) if float(self.df).is_integer() else round(self.df, 2)
            parts.append(f"({df}) ")
        parts.append(f"= {fmt_stat(self.statistic)}, {fmt_p(self.p_value)}, ")
        parts.append(
            f"{self.effect_name} = {fmt_stat(self.effect_size)}, "
            f"{int(self.conf * 100)}% CI {fmt_ci(self.ci_low, self.ci_high)}"
        )
        if self.effect2_name and self.effect2_size is not None:
            parts.append(f", {self.effect2_name} = {fmt_stat(self.effect2_size)}")
        parts.append(f", N = {self.n:,}")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["apa"] = self.apa
        d["significant"] = self.significant
        return d


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    """APA 7: 'p < .001' or 'p = .023'."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".", 1)


def fmt_stat(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.{digits}f}"


def fmt_ci(low: float, high: float, digits: int = 3) -> str:
    return f"[{fmt_stat(low, digits)}, {fmt_stat(high, digits)}]"


def fmt_pct(k: int, n: int, digits: int = 2) -> str:
    """'12.50% (125/1,000)'."""
    if n == 0:
        return f"n/a (0/0)"
    return f"{100.0 * k / n:.{digits}f}% ({k:,}/{n:,})"


# --------------------------------------------------------------------------
# proportions
# --------------------------------------------------------------------------

def wilson_ci(k: int, n: int, conf: float = DEFAULT_CONF) -> tuple[float, float]:
    """Wilson score interval. Correct at extreme proportions, unlike Wald."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(k: int, n: int, conf: float = DEFAULT_CONF) -> dict[str, Any]:
    """A proportion with its Wilson CI, ready for a table row."""
    lo, hi = wilson_ci(k, n, conf)
    return {
        "k": int(k), "n": int(n),
        "proportion": (k / n) if n else float("nan"),
        "pct": (100.0 * k / n) if n else float("nan"),
        "ci_low": lo, "ci_high": hi,
        "ci_low_pct": lo * 100, "ci_high_pct": hi * 100,
        "formatted": fmt_pct(k, n),
    }


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for the difference between two proportions."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def _newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, conf: float) -> tuple[float, float]:
    """Newcombe's hybrid score interval for a difference of proportions."""
    l1, u1 = wilson_ci(k1, n1, conf)
    l2, u2 = wilson_ci(k2, n2, conf)
    p1 = k1 / n1 if n1 else float("nan")
    p2 = k2 / n2 if n2 else float("nan")
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def two_proportion_test(
    k1: int, n1: int, k2: int, n2: int,
    label1: str = "group 1", label2: str = "group 2",
    conf: float = DEFAULT_CONF,
) -> TestResult:
    """
    Pooled two-proportion z test with Cohen's h and a Newcombe CI on the
    difference. This is the workhorse for "is coverage higher in A than B".
    """
    if n1 == 0 or n2 == 0:
        raise ValueError("two_proportion_test needs non-empty groups")
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - sps.norm.cdf(abs(z)))
    lo, hi = _newcombe_diff_ci(k1, n1, k2, n2, conf)
    return TestResult(
        name="Two-proportion z test",
        statistic_name="z", statistic=z, p_value=p, df=None,
        # The reported CI is Newcombe's on the DIFFERENCE, so the difference is
        # the primary effect and Cohen's h is reported alongside it. Pairing a
        # standardised effect with an interval computed for a different
        # quantity is a common and avoidable reporting error.
        effect_name="risk difference", effect_size=p1 - p2,
        effect2_name="Cohen's h", effect2_size=cohens_h(p1, p2),
        ci_low=lo, ci_high=hi, n=n1 + n2, conf=conf,
        groups={
            label1: proportion(k1, n1, conf),
            label2: proportion(k2, n2, conf),
            "difference": p1 - p2,
            "difference_pct": (p1 - p2) * 100,
        },
        note=f"The CI is Newcombe's hybrid score interval on the risk "
             f"difference ({label1} minus {label2}). Cohen's h is reported "
             f"without an interval and is interpreted using Cohen's "
             f"conventions (.20 small, .50 medium, .80 large).",
    )


# --------------------------------------------------------------------------
# contingency tables
# --------------------------------------------------------------------------

def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    k = min(r - 1, c - 1)
    if n == 0 or k <= 0:
        return float("nan")
    return math.sqrt(chi2 / (n * k))


def chi_square_test(
    table: Sequence[Sequence[int]],
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    conf: float = DEFAULT_CONF,
    n_boot: int = 2000,
    seed: int = RNG_SEED,
) -> TestResult:
    """
    Chi-square test of independence with Cramer's V and a bootstrap CI for V.

    Used for "does label status depend on element role / app category".
    With very large n almost anything is significant, so V is what you
    actually report and interpret.
    """
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("chi_square_test needs a 2-D table")

    # Rows or columns that sum to zero make the expected frequencies zero and
    # the test undefined. This happens legitimately: a role such as 'container'
    # is exempt from labelling, so its labelled/unlabelled row is all zeros.
    # Drop those and record exactly which were dropped, so the report says so
    # rather than the script crashing.
    row_labels = list(row_labels) if row_labels else [f"row{i}" for i in range(arr.shape[0])]
    col_labels = list(col_labels) if col_labels else [f"col{j}" for j in range(arr.shape[1])]
    keep_rows = arr.sum(axis=1) > 0
    keep_cols = arr.sum(axis=0) > 0
    dropped = {
        "rows": [row_labels[i] for i, k in enumerate(keep_rows) if not k],
        "cols": [col_labels[j] for j, k in enumerate(keep_cols) if not k],
    }
    arr = arr[keep_rows][:, keep_cols]
    row_labels = [row_labels[i] for i, k in enumerate(keep_rows) if k]
    col_labels = [col_labels[j] for j, k in enumerate(keep_cols) if k]
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(
            "chi_square_test needs at least a 2x2 table after dropping empty "
            f"rows and columns; got {arr.shape}. Dropped: {dropped}"
        )

    chi2, p, dof, _ = sps.chi2_contingency(arr, correction=False)
    n = int(arr.sum())
    v = cramers_v(chi2, n, arr.shape[0], arr.shape[1])

    rng = np.random.default_rng(seed)
    flat = arr.flatten()
    probs = flat / flat.sum()
    vs = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.multinomial(n, probs).reshape(arr.shape).astype(float)
        if (draw.sum(axis=0) == 0).any() or (draw.sum(axis=1) == 0).any():
            vs[i] = np.nan
            continue
        c2, _, _, _ = sps.chi2_contingency(draw, correction=False)
        vs[i] = cramers_v(c2, n, arr.shape[0], arr.shape[1])
    vs = vs[~np.isnan(vs)]
    alpha = (1 - conf) / 2
    lo, hi = (np.quantile(vs, alpha), np.quantile(vs, 1 - alpha)) if vs.size else (np.nan, np.nan)

    return TestResult(
        name="Chi-square test of independence",
        statistic_name="chi2", statistic=float(chi2), p_value=float(p), df=float(dof),
        effect_name="Cramer's V", effect_size=float(v),
        ci_low=float(lo), ci_high=float(hi), n=n, conf=conf,
        groups={
            "table": arr.astype(int).tolist(),
            "row_labels": row_labels,
            "col_labels": col_labels,
            "dropped_empty": dropped,
        },
        note=(
            f"Cramer's V CI from {len(vs):,} parametric bootstrap replicates."
            + (f" Empty rows/columns dropped before testing: {dropped}."
               if (dropped["rows"] or dropped["cols"]) else "")
        ),
    )


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

def permutation_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_perm: int = 10000,
    alternative: str = "two-sided",
    seed: int = RNG_SEED,
    name: str = "Permutation test",
    effect_name: str = "observed difference",
    conf: float = DEFAULT_CONF,
) -> TestResult:
    """
    Label-shuffling permutation test.

    RQ3 uses this to ask whether elements touched by task trajectories are
    labelled at a higher rate than the background population of elements on
    the same screens. The default statistic is the difference in means,
    which for 0/1 data is the difference in proportions.

    The p-value uses the (r+1)/(m+1) convention, so it is never exactly zero.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("permutation_test needs non-empty groups")

    if statistic is None:
        def statistic(x: np.ndarray, y: np.ndarray) -> float:
            return float(x.mean() - y.mean())

    observed = statistic(a, b)
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)

    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        null[i] = statistic(pooled[:n_a], pooled[n_a:])

    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    elif alternative == "less":
        hits = int(np.sum(null <= observed))
    else:
        hits = int(np.sum(np.abs(null) >= abs(observed)))
    p = (hits + 1) / (n_perm + 1)

    # CI on the observed difference by bootstrap, not from the null.
    boot = np.empty(2000)
    for i in range(boot.size):
        boot[i] = statistic(
            rng.choice(a, size=n_a, replace=True),
            rng.choice(b, size=b.size, replace=True),
        )
    alpha = (1 - conf) / 2
    lo, hi = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))

    return TestResult(
        name=name,
        statistic_name="diff", statistic=float(observed), p_value=float(p), df=None,
        effect_name=effect_name, effect_size=float(observed),
        ci_low=lo, ci_high=hi, n=int(a.size + b.size), conf=conf,
        groups={
            "n_a": int(a.size), "n_b": int(b.size),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "n_permutations": int(n_perm), "alternative": alternative,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        },
        note=f"p from {n_perm:,} label permutations using the (r+1)/(m+1) "
             f"convention; CI from 2,000 bootstrap replicates.",
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 5000,
    conf: float = DEFAULT_CONF,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Returns (point estimate, ci_low, ci_high)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    draws = np.array([
        float(statistic(rng.choice(arr, size=arr.size, replace=True)))
        for _ in range(n_boot)
    ])
    alpha = (1 - conf) / 2
    return point, float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size for two independent samples. Range [-1, 1]."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    # rank-based computation, O(n log n) rather than O(n*m)
    combined = np.concatenate([x, y])
    ranks = sps.rankdata(combined)
    rx = ranks[: x.size].sum()
    u = rx - x.size * (x.size + 1) / 2
    return float(2 * u / (x.size * y.size) - 1)


def odds_ratio(a: int, b: int, c: int, d: int, conf: float = DEFAULT_CONF) -> dict[str, float]:
    """2x2 odds ratio with a Woolf log CI and Haldane-Anscombe correction."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    return {
        "odds_ratio": or_,
        "ci_low": math.exp(math.log(or_) - z * se),
        "ci_high": math.exp(math.log(or_) + z * se),
        "log_se": se,
    }


def fdr_bh(p_values: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (rejected, adjusted_p). Use whenever a
    table reports more than a handful of tests."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out < alpha, out


# --------------------------------------------------------------------------
# interpretation (Cohen's conventions, stated so the paper can cite them)
# --------------------------------------------------------------------------

def interpret_h(h: float) -> str:
    a = abs(h)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def interpret_v(v: float, df_small: int = 1) -> str:
    a = abs(v)
    small, medium, large = (0.10, 0.30, 0.50) if df_small <= 1 else (0.07, 0.21, 0.35)
    return "negligible" if a < small else "small" if a < medium else "medium" if a < large else "large"


def interpret_delta(d: float) -> str:
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def interpret_or(or_: float) -> str:
    a = or_ if or_ >= 1 else 1 / or_ if or_ > 0 else float("inf")
    return "negligible" if a < 1.5 else "small" if a < 2.5 else "medium" if a < 4.3 else "large"
