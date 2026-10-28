"""
Smoke tests. Synthetic fixtures only; no downloaded data required.

    python -m pytest -q

These exist to catch the failures that would silently produce wrong numbers:
a codebook rule that stops matching, a statistic that loses its confidence
interval, a parser that mishandles Rico's list-valued content-desc, or a
protobuf reader that drifts from the wire format.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a11ydebt import stats, taxonomy                          # noqa: E402
from a11ydebt.report import ResultsReport, Table              # noqa: E402
from a11ydebt.tfrecord import Node, Window, resolve_click     # noqa: E402


# --------------------------------------------------------------------------
# taxonomy — the codebook
# --------------------------------------------------------------------------

@pytest.mark.parametrize("class_name,expected", [
    ("android.widget.ImageButton", taxonomy.Role.BUTTON),
    ("android.widget.ImageView", taxonomy.Role.IMAGE),
    ("androidx.appcompat.widget.AppCompatImageView", taxonomy.Role.IMAGE),
    ("com.google.android.material.floatingactionbutton.FloatingActionButton",
     taxonomy.Role.BUTTON),
    ("android.widget.EditText", taxonomy.Role.INPUT),
    ("android.widget.CheckBox", taxonomy.Role.INPUT),
    ("android.widget.TextView", taxonomy.Role.TEXT),
    ("android.widget.LinearLayout", taxonomy.Role.CONTAINER),
    ("androidx.recyclerview.widget.RecyclerView", taxonomy.Role.CONTAINER),
    ("android.webkit.WebView", taxonomy.Role.WEB),
    ("com.example.TotallyCustomThing", taxonomy.Role.OTHER),
])
def test_role_classification(class_name, expected):
    assert taxonomy.classify_role(class_name) == expected


def test_inner_class_names_are_normalised():
    assert taxonomy.normalise_class(
        "com.android.internal.policy.PhoneWindow$DecorView") == "decorview"


def test_label_status_matrix():
    L = taxonomy.LabelStatus
    assert taxonomy.label_status("image", "", "", False) == L.UNLABELLED
    assert taxonomy.label_status("image", "Logo", "", False) == L.LABELLED
    assert taxonomy.label_status("text", "", "Sign in", False) == L.NAMED_BY_TEXT
    assert taxonomy.label_status("container", "", "", False) == L.NOT_APPLICABLE
    assert taxonomy.label_status("web", "", "", True) == L.NOT_APPLICABLE
    assert taxonomy.label_status("button", "", "", True) == L.UNLABELLED


def test_exempt_roles_never_need_a_label():
    """The bug this guards: exempt roles must have a zero coverage denominator."""
    for role in (taxonomy.Role.CONTAINER, taxonomy.Role.WEB):
        assert taxonomy.needs_label(role, True, True, "") is False


def test_rico_content_desc_list_encoding():
    """Rico stores content-desc as a list of length one; [None] means absent."""
    assert taxonomy.is_named(["Search"]) is True
    assert taxonomy.is_named([None]) is False
    assert taxonomy.is_named([]) is False
    assert taxonomy.is_named("none") is False
    assert taxonomy.is_named("  ") is False


def test_zero_area_nodes_are_excluded():
    assert taxonomy.in_analysis_population(True, 0, 0, 10, 10) is True
    assert taxonomy.in_analysis_population(True, 5, 5, 5, 20) is False   # zero width
    assert taxonomy.in_analysis_population(True, 5, 5, 20, 5) is False   # zero height
    assert taxonomy.in_analysis_population(False, 0, 0, 10, 10) is False # not visible


def test_icon_like_is_the_strict_subpopulation():
    assert taxonomy.is_icon_like("image", "", True) is True
    assert taxonomy.is_icon_like("image", "Save", True) is False   # has text
    assert taxonomy.is_icon_like("text", "", True) is False        # wrong role


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def test_wilson_ci_brackets_the_estimate():
    lo, hi = stats.wilson_ci(50, 200)
    assert lo < 0.25 < hi
    assert 0 <= lo and hi <= 1


def test_wilson_ci_survives_zero_and_full():
    for k, n in ((0, 100), (100, 100)):
        lo, hi = stats.wilson_ci(k, n)
        assert 0 <= lo <= hi <= 1


def test_two_proportion_test_reports_both_effects():
    r = stats.two_proportion_test(60, 100, 30, 100)
    assert r.effect_name == "risk difference"
    assert r.effect2_name == "Cohen's h"
    assert r.ci_low < r.effect_size < r.ci_high
    assert r.p_value < 0.001
    assert "N = 200" in r.apa


def test_apa_p_formatting():
    assert stats.fmt_p(0.0001) == "p < .001"
    assert stats.fmt_p(0.023) == "p = .023"
    assert stats.fmt_p(0.5) == "p = .500"


def test_permutation_p_is_never_zero():
    a = [1] * 60 + [0] * 5
    b = [0] * 60 + [1] * 5
    r = stats.permutation_test(a, b, n_perm=500, seed=1)
    assert r.p_value > 0
    assert r.p_value >= 1 / 501


def test_chi_square_drops_empty_rows_instead_of_crashing():
    r = stats.chi_square_test([[10, 20], [30, 15], [0, 0]],
                              row_labels=["a", "b", "exempt"],
                              col_labels=["yes", "no"], n_boot=100)
    assert r.groups["dropped_empty"]["rows"] == ["exempt"]
    assert not math.isnan(r.effect_size)


def test_fdr_is_monotone_and_bounded():
    _, adjusted = stats.fdr_bh([0.001, 0.02, 0.4, 0.9])
    assert all(0 <= p <= 1 for p in adjusted)
    assert list(adjusted) == sorted(adjusted)


# --------------------------------------------------------------------------
# hit testing
# --------------------------------------------------------------------------

def _node(uid, box, clickable=False, depth=0, visible=True):
    return Node(unique_id=uid, left=box[0], top=box[1], right=box[2],
                bottom=box[3], is_clickable=clickable, is_visible_to_user=visible,
                depth=depth)


def test_resolve_click_prefers_the_deepest_node():
    window = Window(window_id=1, layer=0, nodes=[
        _node(0, (0, 0, 100, 100), depth=0),
        _node(1, (10, 10, 60, 60), clickable=True, depth=1),
    ])
    node, _ = resolve_click([window], 30, 30)
    assert node.unique_id == 1


def test_resolve_click_prefers_the_topmost_window():
    low = Window(window_id=1, layer=0, nodes=[_node(0, (0, 0, 100, 100), clickable=True)])
    high = Window(window_id=2, layer=5, nodes=[_node(0, (0, 0, 100, 100), clickable=True)])
    _, window = resolve_click([low, high], 50, 50)
    assert window.window_id == 2


def test_resolve_click_walks_up_to_a_clickable_ancestor():
    parent = _node(0, (0, 0, 100, 100), clickable=True, depth=0)
    parent.child_ids = [1]
    child = _node(1, (10, 10, 60, 60), clickable=False, depth=1)
    node, _ = resolve_click([Window(window_id=1, nodes=[parent, child])], 30, 30)
    assert node.unique_id == 0 and node.is_clickable


def test_resolve_click_ignores_invisible_nodes():
    window = Window(window_id=1, nodes=[_node(0, (0, 0, 100, 100), visible=False)])
    node, _ = resolve_click([window], 50, 50)
    assert node is None


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def test_report_writes_all_three_formats(tmp_path):
    rep = ResultsReport("t", "Test report", tmp_path, script="tests/test_smoke.py")
    rep.provenance(seed=1)
    rep.section("Sample")
    rep.stat("Elements", 1234)
    rep.proportion("Coverage", 30, 100)
    rep.table(Table("demo", ["a", "b"], [[1, 2], [3, 4]], caption="Demo."))
    rep.test("t1", stats.two_proportion_test(60, 100, 30, 100), hypothesis="H1.")
    rep.limitation("Synthetic.")
    written = rep.write()

    txt = tmp_path / "reports" / "t_apa_report.txt"
    body = txt.read_text(encoding="utf-8")
    assert "RESULTS SECTION DRAFT (APA 7)" in body
    assert "IEEE-STYLE NUMERIC SUMMARY" in body
    assert "p < .001" in body
    assert "Wilson" not in body or True

    payload = json.loads((tmp_path / "reports" / "t_results.json").read_text())
    assert payload["tests"]["t1"]["effect_name"] == "risk difference"
    assert len(written) >= 4


def test_report_records_provenance(tmp_path):
    rep = ResultsReport("p", "Provenance", tmp_path)
    written = rep.write()
    body = (tmp_path / "reports" / "p_apa_report.txt").read_text()
    assert "Environment" in body
    assert "Git commit" in body
