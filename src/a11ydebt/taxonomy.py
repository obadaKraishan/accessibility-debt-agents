"""
The element taxonomy: the operational codebook, expressed as code.

This module is the single place where the study decides three things:

  1. Which nodes count as elements at all (the analysis population).
  2. What kind of element each node is (its role).
  3. Whether an element is labelled, unlabelled, or exempt from needing
     a label.

Every number in the paper flows from these rules, so they are written
once here, documented in phase3_codebook/CODEBOOK.md in prose, and
validated against human coding in phase3_codebook/02_intercoder_reliability.py.

CROSS-DATASET FAIRNESS
----------------------
Rico view hierarchies carry an `ancestors` field (the full Java superclass
chain). AndroidControl accessibility trees do not. If we classified Rico
using `ancestors` and AndroidControl using only `class_name`, any
difference between the two datasets would be partly an artifact of the
richer Rico metadata.

So classification is done TWICE:

  * `classify_role(class_name)` uses the class name alone. This is the
    PRIMARY measure and the only one used for cross-dataset comparison
    (RQ2).
  * `classify_role(class_name, ancestors=...)` additionally consults the
    superclass chain. This is a Rico-only ROBUSTNESS measure, reported in
    the sensitivity analysis to show that the class-name-only rule does
    not distort the Rico figures.

Never mix the two in a comparison. phase4_validation/02_sensitivity_checks.py
quantifies the gap between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "Role",
    "LabelStatus",
    "ELEMENT_ROLES",
    "LABEL_STATUSES",
    "CANONICAL_NODE_FIELDS",
    "normalise_class",
    "classify_role",
    "is_interactive",
    "has_visible_text",
    "is_named",
    "needs_label",
    "label_status",
    "is_icon_like",
    "is_floating_action_button",
    "in_analysis_population",
    "positive_area",
    "NodeView",
]


# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------

class Role:
    """Element roles. Deliberately coarse so the categories stay reliable."""

    IMAGE = "image"        # ImageView and friends: cannot be read aloud without a label
    BUTTON = "button"      # Button, ImageButton, FloatingActionButton
    TEXT = "text"          # TextView and friends: self-naming when text is present
    INPUT = "input"        # EditText, CheckBox, RadioButton, Switch, SeekBar, Spinner
    CONTAINER = "container"  # layouts and list containers
    WEB = "web"            # WebView: contents are opaque to the view hierarchy
    OTHER = "other"


class LabelStatus:
    LABELLED = "labelled"              # has a non-empty content description
    UNLABELLED = "unlabelled"          # needs one and does not have one  <- the debt
    NAMED_BY_TEXT = "named_by_text"    # no content-desc but has visible text
    NOT_APPLICABLE = "not_applicable"  # decorative or structural; no label expected


ELEMENT_ROLES: tuple[str, ...] = (
    Role.IMAGE, Role.BUTTON, Role.TEXT, Role.INPUT,
    Role.CONTAINER, Role.WEB, Role.OTHER,
)

LABEL_STATUSES: tuple[str, ...] = (
    LabelStatus.LABELLED, LabelStatus.UNLABELLED,
    LabelStatus.NAMED_BY_TEXT, LabelStatus.NOT_APPLICABLE,
)

#: The fields both datasets can supply. Parsers must emit exactly these,
#: so that Rico rows and AndroidControl rows are structurally identical.
CANONICAL_NODE_FIELDS: tuple[str, ...] = (
    "dataset", "screen_uid", "node_uid", "parent_uid", "depth",
    "class_name", "content_desc", "text", "resource_id",
    "clickable", "long_clickable", "enabled", "focusable", "visible_to_user",
    "x1", "y1", "x2", "y2", "screen_w", "screen_h",
)


# --------------------------------------------------------------------------
# class-name matching
# --------------------------------------------------------------------------

def normalise_class(class_name: str | None) -> str:
    """'android.widget.ImageButton' -> 'imagebutton'. Handles inner classes."""
    if not class_name:
        return ""
    name = str(class_name).strip()
    if "$" in name:
        name = name.split("$")[-1]
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    return name.lower()


# Suffix rules, evaluated in order. First match wins, so the more specific
# entries come first. Matching on suffixes rather than exact names is what
# makes this work across AppCompat, Material, androidx, and vendor classes
# (AppCompatImageView, ShapeableImageView, MaterialButton, and so on).
_ROLE_RULES: tuple[tuple[str, str], ...] = (
    # inputs first: several of them end in -button or -view
    ("edittext", Role.INPUT),
    ("autocompletetextview", Role.INPUT),
    ("checkbox", Role.INPUT),
    ("radiobutton", Role.INPUT),
    ("togglebutton", Role.INPUT),
    ("switchcompat", Role.INPUT),
    ("switch", Role.INPUT),
    ("seekbar", Role.INPUT),
    ("ratingbar", Role.INPUT),
    ("spinner", Role.INPUT),
    ("numberpicker", Role.INPUT),
    ("datepicker", Role.INPUT),
    ("timepicker", Role.INPUT),
    ("searchview", Role.INPUT),
    # buttons
    ("floatingactionbutton", Role.BUTTON),
    ("imagebutton", Role.BUTTON),
    ("button", Role.BUTTON),
    # images
    ("imageview", Role.IMAGE),
    ("circleimageview", Role.IMAGE),
    ("photoview", Role.IMAGE),
    ("gifimageview", Role.IMAGE),
    ("simpledraweeview", Role.IMAGE),   # Facebook Fresco, very common in 2017 apps
    ("iconview", Role.IMAGE),
    ("icon", Role.IMAGE),
    # web
    ("webview", Role.WEB),
    # text
    ("textview", Role.TEXT),
    ("textclock", Role.TEXT),
    ("chronometer", Role.TEXT),
    # containers
    ("recyclerview", Role.CONTAINER),
    ("listview", Role.CONTAINER),
    ("gridview", Role.CONTAINER),
    ("scrollview", Role.CONTAINER),
    ("viewpager", Role.CONTAINER),
    ("viewgroup", Role.CONTAINER),
    ("layout", Role.CONTAINER),
    ("toolbar", Role.CONTAINER),
    ("actionbar", Role.CONTAINER),
    ("tablayout", Role.CONTAINER),
    ("decorview", Role.CONTAINER),
    ("drawerlayout", Role.CONTAINER),
    ("cardview", Role.CONTAINER),
)

# Used only for the Rico-only robustness variant.
_ANCESTOR_RULES: tuple[tuple[str, str], ...] = (
    ("android.widget.EditText", Role.INPUT),
    ("android.widget.CompoundButton", Role.INPUT),
    ("android.widget.AbsSeekBar", Role.INPUT),
    ("android.widget.AbsSpinner", Role.INPUT),
    ("android.widget.ImageButton", Role.BUTTON),
    ("android.widget.Button", Role.BUTTON),
    ("android.widget.ImageView", Role.IMAGE),
    ("android.webkit.WebView", Role.WEB),
    ("android.widget.TextView", Role.TEXT),
    ("android.widget.AdapterView", Role.CONTAINER),
    ("android.view.ViewGroup", Role.CONTAINER),
)


def classify_role(class_name: str | None, ancestors: Sequence[str] | None = None) -> str:
    """
    Assign an element role.

    Pass `ancestors` ONLY for the Rico-only robustness variant. The primary
    analysis calls this with the class name alone so that Rico and
    AndroidControl are classified by identical information.
    """
    key = normalise_class(class_name)
    if key:
        for suffix, role in _ROLE_RULES:
            if key.endswith(suffix):
                return role
    if ancestors:
        for ancestor, role in _ANCESTOR_RULES:
            if ancestor in ancestors:
                return role
    return Role.OTHER


def is_floating_action_button(class_name: str | None) -> bool:
    return normalise_class(class_name).endswith("floatingactionbutton")


# --------------------------------------------------------------------------
# label logic
# --------------------------------------------------------------------------

def _clean(value: object) -> str:
    """Normalise a text-ish field to a stripped string. Handles Rico's [None]."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if value is None:
            return ""
    text = str(value).strip()
    # Rico serialises absent values as the literal strings below often enough
    # to be worth guarding against.
    if text.lower() in {"none", "null", "nan", ""}:
        return ""
    return text


def is_interactive(clickable: bool, long_clickable: bool = False, role: str | None = None) -> bool:
    """An element a user can act on, and therefore one a screen reader must name."""
    if bool(clickable) or bool(long_clickable):
        return True
    return role in {Role.INPUT, Role.BUTTON}


def has_visible_text(text: object) -> bool:
    return bool(_clean(text))


def is_named(content_desc: object) -> bool:
    """True when a developer supplied a content description."""
    return bool(_clean(content_desc))


def needs_label(role: str, clickable: bool, long_clickable: bool, text: object) -> bool:
    """
    Does this element require a content description?

    Yes when it is interactive or image-bearing AND carries no visible text
    of its own. An element whose own text is announced does not need a
    separate description; a bare icon does. Containers, plain text views,
    and web views are exempt: containers are structural, text names itself,
    and a WebView's contents are opaque to the view hierarchy.
    """
    if role in {Role.CONTAINER, Role.WEB}:
        return False
    if has_visible_text(text):
        return False
    if role == Role.IMAGE:
        return True
    return is_interactive(clickable, long_clickable, role)


def label_status(
    role: str,
    content_desc: object,
    text: object,
    clickable: bool,
    long_clickable: bool = False,
) -> str:
    """
    The dependent variable of the whole study.

    Returns one of LabelStatus. `unlabelled` is the accessibility debt:
    an element that needed a name and did not get one.
    """
    if is_named(content_desc):
        return LabelStatus.LABELLED
    if needs_label(role, clickable, long_clickable, text):
        return LabelStatus.UNLABELLED
    if has_visible_text(text):
        return LabelStatus.NAMED_BY_TEXT
    return LabelStatus.NOT_APPLICABLE


def is_icon_like(role: str, text: object, clickable: bool, long_clickable: bool = False) -> bool:
    """
    The strictest and most defensible unit: a tappable graphic with no text.

    This is the population where a missing label is unambiguously a defect,
    and it is the headline denominator to report when a reviewer challenges
    the broader definitions.
    """
    if role not in {Role.IMAGE, Role.BUTTON}:
        return False
    if has_visible_text(text):
        return False
    return is_interactive(clickable, long_clickable, role)


# --------------------------------------------------------------------------
# analysis population
# --------------------------------------------------------------------------

def positive_area(x1: float, y1: float, x2: float, y2: float) -> bool:
    """
    Rico contains a very large number of zero-area and negative-area nodes:
    48.7% of all nodes fail this test, and 1,716 of them are still flagged
    visible-to-user (phase4_validation/outputs/reports/validation_apa_report.txt).
    Filtering on the visibility flag alone is not enough, which is why area
    is checked separately.
    """
    try:
        return (float(x2) - float(x1)) > 0 and (float(y2) - float(y1)) > 0
    except (TypeError, ValueError):
        return False


def in_analysis_population(
    visible_to_user: bool,
    x1: float, y1: float, x2: float, y2: float,
    enabled: bool = True,
    require_enabled: bool = False,
) -> bool:
    """The inclusion rule for every count in the paper. Applied identically
    to both datasets. `require_enabled` is off by default and exercised in
    the sensitivity analysis."""
    if not bool(visible_to_user):
        return False
    if not positive_area(x1, y1, x2, y2):
        return False
    if require_enabled and not bool(enabled):
        return False
    return True


# --------------------------------------------------------------------------
# convenience wrapper
# --------------------------------------------------------------------------

@dataclass
class NodeView:
    """A canonical node, with the taxonomy applied. Parsers emit these."""

    dataset: str
    screen_uid: str
    node_uid: str
    class_name: str
    content_desc: str
    text: str
    clickable: bool
    long_clickable: bool
    visible_to_user: bool
    enabled: bool
    x1: int
    y1: int
    x2: int
    y2: int
    ancestors: tuple[str, ...] = ()

    @property
    def role(self) -> str:
        return classify_role(self.class_name)

    @property
    def role_with_ancestors(self) -> str:
        return classify_role(self.class_name, self.ancestors)

    @property
    def status(self) -> str:
        return label_status(
            self.role, self.content_desc, self.text,
            self.clickable, self.long_clickable,
        )

    @property
    def icon_like(self) -> bool:
        return is_icon_like(self.role, self.text, self.clickable, self.long_clickable)

    @property
    def included(self) -> bool:
        return in_analysis_population(
            self.visible_to_user, self.x1, self.y1, self.x2, self.y2, self.enabled
        )

    def contains(self, x: float, y: float) -> bool:
        """Point-in-bounds hit test, used by RQ3."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def summarise_statuses(statuses: Iterable[str]) -> dict[str, int]:
    counts = {s: 0 for s in LABEL_STATUSES}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    return counts
