# Codebook: element roles and label status

This document is the prose statement of the coding scheme. Its executable
counterpart is `src/a11ydebt/taxonomy.py`. **If the two ever disagree, that is
a bug**: `phase0_setup/02_init_project.py` runs nine hand-checked cases through
the code on every launch to keep them aligned, and
`phase3_codebook/02_intercoder_reliability.py` scores the code against human
coding on a stratified sample.

Cite this file in the Method section. Ship it with the replication package.

---

## 1. Unit of analysis

The unit is a **UI element**: one node in an Android view hierarchy (Rico) or
one `AndroidAccessibilityNodeInfo` (AndroidControl).

### Inclusion rule

An element enters the analysis population when **both** hold:

1. `visible_to_user` is true, and
2. the node has **positive area** (`x2 > x1` and `y2 > y1`).

Both conditions are necessary. In Rico, **48.7% of all nodes have zero or
negative area**, and 1,716 of those are still flagged `visible-to-user`
(`phase4_validation/outputs/reports/validation_apa_report.txt`, checks table). Filtering on the visibility flag alone would therefore
admit elements that are not on screen in any meaningful sense.

`enabled` is **not** part of the primary inclusion rule. A disabled control is
still announced by a screen reader and still needs a name. The sensitivity
analysis in `phase4_validation/02_sensitivity_checks.py` reports how much the
headline numbers move if disabled elements are excluded.

---

## 2. Element role

Roles are assigned from the element's **class name only**.

### Why class name only

Rico view hierarchies carry an `ancestors` field with the full Java superclass
chain. AndroidControl accessibility trees do not. Classifying Rico with the
richer signal and AndroidControl with the poorer one would make any 2017-versus-2024
difference partly an artifact of metadata availability rather than a fact about
apps. So the **primary** classifier uses the class name for both corpora, and a
Rico-only variant using `ancestors` is reported as a robustness check.

Matching is on **class-name suffixes**, not exact names, which is what makes the
scheme survive AppCompat, Material, androidx and vendor subclasses
(`AppCompatImageView`, `ShapeableImageView`, `MaterialButton`, and so on).

| Role | Assigned when the normalised class name ends in | Examples |
|---|---|---|
| `input` | `edittext`, `checkbox`, `radiobutton`, `switch`, `seekbar`, `spinner`, `searchview`, pickers | `EditText`, `SwitchCompat` |
| `button` | `button`, `imagebutton`, `floatingactionbutton` | `MaterialButton`, `FAB` |
| `image` | `imageview`, `simpledraweeview`, `photoview`, `icon` | `AppCompatImageView` |
| `web` | `webview` | `WebView` |
| `text` | `textview`, `textclock`, `chronometer` | `AppCompatTextView` |
| `container` | `layout`, `viewgroup`, `recyclerview`, `listview`, `toolbar`, `cardview` | `LinearLayout` |
| `other` | anything unmatched | custom views |

Rules are evaluated in order and the first match wins. Inputs are tested first
because several of them end in `-button` or `-view` and would otherwise be
misfiled.

---

## 3. Label status

The dependent variable. Every included element gets exactly one of four values.

| Status | Definition |
|---|---|
| `labelled` | `content_desc` is non-empty after stripping. |
| `unlabelled` | `content_desc` is empty **and** the element needs a label. **This is the accessibility debt.** |
| `named_by_text` | `content_desc` is empty, the element does not need one, and it has visible `text`. |
| `not_applicable` | `content_desc` is empty, the element does not need one, and it has no text. |

### Does an element need a label?

Yes when **all** of:

1. its role is not `container` and not `web`, **and**
2. it has no visible `text` of its own, **and**
3. it is `image`, **or** it is interactive (`clickable`, `long_clickable`, or
   role `input` or `button`).

Rationale for each exemption:

- **Containers** are structural. A screen reader traverses them; it does not
  announce them as targets.
- **Web views** hide their content from the view hierarchy entirely. Their
  internal accessibility is a separate question this study cannot see.
- **Elements with visible text** are announced by that text. Requiring a
  separate content description would be double-counting.
- **Images always need a label**, whether interactive or not, because a
  screen reader has nothing else to announce.

### Rico's `content-desc` encoding

In Rico, `content-desc` is **always a list of length one**. It holds `[None]`
when absent and `["Search"]` when set. It is never a bare string and never
another length. The parser normalises this, and also treats the literal strings
`"none"`, `"null"` and `"nan"` as absent, because they occur in the data as
serialisation artifacts rather than as real descriptions.

---

## 4. The icon-like subpopulation

`icon_like` is the strictest and most defensible unit:

> a tappable graphic with no text of its own.

Formally: role is `image` or `button`, there is no visible text, and the element
is interactive.

This is the population where a missing content description is **unambiguously a
defect** rather than a judgement call. Report icon-like coverage as the headline
figure whenever a reviewer challenges the broader definitions. Every claim in
the paper should survive being restated on this subpopulation alone.

---

## 5. Known ambiguities and how they are resolved

| Situation | Decision | Why |
|---|---|---|
| Element has both `content_desc` and `text` | `labelled` | A description was supplied; that is what is being measured. |
| Decorative image, not interactive, no text | `unlabelled` | The view hierarchy cannot distinguish decorative from meaningful. This **inflates** the debt estimate, so it is a conservative direction for the paper's claim and must be stated as a limitation. |
| Container that is itself clickable (a tappable row) | Exempt | Its children carry the semantics. Counting it would double-count the row. |
| `content_desc` is whitespace only | `unlabelled` | Stripped before testing. |
| `content_desc` duplicates `text` exactly | `labelled` | Redundant but present. Flagged in the sensitivity analysis as low-quality labelling. |
| Element outside screen bounds | Excluded | Fails the positive-area and visibility rule. |

The decorative-image case is the largest single source of measurement error
and the reason RICO-Semantics validation exists: for the 42,434 Rico screens it
covers, a human annotator's icon label tells us whether a meaningful icon was
present where our scheme says "unlabelled". That converts "no content
description" into "no content description where a human could name the
function", which is a much stronger claim.

---

## 6. Reliability procedure

1. `01_draw_validation_sample.py` draws a stratified sample: 60 screens per
   dataset, up to 10 elements per screen, stratified by role and by
   automated label status so rare cells are represented.
2. **Two coders independently** code `role` and `label_status` on the same
   items, each filling in their own copy of
   `phase3_codebook/outputs/validation_sample_blank.csv`. Each coder works
   blind to the other and blind to `validation_sample_key.csv`.
3. `03_coder_agreement.py` reports **inter-rater reliability** between the two
   coders, plus each coder against the automated rules, with confusion
   matrices and an adjudication sheet listing every disagreement.
4. `02_intercoder_reliability.py` scores a single human sheet against the
   machine. Use it when only one coder is available.

**Target: kappa >= 0.80** on both variables. Below that, revise the rules in
`taxonomy.py`, re-run phase 2, and re-code. Report the achieved kappa and the
sample size in the Method section; a computational paper that assigns
categories without one is a predictable desk reject.

### Two figures, two claims

These are different quantities and the paper must not conflate them.

| Comparison | What it establishes | Where it belongs |
|---|---|---|
| Coder A vs Coder B | **Reliability.** Two trained humans applying the codebook reach the same category. | Method, coding procedure |
| Each coder vs the rules | **Validity.** The deterministic scheme assigns what a trained human would. | Method, automated coding |

The machine coder is deterministic, so a human-versus-machine kappa is not
inter-rater reliability and must never be described as one. It cannot disagree
with itself; only the human-to-human figure carries information about whether
the definitions are stable across people.

### Adjudication

Disagreements are resolved by discussion between the two coders, recorded in
`codebook_irr_disagreements.csv`. Report the count resolved this way. A
disagreement concentrated in one confusion-matrix cell is an ambiguous rule and
should be reworded here before final coding; a scattered one is coder fatigue
and adjudication alone is the fix.
