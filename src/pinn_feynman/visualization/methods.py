"""Colours and styles of the training methods, shared by the figures.

The method of a run keeps the same colour in every figure (`METHOD_COLOUR`, fixed order, never cycled); `group_styles`
gives one colour per group of runs (the tone of its method family) and `group_markers` one marker per group.
"""

from importlib import import_module
from importlib.util import find_spec

from .style import SERIES

#: fixed colour per method, shared by all the figures of all reports
METHOD_COLOUR = {"baseline": SERIES[0], "r3": SERIES[1], "r3_continuous": SERIES[4], "p3": "#8a5cd0"}
METHOD_LABEL = {"baseline": "baseline", "r3": "R3", "r3_continuous": "R3 (continuum)", "p3": "DW (distance weighting)"}
#: families recognised in a group label, in the order they are tried
FAMILIES = ("r3_continuous", "r3")

#: optional module with the colours and figures of further methods; absent from the public repository
_EXTRA = f"{__package__}.extra"
if find_spec(_EXTRA) is not None:
    _extra = import_module(_EXTRA)
    METHOD_COLOUR.update(_extra.METHOD_COLOUR)
    METHOD_LABEL.update(_extra.METHOD_LABEL)
    FAMILIES = _extra.FAMILIES


LINESTYLES = ("-", "--", ":", "-.")      # kept for the callers that still ask for a style per variant
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
PART_COLOUR = {"real": SERIES[0], "imag": SERIES[1]}   # the two parts of a report in which they share a figure
REFERENCE_COLOUR = "#4a4a48"             # the reference group of a comparison: dark neutral grey
#: how far a variant is pushed from the colour of its family: (towards white, towards black), in order of use
TINTS = ((0.0, 0.0), (0.0, 0.40), (0.45, 0.0), (0.0, 0.65), (0.70, 0.0))

_reference = None


def set_reference(label):
    """The reference group of the figures drawn next (None: no reference)."""
    global _reference
    _reference = label


def family_of(label):
    """Method family of a group label ('margin_01pc_r3_continuous' -> 'r3_continuous', 'bu_baseline_p3' -> 'p3', ...)."""
    for family in FAMILIES:
        if family in label:
            return family
    if label.endswith("_p3") or "_p3_" in label or label == "p3":      # distance weighting: never read as baseline
        return "p3"
    return "baseline"


def tint(colour, toward_white=0.0, toward_black=0.0):
    """`colour` ('#rrggbb') mixed with white or with black."""
    rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    rgb = [c + (255 - c) * toward_white for c in rgb]
    rgb = [c * (1 - toward_black) for c in rgb]
    return "#" + "".join(f"{round(c):02x}" for c in rgb)


def group_styles(labels, reference=None):
    """{label: (colour, linestyle)}: one colour per group. The colour is the tone of the method family; the
    k-th group of a family gets the k-th tone (`TINTS`), so groups of one family differ in tone and never share a
    colour for the first five. The reference group (argument, or `set_reference`) is dark grey. The line style is
    always solid: dashes and dots are kept for the nature of a curve (validation, final network, trained points...)."""
    reference = _reference if reference is None else reference
    seen, out = {}, {}
    for label in labels:
        if label == reference:
            out[label] = (REFERENCE_COLOUR, "-")
            continue
        if label in PART_COLOUR:                       # groups named "real" / "imag": one fixed colour per part
            out[label] = (PART_COLOUR[label], "-")
            continue
        family = family_of(label)
        k = seen.get(family, 0)
        seen[family] = k + 1
        white, black = TINTS[k % len(TINTS)]
        out[label] = (tint(METHOD_COLOUR[family], white, black), "-")
    return out


def group_markers(labels):
    """{label: marker}: one marker per group, in order, so that groups with similar tones still differ in dots and bars."""
    return {label: MARKERS[k % len(MARKERS)] for k, label in enumerate(labels)}


if find_spec(_EXTRA) is not None:                       # the figures of the further methods, under their old names
    from .extra import _x, INK_2, cumulative, memory, r3_population, rba_time, rba_weights  # noqa: E402,F401
