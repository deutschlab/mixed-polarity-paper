"""
Every text in a saved figure in Arial, 8 pt.

The journal figures use one font and one size for all text: axis labels, tick labels, legends
and their titles, titles, colour bar labels, annotations, table cells and any other text on the
figure. The figure scripts set many sizes of their own (fontsize=4 to 16, seaborn contexts,
rcParams), so this module sets the text whatever the script set.

A figure script calls use_arial_8() once, near the top, before it draws anything:

    from methods.figure_text import use_arial_8
    use_arial_8()

From then on:
  - every Figure.savefig (and so plt.savefig, a seaborn grid's savefig and PdfPages) first gives
    every text in the figure the family Arial and the size 8 pt, sets the tick label settings of
    every axes so that ticks made while drawing get the same, and saves with maths text in Arial,
    even if the script changed the maths font settings later (exponents and subscripts keep
    matplotlib's smaller size, 0.7 of 8 pt);
  - every legend is built at Arial 8 pt, whatever font size or properties the script passes, so
    its handles, spacing and frame are those of an 8 pt legend (they are fixed when a legend is
    built, so setting its text at the save would leave them at the script's size).
Only text changes: no data, colour or line. Labels take more or less room at 8 pt than at the
script's own size, so a panel's layout can change; matplotlib also chooses how many ticks to draw
from the label size.

use_arial_8() also sets the default sizes in rcParams to 8 pt and maths text to Arial, so that
layout steps run before saving (tight_layout, constrained layout) already allow for 8 pt text.
It warns if matplotlib cannot find Arial on this machine: SVG text still names Arial, but PNG and
PDF output would use another font. Calling it more than once has no further effect.
Figures written with fig.canvas.print_figure, not savefig, do not go through it. A few maths
symbols have no Arial form in matplotlib's maths text and keep its own fonts (\\sqrt, \\left( \\right),
\\cdot, \\sim, ^\\circ, \\mathbb); the figure scripts use none of them.
"""

import functools
import warnings

import matplotlib
import matplotlib.figure
import matplotlib.legend
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table
from matplotlib.text import Text

FONT_FAMILY = "Arial"
FONT_SIZE = 8

_MATH_RC = {
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": FONT_FAMILY + ":italic",
    "mathtext.bf": FONT_FAMILY + ":bold",
    "mathtext.sf": FONT_FAMILY,
    "mathtext.cal": FONT_FAMILY,
    "mathtext.tt": FONT_FAMILY,
}

_RC = {
    "font.family": [FONT_FAMILY],
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "figure.labelsize": FONT_SIZE,
    **_MATH_RC,
}


def set_figure_text(fig, family=FONT_FAMILY, size=FONT_SIZE):
    """Give every text in fig the font family and size, including tick labels made later."""
    for ax in fig.findobj(Axes):
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_tick_params(which="both", labelsize=size, labelfontfamily=family)
    for table in fig.findobj(Table):
        table.auto_set_font_size(False)
        for cell in table.get_celld().values():
            cell.get_text().set_fontfamily(family)
            cell.get_text().set_fontsize(size)
    for text in fig.findobj(Text):
        text.set_fontfamily(family)
        text.set_fontsize(size)
        text.set_math_fontfamily("custom")   # maths in Arial (mathtext.rm etc., set at the save)


def _font(prop, family=FONT_FAMILY, size=FONT_SIZE):
    """A copy of a legend's font properties (None, a dict, a FontProperties or a pattern) in family and size."""
    if prop is None:
        fp = FontProperties()
    elif isinstance(prop, FontProperties):
        fp = prop.copy()
    elif isinstance(prop, dict):
        fp = FontProperties(**prop)
    else:
        fp = FontProperties(prop)
    fp.set_family(family)
    fp.set_size(size)
    return fp


def use_arial_8():
    """Set every text to Arial 8 pt in each figure as it is saved, and build every legend at it, from now on."""
    matplotlib.rcParams.update(_RC)
    try:
        font_manager.findfont(FontProperties(family=FONT_FAMILY), fallback_to_default=False)
    except ValueError:
        warnings.warn("Arial is not installed for matplotlib here: SVG text names Arial, but PNG and PDF "
                      "output will use another font", stacklevel=2)

    current = matplotlib.figure.Figure.savefig
    if not getattr(current, "_sets_figure_text", False):
        @functools.wraps(current)
        def savefig(self, *args, **kwargs):
            set_figure_text(self)
            with matplotlib.rc_context(_MATH_RC):
                return current(self, *args, **kwargs)

        savefig._sets_figure_text = True
        matplotlib.figure.Figure.savefig = savefig

    legend_init = matplotlib.legend.Legend.__init__
    if not getattr(legend_init, "_sets_figure_text", False):
        @functools.wraps(legend_init)
        def __init__(self, *args, **kwargs):
            kwargs.pop("fontsize", None)
            kwargs["prop"] = _font(kwargs.get("prop"))
            if kwargs.get("title_fontproperties") is not None:
                kwargs.pop("title_fontsize", None)
                kwargs["title_fontproperties"] = _font(kwargs["title_fontproperties"])
            else:
                kwargs.pop("title_fontproperties", None)
                kwargs["title_fontsize"] = FONT_SIZE
            legend_init(self, *args, **kwargs)

        __init__._sets_figure_text = True
        matplotlib.legend.Legend.__init__ = __init__
