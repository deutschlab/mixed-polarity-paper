"""Check that methods/figure_text.py gives every saved text Arial 8 pt and changes nothing else.

The figure scripts call use_arial_8() once; from then on each Figure.savefig sets every text in
the figure to Arial 8 pt first. This tool draws made-up figures that hold each kind of text the
scripts use, at sizes from 3 to 20 pt and in another font, saves them as SVG with
svg.fonttype 'none', and reads the font of every text element back from the SVG.

Checks:
  - every text element in the SVG is Arial 8 pt: axis labels, tick labels (set with
    tick_params before the ticks exist and after), offset text, legend entries and title,
    titles, suptitle, figure text, annotations, colour bar ticks and label, maths text, a
    seaborn context with font_scale 2, a seaborn grid saved with its own savefig, and text in
    another font family;
  - table cells (a fixed size and the automatic size) are Arial 8 pt;
  - maths text stays Arial when the script changes the maths font settings after the call;
  - a legend built with other sizes (fontsize, prop, title_fontsize, title_fontproperties), on
    an axes or on the figure, has the handles, spacing and frame of the same legend built at
    8 pt from the start, and everything else (handles, colours, labels, title, loc, columns,
    frame, markerscale, anchor, bold or italic) as the same call without the step;
  - plt.savefig, Figure.savefig and PdfPages all go through the step;
  - calling use_arial_8() twice gives the same file as calling it once;
  - it works under tools/run_figure.py's own savefig (that tool replaces Figure.savefig before a
    script runs, and the script's use_arial_8() then wraps it);
  - the step changes only text: the data, colours and widths of every line, patch, collection
    and image in the figure are the same before and after it;
  - the ticks are those of the same figure drawn at 8 pt from the start (matplotlib chooses how
    many ticks to draw from the label size, so tick marks move with the text; nothing else does).

--mutants plants mistakes in a scratch copy of methods/figure_text.py and requires each to
FAIL: tick labels not set, family not set, size not set, maths upright or italic font left out,
the maths font of each text left as it was, maths settings
not applied at the save, table cells left out, legends not rebuilt, a legend's title size left
as the script set it, a legend's loc, frame, markerscale, label colour, columns and anchor
or bold changed, figure legends not rebuilt, the text set after saving, only fig.texts set, savefig not replaced, the
size set to 7, and line widths changed as well.

--outputs DIR reads every saved figure under DIR instead (the SVG and PDF files, as written by a
run of the figure scripts) and requires every text in them to be Arial 8 pt: in an SVG, the font
of every text element; in a PDF, every font size set (the Tf operator) and every font named
(BaseFont); characters outside the basic 8-bit set have no size the scan can read, but their
font names are still checked. PNG files are listed but cannot be read for fonts. Maths exponents and subscripts, which
matplotlib draws at 0.7 of the text size (5.6 pt), are listed, not failed.

It prints PASS or FAIL per check, writes outputs/tools/check_figure_text/report.txt, and exits
with status 1 on any FAIL. It needs no data; a run takes a few seconds (about a minute with
--mutants). A PASS means the step sets every text it is given; whether a panel still looks
right at 8 pt has to be seen on the panel.

    python tools/check_figure_text.py
    python tools/check_figure_text.py --mutants
    python tools/check_figure_text.py --outputs outputs_rebuild_maxsi/tools/run_all_figures/<run>
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import OUTPUT_DIR  # noqa: E402

MODULE = ROOT / "methods" / "figure_text.py"

# The checks run in a child process, so that each mutant gets a fresh matplotlib.
CHILD = r'''
import io, re, sys, importlib.util
from collections import Counter
module_path, root = sys.argv[1], sys.argv[2]
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "check_figure_text"
matplotlib.rcParams["pdf.compression"] = 0
spec = importlib.util.spec_from_file_location("figure_text", module_path)
ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft)
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))

FONT = re.compile(r"font:\s*([^;\"]*?)\s*'([^']+)'")
def fonts(svg):
    """(size and style, family) of every text and tspan style in an SVG."""
    return Counter((m.group(1).strip(), m.group(2)) for m in FONT.finditer(svg))
def all_arial_8(svg):
    f = fonts(svg)
    bad = {k: v for k, v in f.items() if k[1] != "Arial" or not re.fullmatch(r"(?:(?:italic|bold|oblique)\s+)*8px", k[0])}
    return f, bad
def texts_in(svg):
    return len(re.findall(r"<text\b", svg))
def svg_of(fig, **k):
    b = io.StringIO(); fig.savefig(b, format="svg", metadata={"Date": None}, **k); return b.getvalue()
def paths(svg):
    return re.findall(r'<path d="[^"]*"', svg)

def busy_figure(size=None):
    """The made-up figure; with size, every text is drawn at that size from the start."""
    S = (lambda n: size or n)
    sns.set_context("paper", font_scale=2 if size is None else 1)
    if size:
        plt.rcParams.update({k: size for k in ("font.size", "axes.labelsize", "axes.titlesize",
                             "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
                             "legend.title_fontsize", "figure.titlesize")})
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
    ax[0].tick_params(labelsize=S(14))                    # before any tick exists
    ax[0].plot([1, 2e6], [1, 2], marker="o", label="line")
    ax[0].legend(title="Legend", fontsize=S(14), title_fontsize=S(16))
    ax[0].set_xlabel("x label", fontsize=S(16))
    ax[0].set_ylabel(r"$\alpha$ value")                   # maths text
    ax[0].set_title("title", fontsize=S(12))
    ax[0].text(0.1, 0.1, "n = 3", fontsize=S(4), transform=ax[0].transAxes)
    ax[0].annotate("note", (1, 1), xytext=(2, 1.5), fontsize=S(20), arrowprops={"arrowstyle": "->"})
    im = ax[1].imshow(np.arange(9).reshape(3, 3))
    cb = fig.colorbar(im, ax=ax[1]); cb.set_label("colour", fontsize=S(15))
    ax[1].set_xticks([0, 1, 2], ["a", "b", "c"], fontsize=S(11))
    ax[1].tick_params(axis="y", labelsize=S(5))           # after ticks exist
    ax[1].bar([0, 1], [1, 2])
    fig.suptitle("suptitle", fontsize=S(18))
    fig.text(0.5, 0.01, "figure text", fontsize=S(3), family="DejaVu Sans" if size is None else "Arial")
    return fig

# without the step, the busy figure must not already be all Arial 8 (else the test proves nothing)
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import Collection
from matplotlib.image import AxesImage
def not_text(fig):
    """Data, colour and width of every line, patch, collection and image (tick marks left out)."""
    on_axis = {id(a) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for a in axis.findobj()}
    out = []
    for a in fig.findobj(lambda a: isinstance(a, (Line2D, Patch, Collection, AxesImage))):
        if id(a) in on_axis:
            continue
        if isinstance(a, Line2D):
            out.append(("line", a.get_xydata().tobytes(), str(a.get_color()), a.get_linewidth(), str(a.get_marker())))
        elif isinstance(a, Patch):
            out.append(("patch", a.get_path().vertices.tobytes(), a.get_patch_transform().get_matrix().tobytes(),
                        str(a.get_facecolor()), str(a.get_edgecolor()), a.get_linewidth()))
        elif isinstance(a, Collection):
            out.append(("collection", np.asarray(a.get_offsets()).tobytes(), str(a.get_facecolor().tolist()),
                        str(np.asarray(a.get_linewidths()).tolist())))
        else:
            out.append(("image", np.asarray(a.get_array()).tobytes()))
    return out
def tick_labels(svg):
    """The text of every plain text element (tick labels and the rest), sorted."""
    return sorted(re.findall(r"<text\b[^>]*>([^<]+)</text>", svg))

fig = busy_figure(); raw = svg_of(fig); plt.close(fig)
_, bad_raw = all_arial_8(raw)
check("test figure has other fonts and sizes before the step", len(bad_raw) >= 5, str(sorted(bad_raw)[:6]))
fig = busy_figure(size=8); native = svg_of(fig); plt.close(fig)
from matplotlib.patches import Patch as _Patch
# legend calls as the figure scripts make them; each is built without the step (RAW), at Arial 8 pt
# from the start (NATIVE, the same call with its fonts replaced), and with the step (below)
LEGEND_CASES = [
    ("ax.legend with fontsize 16, title_fontsize 14, loc, ncol, frameon, markerscale",
     "ax", dict(title="Title", fontsize=16, title_fontsize=14, loc="lower left", ncol=2, frameon=False, markerscale=2),
     dict(title="Title", prop={"family": "Arial", "size": 8}, title_fontsize=8, loc="lower left", ncol=2, frameon=False, markerscale=2)),
    ("ax.legend with a bold prop of size 5 and bbox_to_anchor",
     "ax", dict(prop={"size": 5, "weight": "bold"}, bbox_to_anchor=(1.0, 1.0), loc="upper left"),
     dict(prop={"family": "Arial", "size": 8, "weight": "bold"}, title_fontsize=8, bbox_to_anchor=(1.0, 1.0), loc="upper left")),
    ("ax.legend with title_fontproperties of size 20 and labelcolor",
     "ax", dict(title="T", title_fontproperties={"size": 20}, labelcolor="linecolor"),
     dict(title="T", title_fontproperties={"family": "Arial", "size": 8}, prop={"family": "Arial", "size": 8}, labelcolor="linecolor")),
    ("fig.legend with Patch handles, ncol and bbox_to_anchor",
     "fig", dict(fontsize=12, ncol=2, bbox_to_anchor=(0.5, 0.0), loc="lower center"),
     dict(prop={"family": "Arial", "size": 8}, title_fontsize=8, ncol=2, bbox_to_anchor=(0.5, 0.0), loc="lower center")),
]
def legend_figure(where, kw):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], color="tab:red", label="a line"); ax.scatter([0], [1], s=30, color="tab:blue", label="points")
    if where == "ax":
        leg = ax.legend(**kw)
    else:
        leg = fig.legend(handles=[_Patch(color="tab:green"), _Patch(color="tab:purple")], labels=["p", "q"], **kw)
    fig.canvas.draw()
    return fig, leg
def legend_props(leg):
    """Everything about a legend except its fonts."""
    hs = []
    for h in leg.legend_handles:
        c = h.get_color() if hasattr(h, "get_color") else h.get_facecolor()
        hs.append((type(h).__name__, str(np.asarray(matplotlib.colors.to_rgba_array(c)).round(3).tolist())))
    return (hs, [t.get_text() for t in leg.get_texts()], [str(matplotlib.colors.to_rgba(t.get_color())) for t in leg.get_texts()],
            leg.get_title().get_text(), leg._loc, leg._ncols, leg.get_frame_on(), leg.markerscale,
            tuple(round(v, 3) for v in leg.get_bbox_to_anchor().bounds),
            [(t.get_fontweight(), t.get_fontstyle()) for t in leg.get_texts()])
def legend_geometry(leg):
    fr = leg.get_frame().get_window_extent()
    hs = [h.get_window_extent() for h in leg.legend_handles if hasattr(h, "get_window_extent")]
    return ((round(fr.x0, 1), round(fr.y0, 1), round(fr.width, 1), round(fr.height, 1)),
            [(round(h.x0, 1), round(h.y0, 1), round(h.width, 1)) for h in hs])
RAW_LEGEND, NATIVE_LEGEND = {}, {}
for label, where, kw, kw8 in LEGEND_CASES:
    _f, _l = legend_figure(where, kw); RAW_LEGEND[label] = legend_props(_l); plt.close(_f)
    _f, _l = legend_figure(where, kw8); NATIVE_LEGEND[label] = legend_geometry(_l); plt.close(_f)
plt.rcParams.update(matplotlib.rcParamsDefault); matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "check_figure_text"; matplotlib.rcParams["pdf.compression"] = 0
matplotlib.use("Agg")
sns.set_context("notebook", font_scale=1)

ft.use_arial_8()

fig = busy_figure()
svg = svg_of(fig)
f, bad = all_arial_8(svg)
check("every text is Arial 8 pt (Figure.savefig)", not bad and texts_in(svg) >= 25,
      "%d texts; not Arial 8: %s" % (texts_in(svg), dict(bad)))
check("the ticks are those of the figure drawn at 8 pt from the start", tick_labels(svg) == tick_labels(native),
      "%d texts here, %d at 8 pt from the start" % (len(tick_labels(svg)), len(tick_labels(native))))
plt.close(fig)
fig = busy_figure(); before = not_text(fig); ft.set_figure_text(fig); after = not_text(fig); plt.close(fig)
check("the step changes only text (lines, patches, collections, image)", before == after and len(before) >= 10,
      "%d artists; %d differ" % (len(before), sum(x != y for x, y in zip(before, after))))

fig = busy_figure(); b = io.StringIO(); plt.savefig(b, format="svg"); plt.close(fig)
f, bad = all_arial_8(b.getvalue())
check("plt.savefig goes through the step", not bad, str(dict(bad)))

fig = busy_figure(); fig.tight_layout(); svg = svg_of(fig, bbox_inches="tight"); plt.close(fig)
f, bad = all_arial_8(svg)
check("tight_layout and bbox_inches='tight' keep Arial 8", not bad, str(dict(bad)))

import pandas as pd
df = pd.DataFrame({"x": np.arange(20) % 5, "y": np.arange(20), "g": ["p", "q"] * 10})
sns.set_context("talk")
g = sns.relplot(data=df, x="x", y="y", hue="g", col="g", height=2)
b = io.StringIO(); g.savefig(b, format="svg"); plt.close("all")
f, bad = all_arial_8(b.getvalue())
check("seaborn grid saved with its own savefig", not bad and texts_in(b.getvalue()) >= 10, str(dict(bad)))
sns.set_context("notebook")

fig, ax = plt.subplots()
t = ax.table(cellText=[["a", "b"], ["c", "d"]], loc="center"); t.set_fontsize(14)
t2 = ax.table(cellText=[["e"]], loc="top")
svg = svg_of(fig); plt.close(fig)
f, bad = all_arial_8(svg)
check("table cells are Arial 8 pt (fixed and automatic size)", not bad and texts_in(svg) >= 5, "%d texts; %s" % (texts_in(svg), dict(bad)))

_saved_math = {k: matplotlib.rcParams[k] for k in ("mathtext.fontset", "mathtext.rm", "mathtext.it", "mathtext.bf", "mathtext.sf")}
matplotlib.rcParams.update({"mathtext.fontset": "dejavusans", "mathtext.rm": "DejaVu Sans", "mathtext.it": "DejaVu Sans:italic",
                            "mathtext.bf": "DejaVu Sans:bold", "mathtext.sf": "DejaVu Sans"})   # as rcdefaults() would
fig, ax = plt.subplots(); ax.set_xlabel(r"$\alpha$ and $r=0.50,\ p<0.01$"); svg = svg_of(fig); plt.close(fig)
f, bad = all_arial_8(svg)
check("maths text stays Arial after the script changes the maths font", not bad and "<tspan" in svg, str(dict(bad)))
matplotlib.rcParams.update(_saved_math)

for label, where, kw, kw8 in LEGEND_CASES:
    _f, _l = legend_figure(where, kw)
    props, geo = legend_props(_l), legend_geometry(_l)
    fonts_ok = all(t.get_fontsize() == 8 and t.get_fontfamily()[0] == "Arial" for t in _l.get_texts() + [_l.get_title()])
    plt.close(_f)
    check("%s: same as without the step except fonts" % label, props == RAW_LEGEND[label],
          "" if props == RAW_LEGEND[label] else "%s vs %s" % (props, RAW_LEGEND[label]))
    check("%s: Arial 8 pt with the frame and handles of the legend built at 8 pt" % label, fonts_ok and geo == NATIVE_LEGEND[label],
          "fonts %s; %s vs %s at 8 pt" % ("ok" if fonts_ok else "wrong", geo, NATIVE_LEGEND[label]))

fig = busy_figure()
b = io.BytesIO()
with PdfPages(b) as pdf:
    pdf.savefig(fig)
plt.close(fig)
sizes = set(re.findall(rb"/F\d+ ([\d.]+) Tf", b.getvalue()))
check("PdfPages goes through the step (every font size in the PDF is 8)", sizes == {b"8"}, str(sizes))

fig = busy_figure(); once = svg_of(fig); plt.close(fig)
ft.use_arial_8()
fig = busy_figure(); twice = svg_of(fig); plt.close(fig)
check("calling use_arial_8() twice changes nothing", once == twice)

for name, ok, detail in results:
    print("%s\t%s\t%s" % ("PASS" if ok else "FAIL", name, detail))
'''

# run_figure.py replaces Figure.savefig before a script runs; the script's use_arial_8() wraps it.
CHILD_RUN_FIGURE = r'''
import sys, os, re, tempfile, pathlib
root, module_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
script = pathlib.Path(out) / "panel.py"
script.write_text(
    "import sys, importlib.util\n"
    "import matplotlib as mpl\n"
    "mpl.rcParams['svg.fonttype'] = 'none'\n"
    "spec = importlib.util.spec_from_file_location('figure_text', %r)\n"
    "ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft); ft.use_arial_8()\n"
    "import matplotlib.pyplot as plt\n"
    "fig, ax = plt.subplots(); ax.plot([0, 1]); ax.set_xlabel('x', fontsize=16)\n"
    "ax.tick_params(labelsize=13)\n"
    "plt.savefig('panel.svg')\n" % module_path)
'''


def run_checks(module, workdir):
    """Run the checks against one copy of figure_text.py; return a list of (status, name, detail)."""
    py = sys.executable
    child = Path(workdir) / "child.py"
    child.write_text(CHILD)
    p = subprocess.run([py, str(child), str(module), str(ROOT)], capture_output=True, text=True, timeout=600)
    rows = [line.split("\t", 2) for line in p.stdout.splitlines() if line.startswith(("PASS\t", "FAIL\t"))]
    if p.returncode != 0 or not rows:
        rows.append(["FAIL", "the checks ran to the end", (p.stderr or "")[-600:]])

    out = Path(workdir) / "run_figure_out"
    out.mkdir(exist_ok=True)
    maker = Path(workdir) / "make_panel.py"
    maker.write_text(CHILD_RUN_FIGURE)
    subprocess.run([py, str(maker), str(ROOT), str(module), str(out)], check=True, timeout=60)
    r = subprocess.run([py, str(ROOT / "tools" / "run_figure.py"), str(out / "panel.py"), "--out", str(out / "run")],
                       capture_output=True, text=True, timeout=300, cwd=str(out))
    saved = list((out / "run").rglob("panel.svg"))
    if saved:
        svg = saved[0].read_text()
        fonts = set(re.findall(r"font:\s*([^;\"]*?)\s*'([^']+)'", svg))
        ok = bool(fonts) and all(f == ("8px", "Arial") for f in fonts)
        rows.append(["PASS" if ok else "FAIL", "works under tools/run_figure.py", str(sorted(fonts))])
    else:
        rows.append(["FAIL", "works under tools/run_figure.py", "no panel.svg saved: " + (r.stderr or "")[-400:]])
    return rows


MUTANTS = [
    ("tick labels not set",
     '            axis.set_tick_params(which="both", labelsize=size, labelfontfamily=family)\n',
     "            pass\n"),
    ("family not set", "        text.set_fontfamily(family)\n", ""),
    ("size not set", "        text.set_fontsize(size)\n", ""),
    ("maths text left out", '    "mathtext.rm": FONT_FAMILY,\n', ""),
    ("maths italic left out", '    "mathtext.it": FONT_FAMILY + ":italic",\n', ""),
    ("maths settings not applied at the save",
     "            with matplotlib.rc_context(_MATH_RC):\n                return current(self, *args, **kwargs)\n",
     "            return current(self, *args, **kwargs)\n"),
    ("maths font of each text left", "        text.set_math_fontfamily(\"custom\")", "        pass"),
    ("table cells left out", "            cell.get_text().set_fontsize(size)\n", "            pass\n"),
    ("legends not rebuilt", "        matplotlib.legend.Legend.__init__ = __init__\n", ""),
    ("legend loc forced", "            legend_init(self, *args, **kwargs)\n", "            kwargs[\"loc\"] = \"lower left\"\n            legend_init(self, *args, **kwargs)\n"),
    ("legend frame switched off", "            legend_init(self, *args, **kwargs)\n", "            kwargs[\"frameon\"] = False\n            legend_init(self, *args, **kwargs)\n"),
    ("legend markerscale changed", "            legend_init(self, *args, **kwargs)\n", "            kwargs[\"markerscale\"] = 3\n            legend_init(self, *args, **kwargs)\n"),
    ("legend label colour changed", "            legend_init(self, *args, **kwargs)\n", "            kwargs[\"labelcolor\"] = \"red\"\n            legend_init(self, *args, **kwargs)\n"),
    ("legend columns and anchor dropped", "            legend_init(self, *args, **kwargs)\n", "            kwargs.pop(\"ncol\", None); kwargs.pop(\"bbox_to_anchor\", None)\n            legend_init(self, *args, **kwargs)\n"),
    ("legend bold dropped", "    fp.set_family(family)\n    fp.set_size(size)\n    return fp\n", "    fp = FontProperties(family=family, size=size)\n    return fp\n"),
    ("figure legends not rebuilt", "            legend_init(self, *args, **kwargs)\n", "            if isinstance(args[0], matplotlib.figure.Figure):\n                return legend_init(self, *args, **{k: v for k, v in kwargs.items() if k not in (\"prop\", \"title_fontsize\")})\n            legend_init(self, *args, **kwargs)\n"),
    ("legend title size left as set", "                kwargs[\"title_fontsize\"] = FONT_SIZE\n", "                pass\n"),
    ("text set after saving",
     "            set_figure_text(self)\n            with matplotlib.rc_context(_MATH_RC):\n                return current(self, *args, **kwargs)\n",
     "            with matplotlib.rc_context(_MATH_RC):\n                result = current(self, *args, **kwargs)\n            set_figure_text(self)\n            return result\n"),
    ("only fig.texts set", "    for text in fig.findobj(Text):\n", "    for text in fig.texts:\n"),
    ("savefig not replaced", "        matplotlib.figure.Figure.savefig = savefig\n", ""),
    ("size 7", "FONT_SIZE = 8\n", "FONT_SIZE = 7\n"),
    ("line widths changed as well", "        text.set_fontsize(size)\n",
     "        text.set_fontsize(size)\n    for line in fig.findobj(matplotlib.lines.Line2D):\n        line.set_linewidth(0.5)\n"),
]


SVG_FONT = re.compile(r"font:\s*([^;\"]*?)\s*'([^']+)'")
SVG_OK = re.compile(r"(?:(?:italic|bold|oblique)\s+)*8px")
# matplotlib draws maths exponents and subscripts at 0.7 of the text size; they are meant to be smaller
SMALL_MATHS = {("5.6px", "Arial"), ("italic 5.6px", "Arial"), ("oblique 5.6px", "Arial")}


def scan_outputs(folder):
    """Every text in the saved SVG and PDF figures under folder; return report lines and whether all are Arial 8."""
    import zlib
    lines, bad_files, counts = [], 0, {"svg": 0, "pdf": 0, "png": 0}
    for f in sorted(Path(folder).rglob("*")):
        ext = f.suffix.lower()[1:]
        if ext not in counts:
            continue
        counts[ext] += 1
        if ext == "png":
            lines.append("NOTE  %s: PNG, fonts cannot be read" % f.relative_to(folder))
            continue
        if ext == "svg":
            found = [(m.group(1).strip(), m.group(2)) for m in SVG_FONT.finditer(f.read_text(errors="replace"))]
            bad = sorted({x for x in found if x[1] != "Arial" or not SVG_OK.fullmatch(x[0])} - SMALL_MATHS)
            small = sum(x in SMALL_MATHS for x in found)
            if small:
                lines.append("NOTE  %s: %d maths exponents or subscripts at 5.6 pt (0.7 of 8)" % (f.relative_to(folder), small))
            n = len(found)
        else:
            data = f.read_bytes()
            streams = [data]
            for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
                try:
                    streams.append(zlib.decompress(m.group(1)))
                except zlib.error:
                    pass
            sizes = [x for st in streams for x in re.findall(rb"/F\d+ ([\d.]+) Tf", st)]
            names = set(re.findall(rb"/BaseFont /([^\s/>]+)", data))
            bad = sorted({"size %s" % x.decode() for x in sizes if float(x) != 8}
                         | {"font %s" % x.decode() for x in names if b"Arial" not in x})
            n = len(sizes)
        if not n:
            lines.append("NOTE  %s: no text" % f.relative_to(folder))
        elif bad:
            bad_files += 1
            lines.append("FAIL  %s: %d texts, not Arial 8: %s" % (f.relative_to(folder), n, bad[:6]))
    total = counts["svg"] + counts["pdf"]
    lines.append("%d SVG and %d PDF files read (%d PNG not readable); %d with text that is not Arial 8 pt"
                 % (counts["svg"], counts["pdf"], counts["png"], bad_files))
    return lines, total > 0 and bad_files == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mutants", action="store_true", help="also plant mistakes and require each to fail")
    ap.add_argument("--outputs", type=Path, help="read the saved figures under this folder instead")
    args = ap.parse_args()
    out_dir = Path(OUTPUT_DIR) / "tools" / "check_figure_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.outputs:
        lines, ok = scan_outputs(args.outputs)
        report = "outputs read: %s\n" % args.outputs.resolve() + "\n".join(lines) + "\n"
        print(report, end="")
        (out_dir / "outputs_report.txt").write_text(report)
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    lines, failed = [], False
    with tempfile.TemporaryDirectory() as tmp:
        for status, name, detail in run_checks(MODULE, tmp):
            lines.append("%s  %s  %s" % (status, name, detail))
            failed |= status != "PASS"
        if args.mutants:
            source = MODULE.read_text()
            lines.append("")
            lines.append("Planted mistakes (each must make a check FAIL):")
            for name, old, new in MUTANTS:
                if source.count(old) != 1:
                    lines.append("FAIL  mutant '%s': its text is not found once in figure_text.py" % name)
                    failed = True
                    continue
                mdir = Path(tmp) / ("mutant_" + re.sub(r"\W+", "_", name))
                mdir.mkdir()
                mutant = mdir / "figure_text.py"
                mutant.write_text(source.replace(old, new, 1))
                rows = run_checks(mutant, mdir)
                caught = [r[1] for r in rows if r[0] != "PASS"]
                ok = bool(caught)
                failed |= not ok
                lines.append("%s  mutant '%s' %s%s" % ("PASS" if ok else "FAIL", name,
                                                       "caught by: " if ok else "NOT caught",
                                                       "; ".join(caught)))
    report = "\n".join(lines) + "\n"
    print(report, end="")
    (out_dir / "report.txt").write_text(report)
    print("PASS" if not failed else "FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
