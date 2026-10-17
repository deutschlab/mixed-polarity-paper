"""Check that each figures/fig7/fig7*.py port draws, saves and writes exactly what the authors' original does.

The seven Figure 7 panel scripts figures/fig7/fig7e_..., fig7f_..., fig7g_..., fig7i_..., fig7k_...,
fig7s2a_... and fig7s2b_... are generated from the authors' original scripts by
tools/make_fig7_ports.py. The originals are not in this repository; give their folder with
--originals (or the environment variable FIG7_ORIGINALS).

Check 0 (static): tools/make_fig7_ports.py in check mode, so every port is exactly what that tool
makes from the originals (no hand edit, every fix applied). This is the only check that covers a
port whose inputs are missing.

Then, for each pair, the original and the port each run in their own process with matplotlib
drawing to no screen (Agg) and a time limit (--timeout, 1200 s). The original reads its Windows
paths from the folders given with --inputs, by file name (default: the --originals folder), and
runs with svg.fonttype 'none' and Arial set, the two settings the ports add, so text is laid out
alike on both sides; the port reads the files config.py names, with config.OUTPUT_DIR pointed
into a work folder. Nothing real is written: figure saves, to_feather and the other pandas and
pyarrow writers are recorded instead; any other file a script opens for writing (open, np.save
...) is recorded, and refused outside the work folder; and the size and date of every file
config.py names are compared before and after each run. Recorded, at every plt.show(), every
save and the end of the script, for every figure (after drawing its canvas): lines (data, colour,
style, width, markers), collections (offsets, paths or segments, face and edge colours, sizes,
widths, alpha, colour arrays), patches (outline, colours, hatch), images, texts (string,
position, colour, size), titles, axis labels, limits, scales, aspect, ticks and tick labels,
spines, legend texts and handle colours, figure size and figure-level texts, and the whole figure
rendered to SVG in memory (no date, fixed ids), which also carries layout, fonts, rotations and
legend placement; plus every printed line, every table written (columns, dtypes, rows, a content
hash and the config constant it goes to), every file written, and every file read (pandas,
pyarrow or numpy readers; as the config constant it stands for).

Comparisons for each pair:
  F  the end state of every figure, the printed lines, and the files read: original vs port;
  I  both sides read the same bytes for each file (exception: the fig7k port reads
     NEURON_TABLE_FTR, where the original read a copy that differs in a column it does not use);
  S  the port's saves: exactly the file names make_fig7_ports.EXPECTED_SAVES lists, each of the
     figure EXPECTED_SAVE_FIGS names, in OUTPUT_DIR/fig7/<port name>/, with format='svg' and no
     other option, rendering as SVG, with svg.fonttype 'none' and Arial in force;
  P  each figure the port saves equals what the original showed for that figure (the first
     plt.show() that includes it), artist by artist and as SVG bytes;
  T  tables written: same columns, dtypes, rows, content and destination; no other write and no
     file left in the work folder.
Numbers must agree to 1e-9 (relative and absolute) after rounding to 12 decimals, texts exactly.

Known, deliberate differences, applied to the original's run so the comparison is like for like:
  fig7S2b.py  the original stops on a KeyError (the cell-type table names its column root_id);
              the original's read of that table renames the column, as the port does in its code.
A pair where both sides stop because t2.ftr or filopodia_filtered.ftr is missing (fig7e) is
reported NOT RUN, not FAIL; it is compared as soon as the files are there. A run in which no pair
is compared fails, and so do an unknown --only name, a wrong --originals and a script that runs
out of time; each still writes the report.

--mutants shows the check can fail: it copies the ports to a scratch folder, plants one bug at a
time (MUTANTS below: a threshold, a reference line, the statistic, the order, a row dropped, a
legend label, the axes swapped, an excluded type kept, a save removed, the font setting, the
table destination, tick labels, colours, a save before the legend, the wrong figure saved, the
layout, label rotation and size, the legend position, save options, extra reads and writes, and
a hand edit) and requires each to FAIL with both scripts finished. PASS there means all 26 were
caught.

What a PASS does not show: that the original is the right analysis, or that it reproduces the
published panel (the originals ran on the authors' machine with their libraries; this compares
both scripts on this machine, so a library change moves both alike; library versions are in the
report). Writes made outside Python (by compiled code) are seen only through the before-and-after
comparison of the files config.py names. fig7e is covered only by check 0 until its inputs exist.

Run:
    .venv/bin/python tools/check_fig7_ports.py --originals <folder with the seven original scripts>
    .venv/bin/python tools/check_fig7_ports.py --originals <folder> --only fig7g.py
    .venv/bin/python tools/check_fig7_ports.py --originals <folder> --mutants
About a minute for the six runnable pairs, about 5 minutes for --mutants.
Writes outputs/check/check_fig7_ports/ (or --out): report.txt (mutants: mutants_report.txt),
records/ (what each run drew), work/ and mutants/ (scratch).
Exit code 0 on PASS, 1 on FAIL.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
from config import OUTPUT_DIR  # noqa: E402
import make_fig7_ports as mk  # noqa: E402

OUT_DIR = OUTPUT_DIR / "check" / "check_fig7_ports"
PORTS_DIR = REPO_ROOT / "figures" / "fig7"
RENAME_ROOT_ID = {"fig7S2b.py"}
NOT_SUPPLIED_OK = ("t2.ftr", "filopodia_filtered.ftr")
# (original, config constant) pairs whose two sides read different files on purpose: the fig7k port reads
# NEURON_TABLE_FTR where the original read a copy that differs only in a column it does not use
SAME_INPUT_EXCEPTIONS = {("fig7k.py", "NEURON_TABLE_FTR")}

SHIM = r'''
import builtins, os, sys, json, runpy, ntpath, hashlib, io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.colors import to_rgba
import pandas as pd

MODE, SCRIPT, RECORD, INPUTS, WORK, RENAME, READS, WRITES = sys.argv[1:9]
INPUTS, READS, WRITES = json.loads(INPUTS), json.loads(READS), json.loads(WRITES)
record = {"events": [], "tables": [], "reads": [], "read_files": [], "other_writes": [], "file_writes": []}
# the ports add exactly these two settings (make_fig7_ports.py); the original runs with them too, so that
# text is laid out with the same font on both sides and the saved SVGs can be compared byte for byte
if MODE == "original":
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = ["Arial"]
SVG_RC = {"svg.hashsalt": "check_fig7_ports"}
WORK = os.path.realpath(WORK)
CACHE = os.path.realpath(matplotlib.get_cachedir())

def audit(event, args):
    # every file opened for writing by Python code (open, os.open, np.save, pandas writers ...):
    # recorded, and refused outside the work folder, so a check never changes real files
    if event != "open" or not args or isinstance(args[0], int) or args[0] is None:
        return
    path, mode, flags = (list(args) + [None, 0])[:3]
    write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
        (flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
    if not write:
        return
    real = os.path.realpath(os.fsdecode(path))
    if "__pycache__" in real or real.startswith(CACHE) or real == os.path.realpath(RECORD):
        return
    record["file_writes"].append(os.path.relpath(real, WORK) if real.startswith(WORK) else "OUTSIDE " + real)
    if not real.startswith(WORK):
        raise PermissionError("check_fig7_ports refused a write outside its work folder: " + real)
if MODE == "port":
    import config
    CONST = {str(Path(v)): k for k, v in vars(config).items() if isinstance(v, Path)}
    config.OUTPUT_DIR = Path(WORK) / "outputs"
    FIG7_OUT = config.OUTPUT_DIR / "fig7"

def local(p):
    if MODE != "original" or not isinstance(p, str) or ":\\" not in p:
        return p
    name = ntpath.basename(p)
    for d in INPUTS:
        for c in (os.path.join(d, name), os.path.join(d, name + ".gz")):
            if os.path.exists(c):
                return c
    raise FileNotFoundError("MISSING INPUT " + name)

def read_name(p):
    if MODE == "original":
        return READS.get(ntpath.basename(str(p)), "UNMAPPED " + ntpath.basename(str(p)))
    return CONST.get(str(Path(p)), "NOT A CONFIG CONSTANT " + str(p))

def dest(p):
    if MODE == "original":
        return WRITES.get(ntpath.basename(str(p)), "UNMAPPED " + ntpath.basename(str(p)))
    return CONST.get(str(Path(p)), "NOT A CONFIG CONSTANT " + str(p))

_rf, _rc = pd.read_feather, pd.read_csv
_fsave = matplotlib.figure.Figure.savefig
_inside = [0]
def opened(p):
    q = local(p)
    if not os.path.exists(str(q)):
        raise FileNotFoundError("MISSING INPUT " + ntpath.basename(str(q)))
    return q
def recorded(reader, rename=False):
    # wraps a reader so each call records the file (as a config constant) once, even when one
    # recorded reader calls another (pd.read_feather calls pyarrow.feather.read_table)
    def f(p, *a, **k):
        if _inside[0]:
            return reader(p, *a, **k)
        record["reads"].append(read_name(p))
        q = opened(p)
        record["read_files"].append(str(q))
        _inside[0] += 1
        try:
            df = reader(q, *a, **k)
        finally:
            _inside[0] -= 1
        if rename and RENAME == "1" and MODE == "original":
            df = df.rename(columns={"root_id": "neuron"})
        return df
    return f
import pyarrow.feather, pyarrow.parquet
pd.read_feather, pd.read_csv = recorded(_rf), recorded(_rc, rename=True)
pd.read_parquet, pd.read_pickle = recorded(pd.read_parquet), recorded(pd.read_pickle)
pyarrow.feather.read_table = recorded(pyarrow.feather.read_table)
pyarrow.feather.read_feather = recorded(pyarrow.feather.read_feather)
pyarrow.parquet.read_table = recorded(pyarrow.parquet.read_table)
np.load = recorded(np.load)
def to_feather(self, p, *a, **k):
    h = hashlib.sha256(pd.util.hash_pandas_object(self.reset_index(drop=True), index=True).values.tobytes()).hexdigest()
    record["tables"].append({"columns": list(map(str, self.columns)), "dtypes": [str(t) for t in self.dtypes],
                             "rows": len(self), "hash": h, "dest": dest(p)})
pd.DataFrame.to_feather = to_feather
def blocked(kind):
    def f(self, p=None, *a, **k):
        record["other_writes"].append(kind + " " + ntpath.basename(str(p)))
    return f
for kind in ("to_csv", "to_pickle", "to_parquet", "to_excel", "to_json"):
    setattr(pd.DataFrame, kind, blocked(kind))
    setattr(pd.Series, kind, blocked("Series." + kind))
pyarrow.feather.write_feather = lambda df, p, *a, **k: record["other_writes"].append("pyarrow.write_feather " + ntpath.basename(str(p)))
pyarrow.parquet.write_table = lambda t, p, *a, **k: record["other_writes"].append("pyarrow.write_table " + ntpath.basename(str(p)))

def svg_sha(f, **k):
    # the figure rendered to SVG in memory, with no date and a fixed id salt, so equal figures give equal bytes
    buf = io.BytesIO()
    with matplotlib.rc_context(SVG_RC):
        _fsave(f, buf, **dict(k, format="svg", metadata={"Date": None}))
    return hashlib.sha256(buf.getvalue()).hexdigest()

def arr(x):
    try:
        return np.asarray(x, dtype=float).round(12).tolist()
    except Exception:
        return str(x)
def rgba(c):
    try:
        return arr(to_rgba(c))
    except Exception:
        return arr(c)

def snap_fig(f):
    f.canvas.draw()
    axes = []
    for ax in f.axes:
        a = {"title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
             "xlim": arr(ax.get_xlim()), "ylim": arr(ax.get_ylim()),
             "xscale": ax.get_xscale(), "yscale": ax.get_yscale(), "aspect": str(ax.get_aspect()),
             "xticks": arr(ax.get_xticks()), "yticks": arr(ax.get_yticks()),
             "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
             "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
             "spines": {k: bool(v.get_visible()) for k, v in ax.spines.items()},
             "lines": [{"xy": arr(l.get_xydata()), "color": rgba(l.get_color()), "ls": str(l.get_linestyle()),
                        "lw": float(l.get_linewidth()), "marker": str(l.get_marker()), "ms": float(l.get_markersize()),
                        "mfc": rgba(l.get_markerfacecolor()), "mec": rgba(l.get_markeredgecolor()),
                        "alpha": arr(l.get_alpha()), "visible": bool(l.get_visible())} for l in ax.lines],
             "collections": [], "patches": [], "images": [],
             "texts": [{"s": t.get_text(), "xy": arr(t.get_position()), "color": rgba(t.get_color()),
                        "size": float(t.get_fontsize()), "ha": t.get_ha(), "rot": float(t.get_rotation())} for t in ax.texts],
             "legend": None}
        for c in ax.collections:
            d = {"offsets": arr(c.get_offsets()), "facecolor": arr(c.get_facecolor()), "edgecolor": arr(c.get_edgecolor()),
                 "linewidths": arr(c.get_linewidths()), "alpha": arr(c.get_alpha()),
                 "array": arr(c.get_array()) if c.get_array() is not None else None}
            if hasattr(c, "get_sizes"):
                d["sizes"] = arr(c.get_sizes())
            if hasattr(c, "get_segments"):
                d["segments"] = [arr(s) for s in c.get_segments()]
            else:
                d["paths"] = [arr(p.vertices) for p in c.get_paths()]
            a["collections"].append(d)
        for p in ax.patches:
            v = p.get_patch_transform().transform(p.get_path().vertices)
            a["patches"].append({"verts": arr(v), "facecolor": arr(p.get_facecolor()), "edgecolor": arr(p.get_edgecolor()),
                                 "lw": float(p.get_linewidth()), "hatch": str(p.get_hatch())})
        for im in ax.images:
            A = np.asarray(im.get_array())
            a["images"].append({"shape": list(A.shape), "hash": hashlib.sha256(np.ascontiguousarray(A).tobytes()).hexdigest(),
                                "extent": arr(im.get_extent()), "cmap": im.get_cmap().name, "clim": arr(im.get_clim())})
        lg = ax.get_legend()
        if lg:
            a["legend"] = {"texts": [t.get_text() for t in lg.get_texts()],
                           "handles": [rgba(h.get_facecolor()) if hasattr(h, "get_facecolor") else rgba(h.get_color())
                                       for h in lg.legend_handles]}
        axes.append(a)
    return {"size": arr(f.get_size_inches()), "axes": axes,
            "suptitle": f._suptitle.get_text() if f._suptitle else "",
            "figtexts": [{"s": t.get_text(), "xy": arr(t.get_position())} for t in f.texts],
            "figlegend": [t.get_text() for lg in f.legends for t in lg.get_texts()]}

def savefig(self, fname, *a, **k):
    ev = {"kind": "savefig", "fig": self.number, "name": ntpath.basename(str(fname)),
          "format": str(k.get("format")), "fonttype": matplotlib.rcParams["svg.fonttype"],
          "family": list(matplotlib.rcParams["font.family"]), "snap": snap_fig(self),
          "kwargs": {key: repr(v) for key, v in sorted(k.items())}, "svg": svg_sha(self)}
    if MODE == "port":
        ev["in_fig7_folder"] = str(Path(fname).parent.parent) == str(FIG7_OUT)
        ev["folder"] = Path(fname).parent.name
        buf = io.BytesIO()
        _fsave(self, buf, *a, **k)
        ev["rendered_svg"] = buf.getvalue()[:200].lstrip().startswith(b"<?xml")
    record["events"].append(ev)
matplotlib.figure.Figure.savefig = savefig
def show(*a, **k):
    record["events"].append({"kind": "show", "figs": {str(n): snap_fig(plt.figure(n)) for n in plt.get_fignums()},
                             "svg": {str(n): svg_sha(plt.figure(n)) for n in plt.get_fignums()}})
plt.show = show
os.chdir(WORK)
sys.addaudithook(audit)
try:
    runpy.run_path(SCRIPT, run_name="__main__")
    record["final"] = {str(n): snap_fig(plt.figure(n)) for n in plt.get_fignums()}
    record["finished"] = True
except BaseException as e:
    record["finished"] = False
    record["error"] = repr(e)[:300]
json.dump(record, open(RECORD, "w"))
'''

# name, original, port file, exact text to change (must occur once), replacement
MUTANTS = [
    ("si_threshold", "fig7g.py", "fig7g_filopodia_fraction_x_si.py",
     "(bouton_table['SI'] >= 0.1)", "(bouton_table['SI'] >= 0.11)"),
    ("dashed_line", "fig7g.py", "fig7g_filopodia_fraction_x_si.py", "ax.axhline(0.24,", "ax.axhline(0.25,"),
    ("mean_to_median", "fig7f_fig7S4c.py", "fig7f_filopodia_fraction_violins.py",
     "filo_frac=('filopodia', 'mean'),", "filo_frac=('filopodia', 'median'),"),
    ("order_flipped", "fig7i.py", "fig7i_filopodia_fraction_x_nt.py",
     "    .mean()\n    .sort_values(ascending=False)\n    .index", "    .mean()\n    .sort_values(ascending=True)\n    .index"),
    ("table_row_dropped", "fig7i.py", "fig7i_filopodia_fraction_x_nt.py",
     "neuron_frac.to_feather(NEURONS_NT_BWF_FTR)", "neuron_frac.iloc[1:].reset_index(drop=True).to_feather(NEURONS_NT_BWF_FTR)"),
    ("legend_label", "fig7k.py", "fig7k_reciprocal_composition.py",
     "\"Local reciprocal\": \"'Hand Shake'\",", "\"Local reciprocal\": \"'Handshake'\","),
    ("axes_swapped", "fig7S2a.py", "fig7s2a_bouton_spread.py",
     "    neuron_quality['mean_rms'],\n    neuron_quality['mean_centroid_dist'],",
     "    neuron_quality['mean_centroid_dist'],\n    neuron_quality['mean_rms'],"),
    ("type_unexcluded", "fig7S2b.py", "fig7s2b_bouton_spread_by_type.py", "    'Lai',\n    'LTe26',\n    'Mi9',", "    'Lai',\n    'LTe26',"),
    ("save_removed", "fig7S2a.py", "fig7s2a_bouton_spread.py", "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\n", ""),
    ("fonttype_removed", "fig7S2a.py", "fig7s2a_bouton_spread.py", "mpl.rcParams['svg.fonttype'] = 'none'\n", ""),
    ("table_written_elsewhere", "fig7i.py", "fig7i_filopodia_fraction_x_nt.py",
     "neuron_frac.to_feather(NEURONS_NT_BWF_FTR)", "neuron_frac.to_feather(_out_dir / 'neurons_nt_bwf_frac.ftr')"),
    ("xticklabels_reversed", "fig7i.py", "fig7i_filopodia_fraction_x_nt.py",
     "    [nt_labels[n] for n in order_rest],", "    [nt_labels[n] for n in order_rest][::-1],"),
    ("bar_labels_reversed", "fig7k.py", "fig7k_reciprocal_composition.py",
     'ax.set_xticklabels(percentages.index, rotation=45, ha="right")',
     'ax.set_xticklabels(percentages.index[::-1], rotation=45, ha="right")'),
    ("line_colours_swapped", "fig7g.py", "fig7g_filopodia_fraction_x_si.py",
     "    label='Axon',\n    color='#8B2BE2'", "    label='Axon',\n    color='#F4B95A'"),
    ("saved_before_legend", "fig7g.py", "fig7g_filopodia_fraction_x_si.py",
     "plt.legend()\nplt.tight_layout()\n\n# ----------------------------------------------------\n# 12. Save & show\n"
     "# ----------------------------------------------------\nplt.savefig(\n    _out_dir / 'SI_vs_fraction_all_new.svg',\n    format='svg'\n)\n",
     "plt.savefig(\n    _out_dir / 'SI_vs_fraction_all_new.svg',\n    format='svg'\n)\nplt.legend()\nplt.tight_layout()\n"),
    ("extra_read", "fig7S2a.py", "fig7s2a_bouton_spread.py",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\n",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\nfrom config import NEURON_DATA_FTR\n_unused = pd.read_feather(NEURON_DATA_FTR)\n"),
    ("extra_write", "fig7S2a.py", "fig7s2a_bouton_spread.py",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\n",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\nneuron_quality.to_csv(_out_dir / 'inertia.csv')\n"),
    ("wrong_figure_saved", "fig7g.py", "fig7g_filopodia_fraction_x_si.py",
     "plt.savefig(\n    _out_dir / 'SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg',",
     "plt.figure(1).savefig(\n    _out_dir / 'SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg',"),
    ("tight_layout_removed", "fig7k.py", "fig7k_reciprocal_composition.py",
     'plt.title("Reciprocal composition by super class")\nplt.tight_layout()\n',
     'plt.title("Reciprocal composition by super class")\n'),
    ("ticklabel_rotation", "fig7k.py", "fig7k_reciprocal_composition.py", 'rotation=45, ha="right")', 'rotation=90, ha="right")'),
    ("legend_moved", "fig7g.py", "fig7g_filopodia_fraction_x_si.py",
     "plt.legend()\nplt.tight_layout()\n\n# ----------------------------------------------------\n# 12.",
     "plt.legend(loc='upper left')\nplt.tight_layout()\n\n# ----------------------------------------------------\n# 12."),
    ("label_fontsize", "fig7S2a.py", "fig7s2a_bouton_spread.py", "ax.set_xlabel('Mean RMS')", "ax.set_xlabel('Mean RMS', fontsize=4)"),
    ("savefig_options", "fig7S2a.py", "fig7s2a_bouton_spread.py",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\n",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg', facecolor='red', bbox_inches='tight')\n"),
    ("side_writes", "fig7S2a.py", "fig7s2a_bouton_spread.py",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\n",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\nimport numpy as _np\n"
     "_np.save(_out_dir / 'side.npy', neuron_quality['mean_rms'].values)\nopen(_out_dir / 'side.txt', 'w').write('x')\n"),
    ("extra_read_pyarrow", "fig7S2a.py", "fig7s2a_bouton_spread.py",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\n",
     "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\nimport pyarrow.feather as _pf\nfrom config import NEURON_DATA_FTR\n"
     "_unused = _pf.read_table(str(NEURON_DATA_FTR)).to_pandas()\n"),
]
# a hand edit that changes no output: only check 0 can catch it
STATIC_MUTANT = ("hand_edit", "fig7k_reciprocal_composition.py", "import sys\n", "import sys\n# edited by hand\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_files() -> dict:
    """Size and modification time of every file a config.py path names (checked before and after each run)."""
    import config
    out = {}
    for k, v in vars(config).items():
        if k.isupper() and isinstance(v, Path) and v.is_file():
            st = v.stat()
            out[k] = (st.st_size, st.st_mtime_ns)
    return out


def run(mode: str, script: Path, rec: Path, inputs: list, rename: bool, timeout: int) -> dict:
    work = rec.parent.parent / "work" / rec.stem
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0", PYTHONPATH=str(REPO_ROOT))
    if rec.exists():
        rec.unlink()
    before = data_files()
    try:
        p = subprocess.run([sys.executable, "-c", SHIM, mode, str(script), str(rec), json.dumps([str(i) for i in inputs]),
                            str(work), "1" if rename else "0", json.dumps(mk.READS), json.dumps(mk.WRITES)],
                           capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"finished": False, "error": "stopped after the %d-second time limit" % timeout, "printed": []}
    data = json.loads(rec.read_text()) if rec.exists() else {"finished": False, "error": p.stderr[-300:]}
    data["printed"] = [line for line in p.stdout.splitlines() if line.strip()]
    after = data_files()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if changed:  # a data file changed during the run: never a pass
        data["finished"] = False
        data["error"] = "a data file config.py names changed during the run: %s" % changed
    left = sorted(str(f.relative_to(work)) for f in work.rglob("*") if f.is_file())
    data["work_files"] = left
    return data


def has_str(x) -> bool:
    return isinstance(x, str) or (isinstance(x, list) and any(has_str(i) for i in x))


def diff(a, b, path="", out=None) -> list:
    out = [] if out is None else out
    if len(out) >= 8:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append("%s/%s only on one side" % (path, k))
            else:
                diff(a[k], b[k], path + "/" + k, out)
    elif isinstance(a, list) and isinstance(b, list) and not has_str(a) and not has_str(b):
        try:
            x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        except (ValueError, TypeError):
            x = y = None
        if x is not None:
            if x.shape != y.shape:
                out.append("%s: shape %s vs %s" % (path, x.shape, y.shape))
            elif not np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True):
                d = np.abs(np.nan_to_num(x - y)).ravel()
                i = int(np.argmax(d))
                out.append("%s: largest difference %.3g (%r vs %r)" % (path, d[i], x.ravel()[i], y.ravel()[i]))
            return out
        if len(a) != len(b):
            out.append("%s: %d vs %d items" % (path, len(a), len(b)))
        for i, (u, v) in enumerate(zip(a, b)):
            diff(u, v, "%s[%d]" % (path, i), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s: %d vs %d items" % (path, len(a), len(b)))
        for i, (u, v) in enumerate(zip(a, b)):
            diff(u, v, "%s[%d]" % (path, i), out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True):
            out.append("%s: %r vs %r" % (path, a, b))
    elif a != b:
        out.append("%s: %r vs %r" % (path, str(a)[:80], str(b)[:80]))
    return out


_SHA_CACHE = {}


def cached_sha(path: str) -> str:
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = sha256(Path(path))
    return _SHA_CACHE[key]


def compare(orig: str, port_name: str, a: dict, b: dict) -> list:
    problems = ["F " + p for p in diff({"final": a["final"], "printed": a["printed"], "reads": a["reads"]},
                                       {"final": b["final"], "printed": b["printed"], "reads": b["reads"]})]
    # I: both sides read the same bytes for each file, apart from the stated exceptions
    if a["reads"] == b["reads"]:
        for const, fa, fb in zip(a["reads"], a["read_files"], b["read_files"]):
            if (orig, const) not in SAME_INPUT_EXCEPTIONS and cached_sha(fa) != cached_sha(fb):
                problems.append("I %s: the original read %s and the port %s, which differ" % (const, fa, fb))
    problems += ["T " + p for p in diff(a["tables"], b["tables"], "/tables")]
    for what in ("other_writes", "file_writes", "work_files"):
        if b[what] != a[what] or b[what]:
            problems.append("T %s: original %s, port %s" % (what.replace("_", " "), a[what][:5], b[what][:5]))
    saves = [e for e in b["events"] if e["kind"] == "savefig"]
    names = [e["name"] for e in saves]
    if names != mk.EXPECTED_SAVES[orig]:
        problems.append("S saves %s, expected %s" % (names, mk.EXPECTED_SAVES[orig]))
    for e in saves:
        if not e["in_fig7_folder"] or e["folder"] != port_name[:-3]:
            problems.append("S %s saved into folder %s, not OUTPUT_DIR/fig7/%s" % (e["name"], e["folder"], port_name[:-3]))
        if e["format"] != "svg" or not e["rendered_svg"]:
            problems.append("S %s format %s, renders as svg: %s" % (e["name"], e["format"], e["rendered_svg"]))
        if e["fonttype"] != "none" or "Arial" not in e["family"]:
            problems.append("S %s saved with svg.fonttype=%s font.family=%s" % (e["name"], e["fonttype"], e["family"]))
        if e["kwargs"] != {"format": "'svg'"}:
            problems.append("S %s saved with options %s; the originals pass only format='svg'" % (e["name"], e["kwargs"]))
        want = mk.EXPECTED_SAVE_FIGS.get(e["name"])
        if e["fig"] != want:
            problems.append("S %s saves figure %s, expected figure %s" % (e["name"], e["fig"], want))
        shown = next((s for s in a["events"] if s["kind"] == "show" and str(want) in s["figs"]), None)
        if shown is None:
            problems.append("P %s: the original never showed figure %s" % (e["name"], want))
            continue
        problems += ["P %s " % e["name"] + p for p in diff(shown["figs"][str(want)], e["snap"], "/fig%s" % want)]
        if shown["svg"][str(want)] != e["svg"]:
            problems.append("P %s: the saved SVG is not byte for byte what the original showed as figure %s" % (e["name"], want))
    return problems


def not_supplied(d: dict) -> bool:
    err = str(d.get("error", ""))
    return "MISSING INPUT" in err and any(n in err for n in NOT_SUPPLIED_OK)


def check_pairs(originals: Path, inputs: list, ports: Path, only, out_dir: Path, timeout: int) -> tuple:
    """Run and compare each pair. Return (report lines, number failed, number compared, input files read)."""
    lines, failed, compared, read_files = [], 0, 0, set()
    rec_dir = out_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for orig, port_name in mk.PAIRS.items():
        if only and orig not in only:
            continue
        port = ports / port_name
        if not port.exists():
            lines.append("FAIL     %-18s port not found: %s" % (orig, port))
            failed += 1
            continue
        try:
            a = run("original", originals / orig, rec_dir / (orig + ".original.json"), inputs, orig in RENAME_ROOT_ID, timeout)
            b = run("port", port, rec_dir / (orig + ".port.json"), inputs, False, timeout)
            read_files |= set(a.get("read_files", [])) | set(b.get("read_files", []))
            if not a.get("finished") and not b.get("finished") and not_supplied(a) and not_supplied(b):
                lines.append("NOT RUN  %-18s an input is not supplied: %s" % (orig, a.get("error", "")[:120]))
                continue
            problems = ["%s did not finish: %s" % (side, d.get("error"))
                        for side, d in (("original", a), ("port", b)) if not d.get("finished")]
            problems = problems or compare(orig, port_name, a, b)
            compared += 1
        except Exception as e:  # a crash in the comparison is a failure, never a silent pass
            problems = ["the check itself stopped: %r" % e]
        sha = sha256(port)[:12]
        if problems:
            failed += 1
            lines.append("FAIL     %-18s %s (sha %s)" % (orig, port_name, sha))
            lines += ["           " + p[:220] for p in problems[:8]]
        else:
            note = "; root_id renamed for the original too" if orig in RENAME_ROOT_ID else ""
            lines.append("PASS     %-18s %s (sha %s): %d figure(s), %d save(s) byte-identical, %d table(s), %d read(s), %d printed line(s)%s"
                         % (orig, port_name, sha, len(b["final"]), len(mk.EXPECTED_SAVES[orig]), len(b["tables"]),
                            len(b["reads"]), len(b["printed"]), note))
    return lines, failed, compared, read_files


def versions() -> str:
    import matplotlib
    import pandas
    import seaborn
    import scipy
    return "python %s, numpy %s, pandas %s, matplotlib %s, seaborn %s, scipy %s" % (
        sys.version.split()[0], np.__version__, pandas.__version__, matplotlib.__version__, seaborn.__version__, scipy.__version__)


def run_mutants(originals: Path, inputs: list, ports: Path, out_dir: Path, timeout: int) -> list:
    lines, missed = [], 0
    root = out_dir / "mutants"
    if root.exists():
        shutil.rmtree(root)
    for name, orig, port_name, old, new in MUTANTS:
        d = root / name
        shutil.copytree(ports, d / "ports", ignore=shutil.ignore_patterns("__pycache__"))
        f = d / "ports" / port_name
        src = f.read_text(encoding="utf-8")
        if src.count(old) != 1:
            lines.append("BAD MUTANT %-24s the change matches %d times" % (name, src.count(old)))
            missed += 1
            continue
        f.write_text(src.replace(old, new), encoding="utf-8")
        report, failed, _, _ = check_pairs(originals, inputs, d / "ports", [orig], d, timeout)
        first = next((x.strip() for x in report if x.startswith("           ")), "")
        caught = failed == 1 and first and "did not finish" not in first and "stopped" not in first
        missed += not caught
        lines.append("%-7s %-24s %s" % ("caught" if caught else "MISSED", name, first[:160]))
    name, port_name, old, new = STATIC_MUTANT
    d = root / name / "ports"
    shutil.copytree(ports, d, ignore=shutil.ignore_patterns("__pycache__"))
    src = (d / port_name).read_text(encoding="utf-8")
    (d / port_name).write_text(src.replace(old, new, 1), encoding="utf-8")
    static = mk.check(originals, d, quiet=True)
    caught = bool(static) and all(port_name in p for p in static)
    missed += not caught
    lines.append("%-7s %-24s %s" % ("caught" if caught else "MISSED", name, (static or ["check 0 passed"])[0][:160]))
    n = len(MUTANTS) + 1
    lines.append("PASS: all %d mutants caught" % n if not missed else "FAIL: %d of %d mutants not caught" % (missed, n))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--originals", help="folder with the seven original scripts (default: $FIG7_ORIGINALS)")
    parser.add_argument("--inputs", nargs="+", type=Path,
                        help="folders the originals' input files are looked up in, by file name (default: --originals)")
    parser.add_argument("--ports", type=Path, default=PORTS_DIR, help="folder of the ports (default figures/fig7)")
    parser.add_argument("--only", nargs="+", help="original script names, e.g. fig7g.py (default: all seven)")
    parser.add_argument("--mutants", action="store_true", help="plant the built-in bugs and require each to FAIL")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="report folder (default outputs/check/check_fig7_ports)")
    parser.add_argument("--timeout", type=int, default=1200, help="seconds each script may run (default 1200)")
    args = parser.parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / ("mutants_report.txt" if args.mutants else "report.txt")
    if report.exists():
        report.unlink()
    head = ["Figure 7 port check, %s" % datetime.datetime.now().isoformat(timespec="seconds")]
    try:
        unknown = sorted(set(args.only or []) - set(mk.PAIRS))
        if unknown:
            raise SystemExit("FAIL: --only names no original script: %s (they are %s)" % (unknown, ", ".join(mk.PAIRS)))
        originals = mk.resolve_originals(args.originals).resolve()
        inputs = [i.resolve() for i in (args.inputs or [originals])]
        ports = args.ports.resolve()
        head += ["ports: %s; originals: %s; inputs: %s" % (ports, originals, ", ".join(map(str, inputs))), versions()]
        head += ["tool     %-36s sha256 %s" % (Path(f).name, sha256(Path(f))) for f in (__file__, mk.__file__)]
        head += ["original %-36s sha256 %s" % (o, sha256(originals / o)) for o in mk.PAIRS]
        head += ["port     %-36s sha256 %s" % (p, sha256(ports / p)) for p in mk.PAIRS.values() if (ports / p).exists()]
        if args.mutants:
            lines = run_mutants(originals, inputs, ports, out_dir, args.timeout)
            ok = lines[-1].startswith("PASS")
        else:
            static = mk.check(originals, ports, quiet=True)
            lines = ["PASS     check 0: every port is exactly what tools/make_fig7_ports.py makes" if not static else
                     "FAIL     check 0: " + "; ".join(static)[:400]]
            pair_lines, failed, compared, read_files = check_pairs(originals, inputs, ports, args.only, out_dir, args.timeout)
            lines += pair_lines
            lines += ["input    %s sha256 %s" % (f, cached_sha(f)) for f in sorted(read_files) if Path(f).exists()]
            if not compared:
                lines.append("FAIL     no pair was compared")
            ok = not static and not failed and compared > 0
            lines.append("PASS" if ok else "FAIL")
    except SystemExit as e:
        lines, ok = [str(e.code)], False
    except Exception as e:
        lines, ok = ["FAIL: the check stopped: %r" % e], False
    report.write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    print("\n".join(head + lines))
    print("Report:", report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # Reads the seven original Figure 7 scripts and the tables they read from the folder given with
    # --originals (not in this repository; --inputs names other folders for the tables), and
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/figures/fig7/  (the seven fig7*.py ports)
    #   the tables config.py names for Figure 7 (read by the ports)
    # Writes:
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/check_fig7_ports/report.txt  (one line per pair)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/check_fig7_ports/mutants_report.txt  (with --mutants)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/check_fig7_ports/records/  (what each run drew)
    # To run it, uncomment:
    # sys.argv[1:] = [
    #     "--originals", "<folder with the seven original scripts>",
    #     "--mutants",
    # ]
    main()
