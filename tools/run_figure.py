"""Run one figure script so that its output can be compared between runs.

The script runs as it is. What changes is how matplotlib, pandas and the random
number generators behave:
- no plot window opens; plt.show() closes the figures, as Spyder's inline plots do;
- every figure is saved into one output folder, under its own file name;
- SVG files carry no date and PDF files a fixed one (1 Jan 1970), and SVG files use
  fixed internal ids, so the same figure gives the same file;
- random numbers are seeded (numpy, Python's random, and the generators seaborn uses
  for its error bands), so repeated runs draw the same values;
- printed tables show all their columns, and each is followed by a checksum of its
  full contents (values, index, column names and types), so a change in a row or
  decimal that is not shown still changes the printed output; a table holding other
  objects, such as neurons, gets no checksum, because their text differs between runs;
  numpy arrays print every element in full precision.
What the script prints goes to stdout.log in the output folder, and its warnings to
stderr.log. run_status.txt says whether the script finished; if it stopped with an
error, the file holds the traceback, which also appears in the terminal. Figures
are saved by file name alone, so two figures with the same name in different
folders overwrite each other (a warning is written to stderr.log).
Tables that a script writes (for example into data/derived) still go to their
usual place and are not compared.

Usage:
    python tools/run_figure.py figures/fig1/canonicality_axon_dend.py
    python tools/run_figure.py figures/fig1/canonicality_axon_dend.py \
        --dpi 72 --out outputs/check/fig1e

The published SVGs were saved at 72 dpi, the default for plots shown inside
Spyder. A plain script run uses 100 dpi, which gives a slightly different plot
area and sometimes different axis ticks, with the same data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import OUTPUT_DIR  # noqa: E402


class _Tee:
    """Write everything to several streams at once."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name: str):
        return getattr(self.streams[0], name)


def _configure_matplotlib(out_dir: Path, dpi: float | None) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # fixed date in PDF files
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "run_figure"
    if dpi is not None:
        matplotlib.rcParams["figure.dpi"] = dpi

    import matplotlib.figure
    import matplotlib.pyplot as plt

    original_savefig = matplotlib.figure.Figure.savefig
    saved: set[str] = set()

    def savefig(self, fname, *args, **kwargs):
        if not isinstance(fname, (str, os.PathLike)):
            return original_savefig(self, fname, *args, **kwargs)
        name = Path(fname).name
        if name in saved:
            print(f"run_figure: {name} is saved more than once; only the last copy is kept",
                  file=sys.stderr)
        saved.add(name)
        kwargs["metadata"] = {**(kwargs.get("metadata") or {}), "Date": None}
        return original_savefig(self, out_dir / name, *args, **kwargs)

    matplotlib.figure.Figure.savefig = savefig
    plt.savefig = lambda fname, *args, **kwargs: plt.gcf().savefig(fname, *args, **kwargs)
    plt.show = lambda *args, **kwargs: plt.close("all")


def _configure_pandas() -> None:
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 10000)
    pd.set_option("display.max_colwidth", None)
    np.set_printoptions(threshold=sys.maxsize, floatmode="unique")

    # Long tables print only their first and last rows, and values to 6 decimals.
    # The checksum covers every value, so a change hidden by either still shows.
    for cls in (pd.DataFrame, pd.Series):
        if not getattr(cls.__repr__, "_with_checksum", False):
            cls.__repr__ = _with_checksum(cls.__repr__)


# Values whose text form is the same in every run. Any other object (a navis
# neuron, a fitted model) prints its memory address, so its checksum would differ
# between two identical runs.
_STABLE_TYPES = (str, bytes, int, float, bool, complex, type(None))


def _checksum(obj) -> str:
    import numpy as np
    import pandas as pd

    columns = obj.to_frame() if isinstance(obj, pd.Series) else obj
    for _, column in columns.items():
        if column.dtype == object and not all(
                isinstance(value, _STABLE_TYPES + (np.generic, pd.Timestamp, pd.Timedelta))
                or value is pd.NA for value in column):
            return "not available"
    row_hashes = pd.util.hash_pandas_object(obj, index=True).to_numpy()
    header = repr([(str(name), str(dtype)) for name, dtype in columns.dtypes.items()])
    return hashlib.sha1(header.encode("utf-8") + row_hashes.tobytes()).hexdigest()[:16]


def _with_checksum(original_repr):
    def __repr__(self) -> str:
        text = original_repr(self)
        try:
            digest = _checksum(self)
        except Exception:  # values that cannot be hashed
            digest = "not available"
        return f"{text}\n[{len(self)} rows, checksum {digest}]"

    __repr__._with_checksum = True
    return __repr__


def _seed_random() -> None:
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)
    default_rng = np.random.default_rng
    np.random.default_rng = lambda seed=None: default_rng(0 if seed is None else seed)


def run_figure(script: Path, out_dir: Path, dpi: float | None = 72,
               replacements: list[tuple[str, str]] | None = None) -> None:
    """Run a figure script with fixed, comparable output.

    out_dir must be empty or not exist yet, so that no file from an earlier run is
    mistaken for this run's output.
    replacements: (old, new) pairs of exact source text; each old text must occur
    exactly once. Use it to skip a step that is too slow to run, and say so when
    you report the result.
    """
    script = script.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; choose a new --out folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = script.read_text(encoding="utf-8-sig")
    for old, new in replacements or []:
        count = source.count(old)
        if count != 1:
            raise ValueError(f"text to replace occurs {count} times, not once: {old!r}")
        source = source.replace(old, new)

    _configure_matplotlib(out_dir, dpi)
    _configure_pandas()
    _seed_random()

    status = out_dir / "run_status.txt"
    status.write_text("started, not finished\n", encoding="utf-8")
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as out_log, \
            open(out_dir / "stderr.log", "w", encoding="utf-8") as err_log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, out_log), _Tee(stderr, err_log)
        try:
            namespace = {"__file__": str(script), "__name__": "__main__"}
            exec(compile(source, str(script), "exec"), namespace)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                status.write_text("stopped with an error\n" + traceback.format_exc(),
                                  encoding="utf-8")
                raise
        except BaseException:
            status.write_text("stopped with an error\n" + traceback.format_exc(),
                              encoding="utf-8")
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    status.write_text("finished\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("script", type=Path, help="figure script to run")
    parser.add_argument("--dpi", type=float, default=72,
                        help="figure dpi; 72 matches the published SVGs (default: 72)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder, empty or new (default: outputs/check/<script name>)")
    parser.add_argument("--replace", nargs=2, action="append", metavar=("OLD", "NEW"),
                        help="replace one exact piece of source text before running")
    args = parser.parse_args()

    out_dir = args.out or OUTPUT_DIR / "check" / args.script.stem
    run_figure(args.script, out_dir, args.dpi, [tuple(pair) for pair in args.replace or []])
    print(f"Output written to {out_dir}")


if __name__ == "__main__":
    main()
