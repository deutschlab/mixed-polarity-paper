"""Compare two full figure runs made on different split methods, panel by panel, by the numbers drawn.

Takes two output folders made by run_all_figures.py (one subfolder per script), for
example one run under SPLIT_METHOD = "sfc" and one under "maxsi", and reports for
every script:
- its run status on each side (finished; stopped with an error, with the last line of
  the error; timed out; no status yet; missing), and one of three flags: finished on
  both sides, finished on one side only, finished on neither side;
- for every saved SVG, the numbers drawn in its text. With svg.fonttype 'none' every
  label is an SVG <text> element, so the numbers can be read back from the file. Each
  number falls in one of four classes:
    n        a number right after "n=" or "n =" (sample sizes in titles and labels);
    data     a number with decimals or a % sign, outside the axis ticks (data labels,
             statistics, shares);
    integer  any other whole number outside the ticks (counts written on bars, years);
    tick     any number inside a matplotlib tick group (xtick_N / ytick_N), whole or
             not: these are the axes, reported separately and left out of the verdict.
  Within each class, the numbers are paired between the two sides by their position
  in document order, when both sides have the same count; the report then gives them
  side by side with the absolute difference. When the counts differ it says so and
  lists both, without pairing. A pair is marked when its label differs once the
  numbers are masked ("AD 12%" against "DD 12%"), or when the last text without a
  number before it differs (a pie wedge's name drawn before its share), because then
  position pairing has probably matched different things; such pairs are listed even
  when the value is the same. It also says whether the files are
  byte-identical. PDF and PNG files are listed (and checked for identical bytes), not read;
- the printed output in stdout.log: each line's numbers are masked, the masked lines
  of the two sides are aligned (difflib), and aligned lines whose numbers (or table
  checksums, as run_figure.py prints them) differ are listed, up to --max-lines per
  script. Lines found on one side only are counted.

Each figure file is matched to its panel from docs/figure_to_script_map.md where it can
be: a file named in a table row or in a note ("5H is `x.svg`") gets that panel; a file of
a script listed under one panel gets that panel; a file of a script listed under several
panels gets "one of: ..."; anything else is "not a mapped panel". This is best effort.

The summary at the top gives the number of scripts per flag, the figure files whose
numbers are the same on both sides (split-independent as drawn), those whose numbers
changed, those whose counts differ, and the 20 largest absolute changes among the
percentage labels (in percentage points).

Writes report.txt (readable) and cells.csv (one row per compared number) to --out,
by default outputs/tools/compare_split_figures/<label_a>_vs_<label_b>/.

Limits: it reads the drawn text, not the data behind it, so a value is only as precise
as its label (a bar of 41.96% labelled "42.0%" shows as 42.0), and a change in a bar
height, point or curve with no label is not seen. Position pairing assumes both sides
draw their labels in the same order; a panel whose groups are sorted by value can pair
different groups, which the masked-label check catches only when the labels differ in
their text. Mathematical text drawn with superscripts is read as plain digits.
It only reads files and never changes a run folder.

--selftest builds two small fake runs in a temporary folder with known differences,
requires the comparison to report exactly those, and then requires six planted
mistakes (wrong pairings, ticks not told apart, printed lines paired by index, a label
check that misses a reordered pie) to fail.

Usage:
    python tools/compare_split_figures.py RUN_A RUN_B --labels SFC MaxSI
    python tools/compare_split_figures.py RUN_A RUN_B --labels SFC MaxSI --out outputs/tools/compare_split_figures/x
    python tools/compare_split_figures.py --selftest

Exits with status 1 if the selftest fails, and stops with a usage error if a run folder
does not exist; otherwise it exits with 0, whatever the differences.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP_DOC = REPO_ROOT / "docs" / "figure_to_script_map.md"
READ_FIGURES = {".svg"}
LISTED_FIGURES = {".pdf", ".png"}
CLASSES = ("n", "data", "integer", "tick")
VERDICT_CLASSES = ("n", "data", "integer")

NUMBER = re.compile(r"(?<![\w.])(?P<num>[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+−]?\d+)?)"
                    r"(?P<pct>\s?%)?(?![\w]|\.\d)")
CHECKSUM = re.compile(r"checksum [0-9a-f]{8,}")
N_BEFORE = re.compile(r"\bn\s*=\s*$", re.IGNORECASE)
TICK_GROUP = re.compile(r"^[xy]tick_\d+$")


def to_float(text: str) -> float:
    return float(text.replace(",", "").replace("−", "-"))


# ----------------------------------------------------------------------------- run folders

def run_status(folder: Path | None) -> tuple[str, str]:
    """Return (status, last error line) of one script folder."""
    if folder is None:
        return "missing", ""
    path = folder / "run_status.txt"
    if not path.exists():
        return "no status (running or interrupted)", ""
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").split("\n")
             if line.strip()]
    if not lines:
        return "empty status", ""
    if lines[0] == "stopped with an error":
        return lines[0], lines[-1] if len(lines) > 1 else ""
    return lines[0], ""


def flag_of(status_a: str, status_b: str) -> str:
    done = (status_a == "finished") + (status_b == "finished")
    return ("finished on both sides", "finished on one side only", "finished on neither side")[2 - done]


def script_folders(run: Path) -> dict[str, Path]:
    return {p.name: p for p in run.iterdir() if p.is_dir() and ".partial." not in p.name}


def figure_files(folder: Path | None) -> dict[str, Path]:
    if folder is None:
        return {}
    return {str(p.relative_to(folder)): p for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() in READ_FIGURES | LISTED_FIGURES}


def same_bytes(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            chunk_a, chunk_b = fa.read(1 << 20), fb.read(1 << 20)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


# ----------------------------------------------------------------------------- numbers in an SVG

@dataclass
class Number:
    cls: str
    value: float
    raw: str          # the number as written, with its % sign if any
    label: str        # the whole text element
    is_percent: bool
    before: str = ""  # the last text element before it that holds no number, e.g. a pie wedge's name

    @property
    def context(self) -> str:
        return NUMBER.sub("#", self.label).strip() + " | " + self.before


def classify(text: str, in_tick: bool, tick_aware: bool = True) -> list[Number]:
    found = []
    for match in NUMBER.finditer(text):
        num, pct = match.group("num"), match.group("pct")
        if in_tick and tick_aware:
            cls = "tick"
        elif N_BEFORE.search(text[:match.start()]):
            cls = "n"
        elif pct or "." in num or "e" in num.lower():
            cls = "data"
        else:
            cls = "integer"
        found.append(Number(cls, to_float(num), num + (pct or "").strip(), " ".join(text.split()), bool(pct)))
    return found


def svg_numbers(path: Path, tick_aware: bool = True) -> dict[str, list[Number]]:
    """The numbers of every <text> element, in document order, by class. Streams the file."""
    out: dict[str, list[Number]] = {c: [] for c in CLASSES}
    groups: list[str] = []
    inside_text = 0
    before = ""
    for event, elem in ET.iterparse(path, events=("start", "end")):
        tag = elem.tag.rsplit("}", 1)[-1]
        if event == "start":
            groups.append(elem.get("id", "") if tag == "g" else "")
            if tag == "text":
                inside_text += 1
            continue
        groups.pop()
        if tag == "text":
            inside_text -= 1
            in_tick = any(TICK_GROUP.match(g) for g in groups)
            text = "".join(elem.itertext())
            numbers = classify(text, in_tick, tick_aware)
            for number in numbers:
                number.before = before
                out[number.cls].append(number)
            if not numbers and text.strip():
                before = " ".join(text.split())
        if not inside_text:
            elem.clear()
    return out


# ----------------------------------------------------------------------------- pairing

Pairing = Callable[[list[Number], list[Number]], "list[tuple[int, int]] | None"]


def pair_by_position(a: list[Number], b: list[Number]) -> list[tuple[int, int]] | None:
    """Pair the i-th number of one side with the i-th of the other; None when the counts differ."""
    if len(a) != len(b):
        return None
    return [(i, i) for i in range(len(a))]


@dataclass
class ClassResult:
    cls: str
    a: list[Number]
    b: list[Number]
    pairs: list[tuple[int, int]] | None

    @property
    def counts_match(self) -> bool:
        return self.pairs is not None

    @property
    def changed(self) -> list[tuple[Number, Number]]:
        return [(self.a[i], self.b[j]) for i, j in self.pairs or [] if self.a[i].value != self.b[j].value]


@dataclass
class FigureResult:
    name: str
    panel: str
    in_a: bool
    in_b: bool
    byte_identical: bool | None = None
    read: bool = False
    error: str = ""
    classes: dict[str, ClassResult] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if not (self.in_a and self.in_b):
            return "only in " + ("A" if self.in_a else "B")
        if not self.read:
            return "byte-identical" if self.byte_identical else "listed only (bytes differ)"
        if self.error:
            return "could not be read"
        if self.byte_identical:
            return "byte-identical"
        drawn = [self.classes[c] for c in VERDICT_CLASSES]
        if not any(r.a or r.b for r in drawn):
            return "no drawn numbers besides ticks"
        if not all(r.counts_match for r in drawn):
            return "counts differ"
        if any(r.changed for r in drawn):
            return "numbers changed"
        return "numbers identical"


def compare_figure(name: str, pa: Path | None, pb: Path | None, panel: str, pairing: Pairing,
                   tick_aware: bool) -> FigureResult:
    result = FigureResult(name, panel, pa is not None, pb is not None)
    if pa is None or pb is None:
        return result
    result.byte_identical = same_bytes(pa, pb)
    if pa.suffix.lower() not in READ_FIGURES:
        return result
    result.read = True
    try:
        na, nb = svg_numbers(pa, tick_aware), svg_numbers(pb, tick_aware)
    except ET.ParseError as exc:
        result.error = str(exc)
        return result
    for cls in CLASSES:
        result.classes[cls] = ClassResult(cls, na[cls], nb[cls], pairing(na[cls], nb[cls]))
    return result


# ----------------------------------------------------------------------------- printed output

def mask_line(line: str) -> tuple[str, list[str]]:
    """The line with numbers replaced by # and checksums by <checksum>, and the values masked."""
    values: list[str] = []

    def keep_checksum(m: re.Match) -> str:
        values.append(m.group(0).split()[-1])
        return "<checksum>"

    def keep_number(m: re.Match) -> str:
        values.append(m.group("num"))
        return "#" + ("%" if m.group("pct") else "")
    # checksums are taken out first, then numbers; both sides get the same order
    line = CHECKSUM.sub(keep_checksum, line)
    return NUMBER.sub(keep_number, line), values


def values_equal(x: str, y: str) -> bool:
    try:
        return to_float(x) == to_float(y)
    except ValueError:
        return x == y


LinePairing = Callable[[list[str], list[str]], "list[tuple[int, int]]"]


def pair_lines_by_template(ta: list[str], tb: list[str]) -> list[tuple[int, int]]:
    matcher = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    return [(block.a + k, block.b + k) for block in matcher.get_matching_blocks() for k in range(block.size)]


@dataclass
class StdoutResult:
    present: str                                   # "both", "A only", "B only", "neither"
    changed: list[tuple[str, str, list[tuple[str, str]]]] = field(default_factory=list)
    only_a: int = 0
    only_b: int = 0
    only_a_numbers: list[str] = field(default_factory=list)
    only_b_numbers: list[str] = field(default_factory=list)


def compare_stdout(pa: Path | None, pb: Path | None, line_pairing: LinePairing) -> StdoutResult:
    ea, eb = pa is not None and pa.exists(), pb is not None and pb.exists()
    if not (ea and eb):
        return StdoutResult({(True, False): "A only", (False, True): "B only"}.get((ea, eb), "neither"))
    lines_a = pa.read_text(encoding="utf-8", errors="replace").split("\n")
    lines_b = pb.read_text(encoding="utf-8", errors="replace").split("\n")
    masked_a, masked_b = [mask_line(x) for x in lines_a], [mask_line(x) for x in lines_b]
    result = StdoutResult("both")
    pairs = line_pairing([m[0] for m in masked_a], [m[0] for m in masked_b])
    for i, j in pairs:
        va, vb = masked_a[i][1], masked_b[j][1]
        if len(va) != len(vb) or not all(values_equal(x, y) for x, y in zip(va, vb)):
            diffs = [(x, y) for x, y in zip(va, vb) if not values_equal(x, y)]
            result.changed.append((lines_a[i].strip(), lines_b[j].strip(), diffs))
    seen_a, seen_b = {i for i, _ in pairs}, {j for _, j in pairs}
    rest_a = [lines_a[i].strip() for i in range(len(lines_a)) if i not in seen_a and lines_a[i].strip()]
    rest_b = [lines_b[j].strip() for j in range(len(lines_b)) if j not in seen_b and lines_b[j].strip()]
    result.only_a, result.only_b = len(rest_a), len(rest_b)
    result.only_a_numbers = [x for x in rest_a if mask_line(x)[1]]
    result.only_b_numbers = [x for x in rest_b if mask_line(x)[1]]
    return result


# ----------------------------------------------------------------------------- panel map

PANEL_IS = re.compile(r"((?:Fig |Supp )?(?:\d+[A-Z]+|\d-S\d+(?: [A-Z])?)(?: \(\w+\)| [A-Z][a-z]+)?) is "
                      r"`([^`]+\.(?:svg|pdf|png))`")
TICKED = re.compile(r"`([^`]+)`")


def panel_name(text: str) -> str:
    text = text.strip()
    return "Fig " + text if text[:1].isdigit() else text


class PanelMap:
    """File-to-panel lookup read from docs/figure_to_script_map.md (best effort)."""

    def __init__(self, doc: Path):
        self.rows: dict[str, list[str]] = {}
        self.named: dict[tuple[str, str], str] = {}
        self.folders: dict[str, str] = {}
        if not doc.exists():
            return
        lines = doc.read_text(encoding="utf-8", errors="replace").split("\n")
        for line in lines:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not line.startswith("|") or len(cells) < 4 or set(cells[0]) <= set("-: "):
                continue
            script = re.search(r"`(?:[\w./-]*/)?(\w+)\.py`", cells[1])
            if not script:
                continue
            stem, panel, output = script.group(1), panel_name(cells[0]), cells[-1]
            self.rows.setdefault(stem, [])
            if panel not in self.rows[stem]:
                self.rows[stem].append(panel)
            folder = re.search(r"`outputs/[^`]*?([^/`]+)/", output)
            if folder:
                self.folders.setdefault(folder.group(1), stem)
            for m in re.finditer(r"`([^`]+\.(?:svg|pdf|png))`", output):
                name = Path(m.group(1)).name
                after = re.match(r"\s*\((\d+[A-Z]+)\)", output[m.end():])
                before = re.search(r"\b(left|right|[A-D]):\s*$", output[:m.start()])
                this = (panel_name(after.group(1)) if after
                        else f"{panel} ({before.group(1)})" if before else panel)
                self.named.setdefault((stem, name), this)
        paragraph: list[str] = []
        for line in lines + [""]:
            if line.startswith(">"):
                paragraph.append(line.lstrip("> ").rstrip())
            elif paragraph:
                self._read_note(" ".join(paragraph))
                paragraph = []

    def _context(self, token: str) -> str | None:
        name = Path(token.rstrip("/")).name
        if token.endswith(".py") and name[:-3] in self.rows:
            return name[:-3]
        if token.endswith("/"):
            return self.folders.get(name) or (name if name in self.rows else None)
        return None

    def _read_note(self, text: str) -> None:
        last: str | None = None
        for sentence in re.split(r"(?<=\.)\s+(?=[A-Z(`])", text):
            tokens = [(m.start(), self._context(m.group(1))) for m in TICKED.finditer(sentence)]
            tokens = [(pos, stem) for pos, stem in tokens if stem]
            for m in PANEL_IS.finditer(sentence):
                before = [stem for pos, stem in tokens if pos < m.start()]
                after = [stem for pos, stem in tokens if pos > m.start()]
                stem = before[-1] if before else after[0] if after else last
                if stem:
                    self.named.setdefault((stem, Path(m.group(2)).name), panel_name(m.group(1)))
            if tokens:
                last = tokens[-1][1]

    def panel(self, script: str, figure: str) -> str:
        name = Path(figure).name
        if (script, name) in self.named:
            return self.named[(script, name)]
        panels = self.rows.get(script, [])
        others_named = any(stem == script for stem, _ in self.named)
        if len(panels) == 1:
            return panels[0][:90] + (" (file not named in the map)" if others_named else "")
        if panels:
            return "one of: " + "; ".join(p.split(" (")[0][:40] for p in panels) + " (file not named in the map)"
        return "not a mapped panel"


# ----------------------------------------------------------------------------- whole runs

@dataclass
class ScriptResult:
    name: str
    status_a: tuple[str, str]
    status_b: tuple[str, str]
    figures: list[FigureResult]
    stdout: StdoutResult

    @property
    def flag(self) -> str:
        return flag_of(self.status_a[0], self.status_b[0])


def compare_runs(run_a: Path, run_b: Path, panel_map: PanelMap, pairing: Pairing = pair_by_position,
                 line_pairing: LinePairing = pair_lines_by_template, tick_aware: bool = True) -> list[ScriptResult]:
    folders_a, folders_b = script_folders(run_a), script_folders(run_b)
    results = []
    for name in sorted(folders_a.keys() | folders_b.keys()):
        fa, fb = folders_a.get(name), folders_b.get(name)
        figs_a, figs_b = figure_files(fa), figure_files(fb)
        figures = [compare_figure(f, figs_a.get(f), figs_b.get(f), panel_map.panel(name, f), pairing, tick_aware)
                   for f in sorted(figs_a.keys() | figs_b.keys())]
        results.append(ScriptResult(name, run_status(fa), run_status(fb), figures,
                                    compare_stdout(fa / "stdout.log" if fa else None,
                                                   fb / "stdout.log" if fb else None, line_pairing)))
    return results


def fmt(x: float) -> str:
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else f"{x:.6g}"


def status_text(status: tuple[str, str]) -> str:
    return status[0] + (f" ({status[1]})" if status[1] else "")


def first_line(path: Path, last: bool = False) -> str:
    if not path.exists():
        return "(none)"
    lines = [x for x in path.read_text(encoding="utf-8", errors="replace").split("\n") if x.strip()]
    return (lines[-1] if last else lines[0]) if lines else "(empty)"


def write_outputs(results: list[ScriptResult], run_a: Path, run_b: Path, labels: tuple[str, str], out: Path,
                  max_lines: int, max_numbers: int) -> list[str]:
    """Write report.txt and cells.csv; return the summary lines."""
    la, lb = labels
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    flags: dict[str, list[str]] = {}
    groups: dict[str, list[str]] = {}
    percent_changes: list[tuple[float, str, Number, Number]] = []
    misaligned = 0
    details: list[str] = []
    for res in results:
        flags.setdefault(res.flag, []).append(res.name)
        details += ["", f"## {res.name}  [{res.flag}]",
                    f"status {la}: {status_text(res.status_a)}", f"status {lb}: {status_text(res.status_b)}"]
        for fig in res.figures:
            where = f"{res.name}/{fig.name}"
            verdict = {"only in A": f"only in {la}", "only in B": f"only in {lb}"}.get(fig.verdict, fig.verdict)
            groups.setdefault(fig.verdict, []).append(f"{where}  ({fig.panel})")
            details.append(f"- {fig.name}  ({fig.panel}): {verdict}"
                           + ("" if fig.byte_identical is None or fig.byte_identical else "; bytes differ"))
            if fig.error:
                details.append(f"    could not read the SVG: {fig.error}")
            for cls, cr in fig.classes.items():
                if not (cr.a or cr.b):
                    continue
                base = {"script": res.name, "file": fig.name, "panel": fig.panel, "class": cls}
                if not cr.counts_match:
                    details.append(f"    {cls}: counts differ, {len(cr.a)} in {la} and {len(cr.b)} in {lb}")
                    for side, nums in ((la, cr.a), (lb, cr.b)):
                        shown = ", ".join(n.raw for n in nums[:max_numbers])
                        more = f" ... and {len(nums) - max_numbers} more" if len(nums) > max_numbers else ""
                        details.append(f"      {side}: {shown or '-'}{more}")
                    for k in range(max(len(cr.a), len(cr.b))):
                        na = cr.a[k] if k < len(cr.a) else None
                        nb = cr.b[k] if k < len(cr.b) else None
                        rows.append(dict(base, index=k, paired="no (counts differ)",
                                         label_a=na.label if na else "", value_a=fmt(na.value) if na else "",
                                         label_b=nb.label if nb else "", value_b=fmt(nb.value) if nb else "",
                                         abs_diff="", same_label=""))
                    continue
                changed = cr.changed
                details.append(f"    {cls}: {len(cr.a)} numbers, {len(changed)} changed"
                               + "".join(f", {m} paired with a different label" for m in
                                         [sum(cr.a[i].context != cr.b[j].context for i, j in cr.pairs or [])] if m))
                shown = 0
                for k, (i, j) in enumerate(cr.pairs or []):
                    na, nb = cr.a[i], cr.b[j]
                    same_label = na.context == nb.context
                    diff = abs(nb.value - na.value)
                    rows.append(dict(base, index=k, paired="yes", label_a=na.label, value_a=fmt(na.value),
                                     label_b=nb.label, value_b=fmt(nb.value), abs_diff=fmt(diff),
                                     same_label="yes" if same_label else "no"))
                    if not same_label:
                        misaligned += 1
                    if na.value != nb.value or not same_label:
                        if na.value != nb.value and cls != "tick" and na.is_percent and nb.is_percent:
                            percent_changes.append((diff, where, na, nb))
                        if shown < max_numbers:
                            mark = "" if same_label else (f"   [labels differ: \"{na.before}\" / \"{nb.before}\"; "
                                                          f"check the pairing]")
                            details.append(f"      #{k}: \"{na.label}\" -> \"{nb.label}\"  "
                                           f"|diff| {fmt(diff)}{mark}")
                        shown += 1
                if shown > max_numbers:
                    details.append(f"      ... and {shown - max_numbers} more (see cells.csv)")
        so = res.stdout
        if so.present != "both":
            details.append(f"- stdout.log: {so.present}")
        else:
            details.append(f"- stdout.log: {len(so.changed)} aligned lines with different numbers; "
                           f"{so.only_a} lines only in {la}, {so.only_b} only in {lb}")
            for k, (line_a, line_b, diffs) in enumerate(so.changed):
                if k < max_lines:
                    details += [f"    {la}: {line_a[:200]}", f"    {lb}: {line_b[:200]}"]
                for d, (x, y) in enumerate(diffs):
                    try:
                        diff = fmt(abs(to_float(y) - to_float(x)))
                    except ValueError:
                        diff = ""
                    rows.append({"script": res.name, "file": "stdout.log", "panel": "", "class": "stdout",
                                 "index": f"{k}.{d}", "paired": "yes", "label_a": line_a, "value_a": x,
                                 "label_b": line_b, "value_b": y, "abs_diff": diff, "same_label": "yes"})
            if len(so.changed) > max_lines:
                details.append(f"    ... and {len(so.changed) - max_lines} more lines (see cells.csv)")
            for side, rest in ((la, so.only_a_numbers), (lb, so.only_b_numbers)):
                if rest:
                    details.append(f"    lines with numbers only in {side}: {len(rest)}, first: {rest[0][:160]}")

    percent_changes.sort(key=lambda t: -t[0])
    summary = [f"# {la}: {run_a}", f"#   commit: {first_line(run_a / 'commit.txt', last=True)}",
               f"#   progress: {first_line(run_a / 'progress.txt', last=True)}",
               f"# {lb}: {run_b}", f"#   commit: {first_line(run_b / 'commit.txt', last=True)}",
               f"#   progress: {first_line(run_b / 'progress.txt', last=True)}", "",
               f"Scripts: {len(results)} (script folders in either run)"]
    for flag in ("finished on both sides", "finished on one side only", "finished on neither side"):
        names = flags.get(flag, [])
        summary.append(f"  {flag}: {len(names)}" + (f" ({', '.join(names)})" if names and flag != "finished on both sides" else ""))
    nfig = sum(len(r.figures) for r in results)
    summary += ["", f"Figure files: {nfig} (SVG read; PDF/PNG listed). Verdict counts tick labels out."]
    order = ["numbers identical", "byte-identical", "numbers changed", "counts differ",
             "no drawn numbers besides ticks", "could not be read", "listed only (bytes differ)", "only in A", "only in B"]
    for verdict in order + sorted(set(groups) - set(order)):
        items = groups.get(verdict, [])
        if not items:
            continue
        shown = verdict.replace("only in A", f"only in {la}").replace("only in B", f"only in {lb}")
        summary.append(f"  {shown}: {len(items)}")
        if verdict in ("numbers identical", "byte-identical", "numbers changed", "counts differ"):
            summary += [f"      {item}" for item in items]
    summary.append(f"  paired numbers whose label, or the text before it, differs (check the pairing there): {misaligned}")
    summary += ["", f"Largest absolute changes in percentage labels (percentage points), "
                    f"top 20 of {len(percent_changes)}:"]
    for diff, where, na, nb in percent_changes[:20]:
        summary.append(f"  {fmt(diff):>8}  {where}: \"{na.label}\" -> \"{nb.label}\"")
    if not percent_changes:
        summary.append("  none")
    text = "\n".join(summary) + "\n\n# Per script\n" + "\n".join(details) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    columns = ["script", "file", "panel", "class", "index", "paired", "label_a", "value_a",
               "label_b", "value_b", "abs_diff", "same_label"]
    with open(out / "cells.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return summary


# ----------------------------------------------------------------------------- selftest

SVG_HEAD = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n<svg xmlns="http://www.w3.org/2000/svg">\n'


def fake_svg(texts: list[str], ticks: list[str] = ()) -> str:
    body = "".join(f'  <g id="xtick_{k}"><g id="text_{k}"><text style="font: 8px \'Arial\'">{t}</text></g></g>\n'
                   for k, t in enumerate(ticks, 1))
    body += "".join(f'  <g id="text_l{k}"><text style="font: 8px \'Arial\'">{t}</text></g>\n'
                    for k, t in enumerate(texts))
    return SVG_HEAD + body + "</svg>\n"


def build_fake_runs(root: Path) -> tuple[Path, Path, Path]:
    a, b = root / "run_a", root / "run_b"
    files = {
        a / "s_fig" / "fig_n.svg": fake_svg(["AD n=100", "12.5%"], ["0", "50", "100"]),
        b / "s_fig" / "fig_n.svg": fake_svg(["AD n=120", "12.5%"], ["0", "50", "100"]),
        a / "s_fig" / "same.svg": fake_svg(["n = 7", "33.3%", "SI 0.1"], ["0", "1"]),
        b / "s_fig" / "same.svg": fake_svg(["n = 7", "33.3%", "SI 0.1"], ["0", "1"]),
        a / "s_fig" / "swap.svg": fake_svg(["30.0%", "10.0%"]),
        b / "s_fig" / "swap.svg": fake_svg(["10.0%", "30.0%"]),
        a / "s_fig" / "pie.svg": fake_svg(["AA", "30.0%", "AD", "10.0%"]),
        b / "s_fig" / "pie.svg": fake_svg(["AD", "10.0%", "AA", "30.0%"]),
        a / "s_fig" / "count.svg": fake_svg(["1.5", "2.5", "3.5"]),
        b / "s_fig" / "count.svg": fake_svg(["1.5", "2.5"]),
        a / "s_fig" / "stdout.log": "start\ntotal synapses: 500\nmean SI 0.42\n[4 rows, checksum 691bd66a8dd67f70]\n",
        b / "s_fig" / "stdout.log": "start\nloading extra\ntotal synapses: 520\nmean SI 0.42\n"
                                    "[4 rows, checksum 691bd66a8dd67f70]\n",
        a / "s_fig" / "run_status.txt": "finished\n",
        b / "s_fig" / "run_status.txt": "finished\n",
        a / "s_one" / "run_status.txt": "finished\n",
        b / "s_one" / "run_status.txt": "stopped with an error\nTraceback (most recent call last):\nValueError: boom\n",
        a / "s_one" / "stdout.log": "x 1\n",
        b / "s_one" / "stdout.log": "x 1\n",
        a / "s_gone" / "run_status.txt": "finished\n",
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for side in (a, b):
        (side / "s_fig" / "plot.pdf").write_bytes(b"%PDF-1.4 same bytes")
    doc = root / "map.md"
    doc.write_text("| Panel | Script | Input tables | Output path |\n|---|---|---|---|\n"
                   "| Fig 9Z | `figures/fig9/s_fig.py` | `T` | `outputs/fig9/s_fig/`: `fig_n.svg` |\n"
                   "| Fig 9Y | `figures/fig9/s_fig.py` | `T` | `outputs/fig9/s_fig/` |\n\n"
                   "> In `outputs/fig9/s_fig/`, 9Y is `swap.svg`.\n", encoding="utf-8")
    return a, b, doc


def check_fake(results: list[ScriptResult], out: Path) -> list[str]:
    """Return the list of failed expectations (empty when everything is as built)."""
    fails: list[str] = []

    def expect(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)
    by = {r.name: r for r in results}
    expect(sorted(by) == ["s_fig", "s_gone", "s_one"], "three scripts found")
    if sorted(by) != ["s_fig", "s_gone", "s_one"]:
        return fails
    expect(by["s_fig"].flag == "finished on both sides", "s_fig finished on both")
    expect(by["s_one"].flag == "finished on one side only", "s_one finished on one side")
    expect(by["s_one"].status_b == ("stopped with an error", "ValueError: boom"), "s_one error line")
    expect(by["s_gone"].status_b[0] == "missing" and by["s_gone"].flag == "finished on one side only",
           "s_gone missing on B")
    figs = {f.name: f for f in by["s_fig"].figures}
    names = ["count.svg", "fig_n.svg", "pie.svg", "plot.pdf", "same.svg", "swap.svg"]
    expect(sorted(figs) == names, "six figure files")
    if sorted(figs) != names:
        return fails
    changed = {(f.name, cls, na.raw, nb.raw) for f in figs.values() for cls, cr in f.classes.items()
               for na, nb in cr.changed}
    expect(changed == {("fig_n.svg", "n", "100", "120"), ("swap.svg", "data", "30.0%", "10.0%"),
                       ("swap.svg", "data", "10.0%", "30.0%"), ("pie.svg", "data", "30.0%", "10.0%"),
                       ("pie.svg", "data", "10.0%", "30.0%")}, f"exactly the planted changes, got {sorted(changed)}")
    flagged = {(f.name, cr.a[i].raw) for f in figs.values() for cr in f.classes.values() for i, j in cr.pairs or []
               if cr.a[i].context != cr.b[j].context}
    expect(flagged == {("pie.svg", "30.0%"), ("pie.svg", "10.0%")}, f"only the reordered pie flagged, got {flagged}")
    fn = figs["fig_n.svg"]
    expect(fn.verdict == "numbers changed" and fn.byte_identical is False, "fig_n changed, bytes differ")
    expect([n.raw for n in fn.classes["n"].a] == ["100"] and [n.raw for n in fn.classes["data"].b] == ["12.5%"],
           "fig_n classes n and data")
    expect([n.raw for n in fn.classes["tick"].a] == ["0", "50", "100"] and not fn.classes["tick"].changed,
           "fig_n ticks separate and unchanged")
    expect(figs["same.svg"].verdict == "byte-identical" and figs["same.svg"].byte_identical, "same.svg identical")
    expect(all(not cr.changed and cr.counts_match for cr in figs["same.svg"].classes.values()),
           "nothing reported for same.svg")
    expect(figs["count.svg"].verdict == "counts differ" and not figs["count.svg"].classes["data"].counts_match,
           "count.svg counts differ")
    expect(figs["plot.pdf"].byte_identical and not figs["plot.pdf"].read, "pdf listed, not read")
    expect(fn.panel == "Fig 9Z" and figs["swap.svg"].panel == "Fig 9Y", "panels from table row and note")
    expect(figs["same.svg"].panel.startswith("one of:"), "unnamed file of a two-panel script")
    expect(by["s_one"].figures == [] and all(f.panel != "not a mapped panel" for f in figs.values()), "panel map")
    so = by["s_fig"].stdout
    expect([d for _, _, d in so.changed] == [[("500", "520")]], f"one changed printed line, got {so.changed}")
    expect(so.only_b == 1 and so.only_a == 0, "one printed line only in B")
    expect(not by["s_one"].stdout.changed, "no printed change in s_one")
    report = (out / "report.txt").read_text(encoding="utf-8")
    expect('"30.0%" -> "10.0%"  |diff| 20   [labels differ: "AA" / "AD"' in report, "report flags the pie pairing")
    expect('"AD n=100" -> "AD n=120"' in report, "report shows the n change")
    expect("same.svg  (one of:" in report and "33.3%" not in report, "report lists nothing from same.svg")
    expect("counts differ, 3 in A and 2 in B" in report, "report says counts differ")
    with open(out / "cells.csv", encoding="utf-8") as fh:
        cells = list(csv.DictReader(fh))
    pct = [c for c in cells if c["file"] == "swap.svg"]
    expect(len(pct) == 2 and all(c["abs_diff"] == "20" and c["same_label"] == "yes" for c in pct),
           "cells.csv swap rows")
    expect([c["same_label"] for c in cells if c["file"] == "pie.svg"] == ["no", "no"], "cells.csv pie rows")
    top = report.split("top 20 of")[1].split("\n\n")[0]
    expect("swap.svg" in top and "12.5%" not in top and top.startswith(" 4:"), "top percentage changes")
    return fails


def selftest() -> bool:
    def pair_reversed(a, b):
        return None if len(a) != len(b) else [(i, len(b) - 1 - i) for i in range(len(a))]

    def pair_sorted(a, b):
        if len(a) != len(b):
            return None
        ia = sorted(range(len(a)), key=lambda i: a[i].value)
        ib = sorted(range(len(b)), key=lambda i: b[i].value)
        return list(zip(ia, ib))

    def pair_truncated(a, b):
        return [(i, i) for i in range(min(len(a), len(b)))]

    def lines_by_index(ta, tb):
        return [(i, i) for i in range(min(len(ta), len(tb)))]

    planted = {"pairs in reverse order": dict(pairing=pair_reversed),
               "pairs by sorted value": dict(pairing=pair_sorted),
               "pairs the shorter list when counts differ": dict(pairing=pair_truncated),
               "tick labels not told apart": dict(tick_aware=False),
               "printed lines paired by index": dict(line_pairing=lines_by_index),
               "label check ignores the wedge name before a number": "context"}
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_a, run_b, doc = build_fake_runs(root)
        panel_map = PanelMap(doc)
        results = compare_runs(run_a, run_b, panel_map)
        write_outputs(results, run_a, run_b, ("A", "B"), root / "out", 25, 30)
        fails = check_fake(results, root / "out")
        print(("PASS" if not fails else "FAIL") + ": the comparison reports exactly the planted differences")
        for f in fails:
            print(f"    failed: {f}")
        ok = not fails
        for k, (name, kwargs) in enumerate(planted.items()):
            out = root / f"out_planted_{k}"
            real_context = Number.context
            if kwargs == "context":
                Number.context = property(lambda self: NUMBER.sub("#", self.label).strip())
                kwargs = {}
            try:
                res = compare_runs(run_a, run_b, panel_map, **kwargs)
                write_outputs(res, run_a, run_b, ("A", "B"), out, 25, 30)
                caught = bool(check_fake(res, out))
            finally:
                Number.context = real_context
            print(f"{'PASS' if caught else 'FAIL'}: planted mistake caught: {name}")
            ok = ok and caught
    print("selftest " + ("passed" if ok else "FAILED"))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a", type=Path, nargs="?", help="first run_all_figures.py output folder")
    parser.add_argument("run_b", type=Path, nargs="?", help="second run_all_figures.py output folder")
    parser.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("LABEL_A", "LABEL_B"),
                        help="names for the two runs in the report, e.g. SFC MaxSI")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: outputs/tools/compare_split_figures/<label_a>_vs_<label_b>/)")
    parser.add_argument("--map", type=Path, default=MAP_DOC, help="figure-to-script map (default: docs/figure_to_script_map.md)")
    parser.add_argument("--max-lines", type=int, default=25,
                        help="printed lines with changed numbers shown per script in report.txt (default: 25)")
    parser.add_argument("--max-numbers", type=int, default=30,
                        help="changed numbers shown per figure and class in report.txt (default: 30)")
    parser.add_argument("--selftest", action="store_true", help="run the selftest on two fake runs and exit")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.run_a is None or args.run_b is None:
        parser.error("give two run folders, or --selftest")
    for folder in (args.run_a, args.run_b):
        if not folder.is_dir():
            parser.error(f"{folder} is not a folder")
    label_a, label_b = args.labels
    out = args.out or REPO_ROOT / "outputs" / "tools" / "compare_split_figures" / f"{label_a}_vs_{label_b}"
    results = compare_runs(args.run_a, args.run_b, PanelMap(args.map))
    summary = write_outputs(results, args.run_a, args.run_b, (label_a, label_b), out, args.max_lines, args.max_numbers)
    print("\n".join(summary))
    print(f"\nreport: {out / 'report.txt'}\ncells: {out / 'cells.csv'}")


if __name__ == "__main__":
    # Compares two full runs of every figure script (made with tools/run_all_figures.py) by the numbers drawn:
    #   outputs/tools/run_all_figures/sfc     (first run, labelled SFC)
    #   outputs/tools/run_all_figures/maxsi   (second run, labelled MaxSI)
    # and writes report.txt and cells.csv to outputs/tools/compare_split_figures/SFC_vs_MaxSI/.
    # The paths are relative to the repository root, so run it from there. To run it, uncomment:
    # sys.argv[1:] = ["outputs/tools/run_all_figures/sfc", "outputs/tools/run_all_figures/maxsi",
    #                 "--labels", "SFC", "MaxSI"]
    main()
