"""Compare the output of two figure runs.

Files are matched by name (with --figures-only, only .svg, .png and .pdf files;
a figure that exists in only one folder counts as a difference). For an SVG it
reports whether the files are identical, or how many lines differ once the parts
that change on every save are ignored
(element ids, links between elements, and metadata such as the date). Any other
file, such as a PNG, must be identical byte for byte. stdout.log must be identical
too, unless --allow-extra-lines is given; then every line of the first run must
appear, in the same order, in the second. stderr.log (warnings) is not compared.

Usage:
    python tools/compare_outputs.py OLD_RUN_DIR NEW_RUN_DIR
    python tools/compare_outputs.py OLD_RUN_DIR NEW_RUN_DIR --allow-extra-lines
    python tools/compare_outputs.py FOLDER_OF_PUBLISHED_SVGS NEW_RUN_DIR --figures-only
    python tools/compare_outputs.py FOLDER_OF_PUBLISHED_SVGS NEW_RUN_DIR --figures-only --content

With --figures-only against a folder of SVGs, the run must save only SVGs; a PNG or
PDF it also saves is reported as missing from the other folder.

With --content, an SVG whose lines still differ is compared by what it draws: its
text and its colours, both in drawing order, and, inside each plot, its markers (with
their shapes) and shapes measured against that plot's background rectangle, or the
box around everything in it when there is none, as in a pie. Figure size and margins
cancel out, but a bar's height, a point's position or a label moved to another bar
do not. If all four match, the file passes as the same content in a different
layout, which is what a different figure size or font produces. Legends are left out
of the shapes and markers, positions are rounded to 0.1% of the plot, line widths
and font sizes are not compared, and an SVG with no text elements (text saved as
outlines) fails, because its text cannot be read.

A folder made by run_figure.py holds run_status.txt; if it does not say the run
finished, the comparison fails, because two runs that stop at the same error can
otherwise look identical. A folder without the file (such as the published SVGs)
is noted, not failed.

Exits with status 1 if anything differs, if a file exists in only one folder, if a
run did not finish, or if neither folder holds a figure (printed output alone is
not accepted as a match unless --allow-no-figures is given). Ignoring ids hides one
kind of change: an element that switches to a different clip region or marker that
already exists in the file.
"""

from __future__ import annotations

import argparse
import difflib
import html
import xml.etree.ElementTree as ET
import re
import sys
from pathlib import Path

_METADATA = re.compile(r"dc:date|<dc:|<cc:|<rdf:|</rdf|</cc|metadata")
_IDS = re.compile(r'(id|href|clip-path)="[^"]*"')
_URLS = re.compile(r"url\(#[^)]*\)")
_STATUS = "run_status.txt"
_NOT_COMPARED = {"stderr.log", _STATUS}
_FIGURES = {".svg", ".png", ".pdf"}


def normalise_svg(path: Path) -> list[str]:
    """Return the SVG's lines with ids, links and metadata blanked out."""
    lines = path.read_text(encoding="utf-8").replace("\r", "").split("\n")
    kept = [line for line in lines if not _METADATA.search(line)]
    return [_URLS.sub("url()", _IDS.sub(r'\1=""', line)) for line in kept]


def count_differing_lines(a: list[str], b: list[str]) -> int:
    diff = difflib.unified_diff(a, b, lineterm="", n=0)
    return sum(1 for line in diff
               if line[:1] in "+-" and not line.startswith(("+++", "---")))


_SVG = "{http://www.w3.org/2000/svg}"
_XLINK = "{http://www.w3.org/1999/xlink}href"
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_COLOUR = re.compile(r"(?:fill|stroke)\s*:\s*(#[0-9a-fA-F]{3,8}|none)")


def _points(d: str) -> list[tuple[float, float]]:
    nums = [float(v) for v in _NUMBER.findall(d)]
    return list(zip(nums[0::2], nums[1::2]))


def _is_rectangle(points: list[tuple[float, float]]) -> bool:
    xs, ys = {round(x, 3) for x, _ in points}, {round(y, 3) for _, y in points}
    return len(points) >= 4 and len(xs) == 2 and len(ys) == 2


def svg_content(path: Path, digits: int = 3) -> tuple[list[str], list[str], list[list], list[list]]:
    """The text, the colours, the markers and the shapes an SVG draws, with the layout removed.

    Text and colours are kept in drawing order, so a label moved to another bar or a
    colour moved to another group counts as a change. matplotlib draws each plot as a
    group "axes_N"; inside it, every path and marker is rescaled to the plot's background
    rectangle (or, for a plot without one, such as a pie, to the box around everything
    in it), separately for x and y, so a different figure size or margin cancels out
    while a bar's height or a point's position still counts. A marker keeps its own
    shape. Legends are left out of the shapes and markers (their size follows the font).
    """
    root = ET.parse(path).getroot()
    defs = {}
    for group in root.iter(_SVG + "defs"):
        for element in group.iter(_SVG + "path"):
            if element.get("id"):
                defs[element.get("id")] = element.get("d", "")
    texts = [html.unescape("".join(el.itertext())).strip() for el in root.iter(_SVG + "text")]
    in_defs = {id(el) for group in root.iter(_SVG + "defs") for el in group.iter()}
    colours = []
    for element in root.iter():
        if id(element) in in_defs:
            continue
        colours += _COLOUR.findall(element.get("style", "") or "")
        colours += [element.get(k) for k in ("fill", "stroke") if element.get(k)]
    markers, shapes = [], []
    for axes in root.iter(_SVG + "g"):
        if not re.fullmatch(r"axes_\d+", axes.get("id", "")):
            continue
        paths, uses = [], []

        def walk(element, in_legend):
            for child in element:
                legend = in_legend or bool(re.fullmatch(r"legend_\d+", child.get("id", "") or ""))
                if not legend and child.tag == _SVG + "path" and _points(child.get("d", "")):
                    paths.append(_points(child.get("d")))
                elif not legend and child.tag == _SVG + "use":
                    ref = (child.get(_XLINK) or child.get("href") or "").lstrip("#")
                    uses.append((float(child.get("x", 0)), float(child.get("y", 0)), defs.get(ref, ref)))
                if child.tag == _SVG + "g":
                    walk(child, legend)

        walk(axes, False)
        if not paths and not uses:
            continue
        if paths and _is_rectangle(paths[0]):
            frame, drawn = paths[0], paths[1:]
        else:
            frame, drawn = [p for path_ in paths for p in path_] + [(x, y) for x, y, _ in uses], paths
        xs, ys = [p[0] for p in frame], [p[1] for p in frame]
        x0, y0 = min(xs), min(ys)
        xr, yr = (max(xs) - x0) or 1.0, (max(ys) - y0) or 1.0

        def scaled(points):
            return tuple((round((x - x0) / xr, digits), round((y - y0) / yr, digits)) for x, y in points)

        shapes.append(sorted(scaled(p) for p in drawn))
        markers.append(sorted(scaled([(x, y)]) + (shape,) for x, y, shape in uses))
    return texts, colours, markers, shapes


def same_content(a: Path, b: Path) -> tuple[bool, str]:
    """Whether two SVGs draw the same text, colours, markers and shapes; and a short reason."""
    ta, ca, ma, sa = svg_content(a)
    tb, cb, mb, sb = svg_content(b)
    if not ta and not tb:
        return False, "no text elements (text saved as outlines?), so the text cannot be compared"
    if ta != tb:
        return False, "the text differs (or is in a different order)"
    if ca != cb:
        return False, "the colours differ (or are in a different order)"
    if ma != mb:
        return False, "the markers differ"
    if sa != sb:
        return False, "the shapes differ"
    return True, (f"same text, colours, {sum(map(len, ma))} markers and {sum(map(len, sa))} shapes "
                  f"in {len(sa)} plot(s), in a different layout")


def printed_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def contains_in_order(first: list[str], second: list[str]) -> bool:
    remaining = iter(second)
    return all(any(line == other for other in remaining) for line in first)


def _compare_log(log_a: Path, log_b: Path, allow_extra_lines: bool) -> bool:
    first, second = printed_lines(log_a), printed_lines(log_b)
    if first == second:
        print(f"{log_a.name}: identical")
        return True
    if allow_extra_lines and contains_in_order(first, second):
        print(f"{log_a.name}: all {len(first)} lines of the first run appear in order "
              f"in the second ({len(second)} lines)")
        return True
    print(f"{log_a.name}: differs")
    return False


def _files(folder: Path, figures_only: bool) -> dict[str, Path]:
    files = {p.name: p for p in folder.iterdir() if p.is_file() and p.name not in _NOT_COMPARED}
    if figures_only:
        files = {name: p for name, p in files.items() if p.suffix.lower() in _FIGURES}
    return files


def _run_finished(folder: Path) -> bool:
    status = folder / _STATUS
    if not status.exists():
        print(f"{folder}: no {_STATUS}, so whether that run finished is not known")
        return True
    first = status.read_text(encoding="utf-8").split("\n", 1)[0]
    if first != "finished":
        print(f"{folder}: the run did not finish ({_STATUS}: {first})")
        return False
    return True


def compare(dir_a: Path, dir_b: Path, allow_extra_lines: bool = False,
            figures_only: bool = False, allow_no_figures: bool = False,
            content: bool = False) -> bool:
    files_a, files_b = _files(dir_a, figures_only), _files(dir_b, figures_only)
    ok = _run_finished(dir_a)
    ok = _run_finished(dir_b) and ok
    if not files_a and not files_b:
        print("nothing to compare: neither folder holds a file that is compared")
        return False
    names = files_a.keys() | files_b.keys()
    has_figures = any(Path(name).suffix.lower() in _FIGURES for name in names)
    if not has_figures and not allow_no_figures:
        print("no figure in either folder: printed output alone is not accepted as a match "
              "(use --allow-no-figures to compare it anyway)")
        ok = False
    for name in sorted(files_a.keys() | files_b.keys()):
        if name not in files_a or name not in files_b:
            print(f"{name}: only in {dir_a if name in files_a else dir_b}")
            ok = False
            continue
        a, b = files_a[name], files_b[name]
        if name.endswith(".log"):
            ok = _compare_log(a, b, allow_extra_lines) and ok
        elif a.read_bytes() == b.read_bytes():
            print(f"{name}: identical")
        elif name.endswith(".svg"):
            lines_a, lines_b = normalise_svg(a), normalise_svg(b)
            differing = count_differing_lines(lines_a, lines_b)
            print(f"{name}: {differing} of {max(len(lines_a), len(lines_b))} lines differ "
                  "after normalising")
            if differing and content:
                same, why = same_content(a, b)
                print(f"    content: {why}")
                ok = ok and same
            else:
                ok = ok and differing == 0
        else:
            print(f"{name}: differs")
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("first", type=Path, help="output folder of the first run")
    parser.add_argument("second", type=Path, help="output folder of the second run")
    parser.add_argument("--allow-extra-lines", action="store_true",
                        help="pass if the second run prints extra lines, as long as every "
                             "line of the first run appears in order")
    parser.add_argument("--figures-only", action="store_true",
                        help="compare only figure files (.svg, .png, .pdf), for example "
                             "against a folder that holds just the published SVGs")
    parser.add_argument("--content", action="store_true",
                        help="let an SVG whose lines differ pass if it draws the same text, "
                             "markers and shapes (layout removed)")
    parser.add_argument("--allow-no-figures", action="store_true",
                        help="compare folders that hold no figure, for a script that only "
                             "prints or writes tables")
    args = parser.parse_args()
    for folder in (args.first, args.second):
        if not folder.is_dir():
            parser.error(f"{folder} is not a folder")
    sys.exit(0 if compare(args.first, args.second, args.allow_extra_lines,
                          args.figures_only, args.allow_no_figures, args.content) else 1)


if __name__ == "__main__":
    main()
