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

With --figures-only against a folder of SVGs, the run must save only SVGs; a PNG or
PDF it also saves is reported as missing from the other folder.

Exits with status 1 if anything differs, if a file exists in only one folder, or if
there was nothing to compare. Ignoring ids hides one kind of change: an element
that switches to a different clip region or marker that already exists in the file.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

_METADATA = re.compile(r"dc:date|<dc:|<cc:|<rdf:|</rdf|</cc|metadata")
_IDS = re.compile(r'(id|href|clip-path)="[^"]*"')
_URLS = re.compile(r"url\(#[^)]*\)")
_NOT_COMPARED = {"stderr.log"}
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


def compare(dir_a: Path, dir_b: Path, allow_extra_lines: bool = False,
            figures_only: bool = False) -> bool:
    files_a, files_b = _files(dir_a, figures_only), _files(dir_b, figures_only)
    if not files_a and not files_b:
        print("nothing to compare: both folders are empty")
        return False

    ok = True
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
    args = parser.parse_args()
    for folder in (args.first, args.second):
        if not folder.is_dir():
            parser.error(f"{folder} is not a folder")
    sys.exit(0 if compare(args.first, args.second, args.allow_extra_lines,
                          args.figures_only) else 1)


if __name__ == "__main__":
    main()
