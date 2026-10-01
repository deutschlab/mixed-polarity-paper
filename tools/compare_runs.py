"""Compare two full runs of the figure scripts, script by script.

Takes two output folders made by run_all_figures.py (one subfolder per script) and,
for every script in either, compares:
- the run status (finished, stopped with an error, timed out); for a script that
  stopped with an error, also the last line of its traceback, so that two different
  errors do not count as the same;
- every saved figure (.svg, .png, .pdf), matched by its path inside the script's
  folder: byte-identical, identical once ids and metadata are ignored (as in
  compare_outputs.py), or the number of SVG lines that still differ;
- the printed output in stdout.log.

A script counts as the same only if all three match. The report lists the scripts
that are the same and finished, those that are the same but did not finish (for
example both stopped at the same missing file), those that differ, and those found in
only one run, followed by the details of each. Folders named <name>.partial.<time>
are ignored.

--mask replaces every match of a regular expression with <masked> in both printed
outputs before they are compared, so that text which changes on every run, such as
the date and time statsmodels prints in its summary tables, does not count as a
difference while the rest of the line, which can hold a statistic, still does:
    python tools/compare_runs.py RUN_A RUN_B \\
        --mask "[A-Z][a-z]{2}, \\d{2} [A-Z][a-z]{2} \\d{4}" --mask "\\d{2}:\\d{2}:\\d{2}"

Usage:
    python tools/compare_runs.py outputs/runs/before outputs/runs/after
    python tools/compare_runs.py outputs/runs/before outputs/runs/after --report report.md

Exits with status 1 if any script differs or is found in only one run; scripts that
did not finish in either run but did so in the same way do not change the exit status.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_outputs import count_differing_lines, normalise_svg  # noqa: E402

FIGURES = {".svg", ".png", ".pdf"}


def run_status(folder: Path) -> str:
    """The status line, plus the last traceback line for a script that stopped with an error."""
    path = folder / "run_status.txt"
    if not path.exists():
        return "not run"
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").split("\n")
             if line.strip()]
    if not lines:
        return "empty status"
    if lines[0] == "stopped with an error" and len(lines) > 1:
        return f"{lines[0]}: {lines[-1]}"
    return lines[0]


def script_folders(run: Path) -> dict[str, Path]:
    return {p.name: p for p in run.iterdir() if p.is_dir() and ".partial." not in p.name}


def figures(folder: Path) -> dict[str, Path]:
    return {str(p.relative_to(folder)): p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in FIGURES}


def printed(path: Path, masks: list[re.Pattern]) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for mask in masks:
        text = mask.sub("<masked>", text)
    return text.split("\n")


def compare_script(a: Path, b: Path, masks: list[re.Pattern]) -> tuple[bool, list[str]]:
    """Return whether two runs of one script match, and one line per thing compared."""
    status_a, status_b = run_status(a), run_status(b)
    same = status_a == status_b
    lines = [f"status: {status_a} -> {status_b}"]
    figs_a, figs_b = figures(a), figures(b)
    for name in sorted(figs_a.keys() | figs_b.keys()):
        if name not in figs_a or name not in figs_b:
            lines.append(f"- {name}: only in the {'first' if name in figs_a else 'second'} run")
            same = False
        elif figs_a[name].read_bytes() == figs_b[name].read_bytes():
            lines.append(f"- {name}: byte-identical")
        elif name.lower().endswith(".svg"):
            differing = count_differing_lines(normalise_svg(figs_a[name]), normalise_svg(figs_b[name]))
            lines.append(f"- {name}: " + ("identical after normalising" if differing == 0
                                          else f"{differing} lines differ after normalising"))
            same = same and differing == 0
        else:
            lines.append(f"- {name}: bytes differ")
            same = False
    log_a, log_b = a / "stdout.log", b / "stdout.log"
    if log_a.exists() and log_b.exists():
        log_same = printed(log_a, masks) == printed(log_b, masks)
        lines.append(f"- stdout.log: {'identical' if log_same else 'differs'}")
        same = same and log_same
    elif log_a.exists() or log_b.exists():
        lines.append(f"- stdout.log: only in the {'first' if log_a.exists() else 'second'} run")
        same = False
    return same, lines


def compare_runs(first: Path, second: Path, masks: list[re.Pattern]) -> tuple[dict[str, list[str]], str]:
    runs_a, runs_b = script_folders(first), script_folders(second)
    groups: dict[str, list[str]] = {"same, finished": [], "same, did not finish": [], "differs": [],
                                    "only in the first run": [], "only in the second run": []}
    details = []
    for name in sorted(runs_a.keys() | runs_b.keys()):
        if name not in runs_b:
            groups["only in the first run"].append(name)
            details.append(f"## {name}\nonly in the first run ({run_status(runs_a[name])})\n")
        elif name not in runs_a:
            groups["only in the second run"].append(name)
            details.append(f"## {name}\nonly in the second run ({run_status(runs_b[name])})\n")
        else:
            same, lines = compare_script(runs_a[name], runs_b[name], masks)
            if not same:
                group = "differs"
            elif run_status(runs_a[name]) == "finished":
                group = "same, finished"
            else:
                group = "same, did not finish"
            groups[group].append(name)
            details.append(f"## {name}\n" + "\n".join(lines) + "\n")
    head = [f"# {first} vs {second}", ""]
    head += [f"- {group} ({len(names)}): {', '.join(names) if names else '-'}"
             for group, names in groups.items()]
    return groups, "\n".join(head) + "\n\n" + "\n".join(details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("first", type=Path, help="output folder of the first full run")
    parser.add_argument("second", type=Path, help="output folder of the second full run")
    parser.add_argument("--report", type=Path, default=None,
                        help="also write the full report to this markdown file")
    parser.add_argument("--mask", action="append", default=[], metavar="REGEX",
                        help="replace matches of this regular expression in both printed outputs "
                             "before comparing them (can be given more than once)")
    args = parser.parse_args()
    for folder in (args.first, args.second):
        if not folder.is_dir():
            parser.error(f"{folder} is not a folder")
    try:
        masks = [re.compile(pattern) for pattern in args.mask]
    except re.error as exc:
        parser.error(f"--mask is not a valid regular expression: {exc}")
    groups, report = compare_runs(args.first, args.second, masks)
    for group, names in groups.items():
        shown = f" ({', '.join(names)})" if names and group != "same, finished" else ""
        print(f"{group}: {len(names)}{shown}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"report written to {args.report}")
    sys.exit(0 if not groups["differs"] and not groups["only in the first run"]
             and not groups["only in the second run"] else 1)


if __name__ == "__main__":
    main()
