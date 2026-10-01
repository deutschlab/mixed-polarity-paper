"""Run every figure script with run_figure.py, one after another.

Each script under figures/ runs in its own process through tools/run_figure.py and
writes into its own subfolder of the output folder, named after the script. Some
scripts read what others write, so those run first, in this order:
figures/fig4/build_pc1_table.py (read by figures/fig4/syntype_x_pc1.py),
figures/fig5/reciprocal_fraction.py (read by figures/fig5/reciprocal_fraction_model.py)
and the three figures/fig7/si_sim_*.py scripts (read by figures/fig7/si_comb_analysis.py).
The rest run in name order.

Each script has a time limit: 3 hours for the few known to be slow, 20 minutes for the
rest (--timeout and --long-timeout change them). A script that runs out of time is
stopped together with any worker processes it started, and its run_status.txt says
"timed out". What a script prints goes to <name>.console next to its folder, also when
it times out. progress.txt gets one line per script (name, exit code, seconds, status)
and a summary line at the end of each invocation; commit.txt gets the git commit and
whether the working tree had uncommitted changes; inputs.txt lists the size and
modification time of every file under data/raw and data/derived when the run started,
so that a difference between two runs can be traced to changed input data.

Running the command again with the same --out continues an interrupted run. Scripts
that finished, or stopped with an error, are skipped; a script that timed out, was
interrupted, or has no status is run again, after its folder is renamed to
<name>.partial.<time>. --rerun-failed also runs again the scripts that stopped with
an error.

Scripts that write tables still write them in their usual place, replacing what is
there, exactly as when they are run by hand: among them build_pc1_table.py rewrites
data/derived/PC1_table.csv (about 3 GB), reciprocal_fraction_model.py saves its model,
and si_x_pca.py writes full_info.ftr under outputs/.

The command exits with status 1 if any script did not finish. On this repository,
scripts whose input tables are not supplied always stop with an error.

Usage:
    python tools/run_all_figures.py --out outputs/runs/before
    python tools/run_all_figures.py --out outputs/runs/after --only fig4
    python tools/compare_runs.py outputs/runs/before outputs/runs/after

On a 36 GB Mac most scripts took seconds and syntype_x_features.py about an hour;
si_x_primary_types.py (6 to 10 hours, its Games-Howell tests) and
reciprocal_fraction_model.py did not finish within 20 minutes.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FIRST = ["figures/fig4/build_pc1_table.py", "figures/fig5/reciprocal_fraction.py",
         "figures/fig7/si_sim_baseline.py", "figures/fig7/si_sim_no_filopodia.py",
         "figures/fig7/si_sim_sampleout.py"]
SLOW = {"figures/fig3/syntype_x_features.py", "figures/fig4/syntype_x_pc1.py",
        "figures/fig4/build_pc1_table.py", "figures/fig3/fig3supp_ad_content_si_mixed.py"}
DONE = ("finished", "stopped with an error")


def figure_scripts(only: str | None) -> list[str]:
    scripts = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "figures").rglob("*.py"))
    scripts = [s for s in FIRST if s in scripts] + [s for s in scripts if s not in FIRST]
    if only:
        scripts = [s for s in scripts if only in s]
    return scripts


def status_of(folder: Path) -> str:
    status = folder / "run_status.txt"
    if not status.exists():
        return "no status"
    return status.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]


def record_run(out: Path) -> None:
    """Append the commit and the state of the input data to commit.txt and inputs.txt."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
    with open(out / "commit.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {commit.stdout.strip() or 'no git commit'}"
                 f"{' (uncommitted changes)' if dirty.stdout.strip() else ''}\n")
    with open(out / "inputs.txt", "a", encoding="utf-8") as fh:
        fh.write(f"# {stamp}\n")
        for folder in (REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "derived"):
            for path in sorted(p for p in folder.rglob("*") if p.is_file()) if folder.exists() else []:
                info = path.stat()
                fh.write(f"{path.relative_to(REPO_ROOT)}\t{info.st_size}\t"
                         f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.st_mtime))}\n")


def run_one(script: str, folder: Path, limit: int, dpi: float) -> tuple[str, str]:
    """Run one script; return its exit code (or 'timeout') and what it printed."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    child = subprocess.Popen([sys.executable, "tools/run_figure.py", script, "--out", str(folder),
                              "--dpi", str(dpi)], cwd=REPO_ROOT, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    try:
        printed, _ = child.communicate(timeout=limit)
        return str(child.returncode), printed
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)       # the script and any workers it started
        printed, _ = child.communicate()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run_status.txt").write_text("timed out\n", encoding="utf-8")
        return "timeout", printed


def run_all(out: Path, scripts: list[str], timeout: int, long_timeout: int, dpi: float,
            rerun_failed: bool) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    record_run(out)
    counts: dict[str, int] = {}
    with open(out / "progress.txt", "a", encoding="utf-8") as progress:
        for script in scripts:
            name = Path(script).stem
            folder = out / name
            status = status_of(folder)
            if status == "finished" or (status in DONE and not rerun_failed):
                print(f"{name}: already run ({status}), skipped")
                counts[status] = counts.get(status, 0) + 1
                continue
            if folder.exists():
                folder.rename(out / f"{name}.partial.{int(time.time())}")
            print(f"{name}: running", flush=True)
            start = time.time()
            code, printed = run_one(script, folder, long_timeout if script in SLOW else timeout, dpi)
            (out / f"{name}.console").write_text(printed or "", encoding="utf-8")
            status = status_of(folder)
            counts[status] = counts.get(status, 0) + 1
            line = f"{name} exit={code} {time.time() - start:.0f}s {status}"
            progress.write(line + "\n")
            progress.flush()
            print(line, flush=True)
        summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
        progress.write(f"END {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(scripts)} scripts: {summary}\n")
    print(f"{len(scripts)} scripts: {summary}; progress in {out / 'progress.txt'}")
    return set(counts) <= {"finished"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="output folder; reuse it to continue an interrupted run")
    parser.add_argument("--only", default=None,
                        help="run only the scripts whose path contains this text, e.g. fig4")
    parser.add_argument("--timeout", type=int, default=20 * 60,
                        help="time limit per script, in seconds (default: 1200)")
    parser.add_argument("--long-timeout", type=int, default=3 * 3600,
                        help="time limit for the known slow scripts, in seconds (default: 10800)")
    parser.add_argument("--dpi", type=float, default=72,
                        help="figure dpi, passed to run_figure.py (default: 72)")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="also run again the scripts that stopped with an error")
    args = parser.parse_args()
    scripts = figure_scripts(args.only)
    if not scripts:
        parser.error(f"no figure script path contains {args.only!r}")
    ok = run_all(args.out.resolve(), scripts, args.timeout, args.long_timeout, args.dpi,
                 args.rerun_failed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
