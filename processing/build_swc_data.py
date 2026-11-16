# Build swc_data.ftr: four shape measurements per skeleton, which step 05 adds to the neuron table (and step 07 uses
# leafs as a PCA feature). Run it before step 05.
#
# For every skeleton under SWC_DIR (one file per neuron, <root_id>.swc, in subfolders), the script reads the file with
# navis.read_swc, as it comes (no healing, pruning or resampling), and records:
#   root_id       the cell ID, from the file name, as int64 (the id in the file's header must be the same)
#   nodes         number of skeleton nodes (n_nodes)
#   leafs         number of end nodes (n_leafs); the root is never counted, even with a single child
#   branches      number of branch nodes (n_branches); the root is never counted, even with several children
#   cable_length  total cable length in the skeleton's units, nanometres for the FlyWire 783 skeletons (float32, as
#                 navis gives it)
# Rows are sorted by root_id. The values depend on the navis version, so the script stops unless navis is 1.7.0, the
# version in requirements.txt.
#
# It stops without writing anything if:
#   - SWC_DIR holds no skeleton, or holds any file other than <subfolder>/<root_id>.swc (hidden files such as
#     .DS_Store aside), so that no file is skipped without notice;
#   - a file name is not an 18-digit FlyWire cell ID starting with 720575940, or two subfolders hold the same name;
#   - any file fails to read, or the id in its header is not its file name.
# It prints the files concerned (up to 50). It never writes over an existing file.
#
# Output: SWC_DATA_FTR, which is data/derived_rebuild/swc_data.ftr under PIPELINE_RUN = "rebuild"; under "published"
# the script stops, as steps 01 to 07 do. To build a copy somewhere else, under either setting (for example to compare
# it with another copy), set the environment variable BUILD_SWC_DATA_OUT to the output path. Next to the output it
# writes <name>_run.txt: the navis version, SWC_DIR, the number of skeletons, how many have more than one root, and
# the run time.
#
# Reads about 31 GB of skeletons, one file at a time. On 139,273 skeletons it took 713 seconds (one run, macOS).
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SWC_DIR, SWC_DATA_FTR, refuse_published_run

import numpy as np
import pandas as pd
import navis

if "BUILD_SWC_DATA_OUT" in os.environ:
    OUT = Path(os.environ["BUILD_SWC_DATA_OUT"])
else:
    refuse_published_run("build_swc_data.py")
    OUT = Path(SWC_DATA_FTR)
RECORD = OUT.with_name(OUT.stem + "_run.txt")
if navis.__version__ != "1.7.0":
    raise SystemExit(f"navis {navis.__version__} is installed; this table is built with navis 1.7.0 "
                     "(requirements.txt), and other versions can count leafs or branches differently")
for p in (OUT, RECORD):
    if p.exists():
        raise SystemExit(f"{p} exists; this script does not write over it. Delete it first, or set "
                         "BUILD_SWC_DATA_OUT to another path.")


def stop(what, items):
    for i in items[:50]:
        print(f"  {i}")
    raise SystemExit(f"{what}: {len(items):,} (up to 50 listed above); nothing written")


#%% The skeleton files
top = Path(SWC_DIR)
everything = [p for p in top.rglob("*") if p.is_file() and not p.name.startswith(".")]
files = sorted(p for p in everything if p.parent.parent == top and p.suffix == ".swc")
if not files:
    raise SystemExit(f"no skeleton found in the subfolders of {top} (files directly in it are not read)")
other = sorted(set(everything) - set(files))
if other:
    stop(f"files under {top} that are not <subfolder>/<root_id>.swc", [str(p) for p in other])
not_id = [str(f) for f in files if not (len(f.stem) == 18 and f.stem.isdigit() and f.stem.startswith("720575940"))]
if not_id:
    stop("skeleton files whose name is not an 18-digit cell ID starting with 720575940", not_id)
by_name = pd.Series([str(f.parent) for f in files], index=[f.stem for f in files])
dup = by_name[by_name.index.duplicated(keep=False)]
if len(dup):
    stop("skeleton files with the same name in more than one subfolder",
         [f"{n}: {', '.join(dup.loc[[n]])}" for n in dup.index.unique()])
n_folders = len({f.parent for f in files})
print(f"{len(files):,} skeleton files in {n_folders} subfolders of {top}", flush=True)

#%% Read each skeleton and record its measurements
rows, failed = [], []
several_roots = 0
t0 = time.time()
for k, f in enumerate(files, 1):
    try:
        swc = navis.read_swc(str(f))
        if str(swc.id) != f.stem:
            raise ValueError(f"the id in the file's header is {swc.id!r}, not its file name (a header without an "
                             "id gets a random one)")
        rows.append((int(f.stem), int(swc.n_nodes), int(swc.n_leafs), int(swc.n_branches),
                     np.float32(swc.cable_length)))
        several_roots += len(np.atleast_1d(swc.root)) > 1
    except Exception as e:  # every failure is listed below and stops the script
        cause = f" (cause: {type(e.__cause__).__name__}: {e.__cause__})" if e.__cause__ else ""
        failed.append(f"{f}: {type(e).__name__}: {e}{cause}")
    if k % 10_000 == 0:
        print(f"{k:,} of {len(files):,} read ({time.time() - t0:.0f} s)", flush=True)
if failed:
    stop(f"skeletons that failed, of {len(files):,}", failed)

#%% Write the table and the run record
df = pd.DataFrame(rows, columns=["root_id", "nodes", "leafs", "branches", "cable_length"])
df = df.astype({"root_id": np.int64, "nodes": np.int64, "leafs": np.int64, "branches": np.int64,
                "cable_length": np.float32})
df = df.sort_values("root_id", ignore_index=True)
if len(df) != len(files) or not df["root_id"].is_unique:
    raise SystemExit(f"{len(df):,} rows for {len(files):,} files, or a repeated root_id; nothing written")
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_name(OUT.name + ".tmp")
df.to_feather(tmp)
if OUT.exists():   # appeared during the run
    tmp.unlink()
    raise SystemExit(f"{OUT} appeared during the run; not written over. The table is not written.")
tmp.rename(OUT)
seconds = time.time() - t0
RECORD.write_text(f"navis {navis.__version__}\nSWC_DIR {top}\nskeletons {len(df)} in {n_folders} subfolders\n"
                  f"with more than one root {several_roots}\nseconds {seconds:.0f}\n")
print(f"wrote {OUT}: {len(df):,} skeletons, {several_roots:,} of them with more than one root ({seconds:.0f} s); "
      f"run record in {RECORD.name}")
