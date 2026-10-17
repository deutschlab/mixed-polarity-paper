"""Make, or check, the seven figures/fig7/fig7*.py scripts from the authors' original Figure 7 scripts.

The seven panel scripts figures/fig7/fig7e_..., fig7f_..., fig7g_..., fig7i_..., fig7k_...,
fig7s2a_... and fig7s2b_... are generated from the authors' original scripts (fig7e.py,
fig7f_fig7S4c.py, fig7g.py, fig7i.py, fig7k.py, fig7S2a.py, fig7S2b.py), which are not in this
repository. Each generated script is the original text with only these changes:
  - a header comment (panel, origin, changes, notes) and the repository preamble (config import,
    output folder);
  - every Windows path replaced by its config.py constant (reads and the one table write) or by
    _out_dir (figure saves);
  - figure saves that were commented out switched on, and every save placed before plt.show();
  - svg.fonttype='none' and Arial added where the original did not set them;
  - the per-script fixes listed in PORTS below, each of which must match the original exactly once.
So a change to one of these scripts belongs in PORTS here, never in the script itself: a
regeneration would undo it.

Modes:
  (default)  check: generate into memory and compare, byte for byte, with the scripts in --ports
             (default figures/fig7/). PASS means every script is exactly what this tool makes from
             the originals, so nobody has edited one by hand and every fix is still applied.
             A difference prints a unified diff.
  --write    write the generated scripts into --ports. Refused unless every original's sha256
             equals ORIGINAL_SHA256 below (the files the ports were made from).
Both modes also check, for every port: that each original's sha256 equals ORIGINAL_SHA256 below
(the files the ports were made from); that each fix and each path replacement applied, and every
file name read or written has a config constant; that no Windows path (any drive letter) is left;
that the figure saves switched on are exactly EXPECTED_SAVES; that each save comes before the
next plt.show(); and that svg.fonttype 'none' and Arial are set. --selftest plants mistakes in
the fix list (a save after plt.show(), the Arial line left out, a fix that no longer matches)
and requires each to be reported. It prints its result and writes no report file.
Exit code 0 on PASS, 1 on FAIL.

Run:
    .venv/bin/python tools/make_fig7_ports.py --originals <folder with the seven original scripts>
    .venv/bin/python tools/make_fig7_ports.py --originals <folder> --write
--originals defaults to the environment variable FIG7_ORIGINALS.
tools/check_fig7_ports.py imports READS, WRITES, PAIRS and EXPECTED_SAVES from here.
"""
import argparse
import difflib
import hashlib
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PORTS_DIR = REPO_ROOT / "figures" / "fig7"

# file name an original reads -> the config.py constant the port reads instead
READS = {
    "neuron_data.ftr": "NEURON_DATA_FTR",
    "t2.ftr": "SYNAPSES_NO_SELF_FTR",
    "b_table_full_prince(new)_filtered.ftr": "BOUTON_TABLE_FTR",
    "bwf_table_full_prince(new)_filtered.ftr": "BOUTON_FILOPODIA_TABLE_FTR",
    "filopodia_filtered.ftr": "FILOPODIA_FILTERED_FTR",
    "filopodia_reciprocal.ftr": "FILOPODIA_RECIPROCAL_FTR",
    "inertia_all.ftr": "BOUTON_INERTIA_FTR",
    "consolidated_cell_types.csv": "CONSOLIDATED_CELL_TYPES_CSV",
    "neuron_data_full_article_princeton.ftr": "NEURON_TABLE_FTR",
}
# file name an original writes -> the config.py constant the port writes instead
WRITES = {"neurons_nt_bwf_frac_filtered.ftr": "NEURONS_NT_BWF_FTR"}

# sha256 of the originals the ports were made from (received 15 Oct 2026)
ORIGINAL_SHA256 = {
    "fig7e.py": "a39bd0704e54dccb271e3c142e0f39df4be1204071cccafc38dca5d66064fa43",
    "fig7f_fig7S4c.py": "e92211a212750d085c7b5328a0d1b548d69c410757423279542377f78a516a1a",
    "fig7g.py": "3de9a6d9a83530483fea26c8b24068307c833151cf1b2c1a5a7b505190c07889",
    "fig7i.py": "7193e7524011dd6c318d262d4fd3c8912dff68b66d4bcae3e528baff4ac2f7d3",
    "fig7k.py": "24c2e7937a2b35b9769b72e17147700bd3874428adc72577039d7cacb12cbcfa",
    "fig7S2a.py": "1bd5ea3961b843263185a202b5aa4a76a18cf2f728f71c60b207ecf066187308",
    "fig7S2b.py": "68c4244e4ebcdc56e02c7d8c619b87a77b9618df15760d082d863aebeb9ed998",
}

# the figure files each port saves, in order (the saves this tool switches on)
EXPECTED_SAVES = {
    "fig7e.py": ["pie_plot_nt_post_frac_and_bwf_3_newest.svg"],
    "fig7f_fig7S4c.py": ["super_fractions_split2_new(2).svg"],
    "fig7g.py": ["SI_vs_fraction_all_new.svg", "SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg"],
    "fig7i.py": ["nt_type_frac_new2.svg"],
    "fig7k.py": ["super_classes_reciprocal_new.svg"],
    "fig7S2a.py": ["inertia_all.svg"],
    "fig7S2b.py": ["inertia_grouped.svg"],
}

# the figure each save writes (matplotlib's figure number in the original)
EXPECTED_SAVE_FIGS = {name: 1 for names in EXPECTED_SAVES.values() for name in names}
EXPECTED_SAVE_FIGS["SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg"] = 2

FONT = "mpl.rcParams['svg.fonttype'] = 'none'\nmpl.rcParams['font.family'] = 'Arial'\n"

# original -> (port name, panel line, notes, exact (old, new) fixes applied after the path changes)
PORTS = {
    "fig7e.py": (
        "fig7e_nt_presynaptic_pies.py",
        "Fig 7E: neurotransmitter pies, 'Neuron presynaptic to bouton' (left, from FILOPODIA_FILTERED_FTR) against\n"
        "# 'Any presynaptic neuron (control)' (right: every synapse whose presynaptic neuron has boutons).",
        ["Cannot run until SYNAPSES_NO_SELF_FTR (t2.ftr) and FILOPODIA_FILTERED_FTR are supplied; neither is in this repository.",
         "Both pies count table rows (synapses), not neurons. The left pie keeps every row of FILOPODIA_FILTERED_FTR,",
         "whatever its presynaptic neuron; the right pie only synapses whose presynaptic neuron is in BOUTON_TABLE_FTR."],
        [("plt.show()\nplt.savefig(_out_dir / 'pie_plot_nt_post_frac_and_bwf_3_newest.svg', format='svg')",
          "plt.savefig(_out_dir / 'pie_plot_nt_post_frac_and_bwf_3_newest.svg', format='svg')\nplt.show()")]),
    "fig7f_fig7S4c.py": (
        "fig7f_filopodia_fraction_violins.py",
        "Fig 7F (left half: intrinsic super-classes) and Supp 7-S4 C (right half: the other super-classes): "
        "per-neuron fraction of boutons with a postsynaptic terminal.",
        ["Each violin value is one neuron and compartment (the mean of 'filopodia' over its boutons), with no minimum bouton count.",
         "The 'All' violins use every compartment row with no SI filter; the axon and dendrite violins keep SI >= 0.1.",
         "Boutons with no compartment are left out of the violins but count in the bouton-level means that order the super-classes.",
         "Positions 3 and 4 of the non-intrinsic order are typed in ('sensory', 'ascending'); on these tables that is already the computed order."],
        [("plt.show()\nplt.savefig(\n    _out_dir / 'super_fractions_split2_new(2).svg',\n    format='svg'\n)",
          "plt.savefig(\n    _out_dir / 'super_fractions_split2_new(2).svg',\n    format='svg'\n)\nplt.show()")]),
    "fig7g.py": (
        "fig7g_filopodia_fraction_x_si.py",
        "Fig 7G: fraction of boutons with a postsynaptic terminal (left) and postsynaptic sites per bouton, among boutons "
        "with at least one (right), against SI.",
        ["The second plot had two commented-out saves under different names; the first is switched on, the second (after plt.show()) stays off.",
         "The dashed lines at 0.24 and 0.75 are typed in; the 'All' lines use every super-class with no SI filter.",
         "The axon and dendrite lines keep intrinsic boutons with SI >= 0.1 (left) or SI > 0.1 (right), binned in equal-count",
         "SI bins over axon and dendrite boutons together; r and p are computed over boutons, not neurons."],
        [("plt.show()\n# plt.savefig(\n#     _out_dir / 'SI_vs_fraction_all_new.svg',\n#     format='svg'\n# )",
          "plt.savefig(\n    _out_dir / 'SI_vs_fraction_all_new.svg',\n    format='svg'\n)\nplt.show()"),
         ("# plt.savefig(\n#     _out_dir / 'SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg',\n#     format='svg'\n# )",
          "plt.savefig(\n    _out_dir / 'SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg',\n    format='svg'\n)")]),
    "fig7i.py": (
        "fig7i_filopodia_fraction_x_nt.py",
        "Fig 7I: per-neuron fraction of boutons with a postsynaptic terminal, by neurotransmitter.",
        ["It writes NEURONS_NT_BWF_FTR, which figures/fig7/filopodia_x_mirror_neurons.py and figures/fig5/reciprocal_fraction_x_bwf.py read;",
         "the original wrote neurons_nt_bwf_frac_filtered.ftr. Its neuron column is pandas' nullable Int64, as in the original.",
         "Its neuron set is every bouton neuron with a neurotransmitter in NEURON_DATA_FTR, all super-classes, no SI filter.",
         "Each neuron's fraction pools its axonal and dendritic boutons, with no minimum bouton count."],
        [("plt.show()\n# plt.savefig(_out_dir / 'nt_type_frac_new2.svg', format='svg')",
          "plt.savefig(_out_dir / 'nt_type_frac_new2.svg', format='svg')\nplt.show()")]),
    "fig7k.py": (
        "fig7k_reciprocal_composition.py",
        "Fig 7K: share of postsynaptic terminals on boutons that are 'hand shake', other reciprocal, or not reciprocal,\n"
        "# by super-class of the neuron that owns the bouton (the postsynaptic neuron of these synapses).",
        ["Reads super_class from NEURON_TABLE_FTR. The original read a copy of that table whose super_class is the same for all 138,289 neurons (only its nt_type column differs, and it is not used).",
         "The bars pool terminals (synapses) across neurons and connections."],
        [("# plt.savefig(\n#     _out_dir / 'super_classes_reciprocal_new.svg',\n#     format=\"svg\"\n# )\n\nplt.show()",
          "plt.savefig(\n    _out_dir / 'super_classes_reciprocal_new.svg',\n    format=\"svg\"\n)\n\nplt.show()")]),
    "fig7S2a.py": (
        "fig7s2a_bouton_spread.py",
        "Supp 7-S2 A: mean RMS against mean centroid distance of each neuron's boutons.",
        ["It plots every neuron of BOUTON_INERTIA_FTR, before any cell type is excluded; it reads CONSOLIDATED_CELL_TYPES_CSV but does not use it."],
        [("plt.show()\n# plt.savefig(_out_dir / 'inertia_all.svg', format='svg')",
          "plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\nplt.show()")]),
    "fig7S2b.py": (
        "fig7s2b_bouton_spread_by_type.py",
        "Supp 7-S2 B: the same per cell type, coloured by why a type was excluded from the bouton analysis.",
        ["Fix: the cell-type table names its id column root_id; renamed to neuron so the merge runs (the original stopped with a KeyError).",
         "Not changed: the excluded-type lists write 'R1\u20136' with an en dash, while the table has 'R1-6', so R1-6 is coloured 'included'."],
        [("neuron_types = pd.read_csv(CONSOLIDATED_CELL_TYPES_CSV)\n",
          "neuron_types = pd.read_csv(CONSOLIDATED_CELL_TYPES_CSV)\nneuron_types = neuron_types.rename(columns={'root_id': 'neuron'})\n"),
         ("plt.show()\n# plt.savefig(_out_dir / 'inertia_grouped.svg', format='svg')",
          "plt.savefig(_out_dir / 'inertia_grouped.svg', format='svg')\nplt.show()")]),
}
PAIRS = {orig: spec[0] for orig, spec in PORTS.items()}

PATH_RE = re.compile(r"r(['\"])C:\\[^'\"]*\\([^\\'\"]+)\1")
SAVE_RE = re.compile(r"^[^#\n]*savefig\(\s*_out_dir / '([^']+)'", re.M)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(originals: Path, src_name: str, ledger: list) -> str:
    """Return the port text of one original; append (port, what, count) to ledger (count 1 = ok)."""
    new_name, panel, notes, fixes = PORTS[src_name]
    text = (originals / src_name).read_text(encoding="utf-8")
    text = re.sub(r'\A(\n)?# -\*- coding: utf-8 -\*-\n"""\nCreated on [^\n]*\n\n@author: [^\n]*\n"""\n', "", text)
    used = set()

    def sub(m):
        name = m.group(2)
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start:text.find("\n", m.end())]
        if name.endswith(".svg"):
            return "_out_dir / '%s'" % name
        const = (WRITES if ".to_feather(" in line else READS).get(name)
        if const is None:
            ledger.append((new_name, "no config constant for the file name %s" % name, 0))
            return m.group(0)
        used.add(const)
        return const

    text, n_paths = PATH_RE.subn(sub, text)
    for old, new in fixes:
        c = text.count(old)
        ledger.append((new_name, "fix: " + old.splitlines()[0][:50], c))
        if c == 1:
            text = text.replace(old, new)
    ledger.append((new_name, "no Windows path left", 0 if re.search(r"\b[A-Za-z]:[\\/]", text) else 1))
    has_font = "svg.fonttype" in text
    head = ["# " + panel,
            "# Ported from the authors' original script %s." % src_name,
            "# Changes: paths via config.py; figure saves switched on and placed before plt.show()"
            + ("" if has_font else "; svg.fonttype='none' and Arial added (repository convention)") + "."]
    head += ["# " + n for n in notes]
    pre = ["import sys", "from pathlib import Path", "sys.path.insert(0, str(Path(__file__).parent.parent.parent))",
           "from config import %s" % ", ".join(sorted(used | {"OUTPUT_DIR"})), ""]
    if not has_font:
        pre += ["import matplotlib as mpl", FONT.rstrip("\n"), ""]
    pre += ['_out_dir = OUTPUT_DIR / "fig7" / "%s"' % new_name[:-3], "_out_dir.mkdir(parents=True, exist_ok=True)", ""]
    out = "\n".join(head + pre) + "\n" + text.lstrip("\n")
    ledger.append((new_name, "paths replaced: %d" % n_paths, 1 if n_paths else 0))
    saves = SAVE_RE.findall(out)
    ledger.append((new_name, "saves switched on: %s" % saves, 1 if saves == EXPECTED_SAVES[src_name] else 0))
    # each save comes before the next plt.show(), and the editable-text settings are there
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^[^#]*savefig\(", line):
            prev = [l for l in lines[:i] if not l.lstrip().startswith("#")]
            last_show = max((j for j, l in enumerate(prev) if "plt.show()" in l), default=-1)
            last_fig = max((j for j, l in enumerate(prev) if re.search(r"plt\.(figure|subplots)\(", l)), default=-1)
            ledger.append((new_name, "save at line %d comes before plt.show()" % (i + 1), 1 if last_show < last_fig or last_show < 0 else 0))
    for setting in ("mpl.rcParams['svg.fonttype'] = 'none'", "mpl.rcParams['font.family'] = 'Arial'"):
        ledger.append((new_name, "sets " + setting, 1 if setting in out else 0))
    return out


def resolve_originals(arg) -> Path:
    path = arg or os.environ.get("FIG7_ORIGINALS")
    if not path:
        sys.exit("FAIL: give --originals DIR or set FIG7_ORIGINALS (the folder with the seven original scripts)")
    path = Path(path)
    missing = [n for n in PORTS if not (path / n).is_file()]
    if missing:
        sys.exit("FAIL: not in %s: %s" % (path, ", ".join(missing)))
    return path


def check(originals: Path, ports: Path, write: bool = False, quiet: bool = False) -> list:
    """Generate every port; compare with (or write into) ports. Return the list of problems."""
    ledger, problems, generated = [], [], {}
    for src in PORTS:
        if sha256(originals / src) != ORIGINAL_SHA256[src]:
            problems.append("%s: sha256 differs from the original the ports were made from" % src)
        generated[PORTS[src][0]] = generate(originals, src, ledger)
    problems += ["%s: %s (matched %d times)" % (n, w, c) for n, w, c in ledger if c != 1]
    if write:
        if problems:
            return problems + ["nothing written"]
        for name, text in generated.items():
            (ports / name).write_text(text, encoding="utf-8")
        return []
    for name, text in generated.items():
        target = ports / name
        if not target.exists():
            problems.append("%s: not found in %s" % (name, ports))
            continue
        current = target.read_text(encoding="utf-8")
        if current != text:
            d = list(difflib.unified_diff(current.splitlines(), text.splitlines(), "in " + str(ports), "generated", lineterm="", n=1))
            problems.append("%s: differs from what this tool makes (%d diff lines)" % (name, len(d)))
            if not quiet:
                problems += ["    " + line for line in d[:40]]
    return problems


def selftest(originals: Path) -> list:
    """Generate with planted mistakes in this tool's own fix list; each must be reported."""
    global PORTS, FONT
    saved_ports, saved_font = dict(PORTS), FONT
    cases = []
    try:
        name, panel, notes, fixes = saved_ports["fig7S2a.py"]
        old, new = fixes[0]
        PORTS["fig7S2a.py"] = (name, panel, notes, [(old, new.replace("plt.savefig(_out_dir / 'inertia_all.svg', format='svg')\nplt.show()",
                                                                  "plt.show()\nplt.savefig(_out_dir / 'inertia_all.svg', format='svg')"))])
        cases.append(("save placed after plt.show()", "comes before plt.show()"))
        ledger = []
        generate(originals, "fig7S2a.py", ledger)
        got1 = [w for n, w, c in ledger if c != 1]
        PORTS = dict(saved_ports)
        FONT = "mpl.rcParams['svg.fonttype'] = 'none'\n"
        cases.append(("Arial line left out", "font.family"))  # fig7k.py sets no font itself, so the tool adds both lines
        ledger = []
        generate(originals, "fig7k.py", ledger)
        got2 = [w for n, w, c in ledger if c != 1]
        PORTS = dict(saved_ports)
        FONT = saved_font
        name, panel, notes, fixes = saved_ports["fig7k.py"]
        PORTS["fig7k.py"] = (name, panel, notes, [(fixes[0][0] + "x", fixes[0][1])])
        cases.append(("a fix that no longer matches", "fix:"))
        ledger = []
        generate(originals, "fig7k.py", ledger)
        got3 = [w for n, w, c in ledger if c != 1]
    finally:
        PORTS, FONT = saved_ports, saved_font
    lines, missed = [], 0
    for (label, must), got in zip(cases, (got1, got2, got3)):
        ok = any(must in g for g in got)
        missed += not ok
        lines.append("%-7s %-32s %s" % ("caught" if ok else "MISSED", label, "; ".join(got)[:150]))
    lines.append("PASS: all %d planted mistakes reported" % len(cases) if not missed else "FAIL: %d not reported" % missed)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--originals", help="folder with the seven original scripts (default: $FIG7_ORIGINALS)")
    parser.add_argument("--ports", type=Path, default=PORTS_DIR, help="folder of the ports (default figures/fig7)")
    parser.add_argument("--write", action="store_true", help="write the ports instead of checking them")
    parser.add_argument("--selftest", action="store_true", help="plant mistakes in the fix list and require each to be reported")
    args = parser.parse_args()
    originals = resolve_originals(args.originals)
    if args.selftest:
        lines = selftest(originals)
        print("\n".join(lines))
        sys.exit(0 if lines[-1].startswith("PASS") else 1)
    problems = check(originals, args.ports, write=args.write)
    for p in problems:
        print(p)
    if problems:
        print("FAIL")
        sys.exit(1)
    print("PASS: %d ports %s" % (len(PORTS), "written" if args.write else "are exactly what this tool makes"))


if __name__ == "__main__":
    # Reads the seven original Figure 7 scripts from the folder given with --originals (not in this repository),
    # and compares what it makes with
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/figures/fig7/  (the seven fig7*.py ports)
    # With --write it overwrites those seven ports. It writes nothing else.
    # To run it, uncomment:
    # sys.argv[1:] = [
    #     "--originals", "<folder with the seven original scripts>",
    # ]
    main()
