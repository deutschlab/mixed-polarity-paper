from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .clustering import append_matched_synapses, prepare_clustering
    from .config import Config
    from .utils import iter_swc_files, load_synapse_table
except ImportError:  # pragma: no cover - allows running the file directly
    from clustering import append_matched_synapses, prepare_clustering
    from config import Config
    from utils import iter_swc_files, load_synapse_table


PACKAGE_ROOT = Path(__file__).resolve().parent


def run_pipeline(config: Config | None = None) -> pd.DataFrame:
    cfg = config or Config()
    allsynapses = load_synapse_table(cfg.synapse_table_path)

    clustered_frames = []
    for swc_file in iter_swc_files(cfg.swc_dir):
        swc_id = int(swc_file.stem)
        if swc_id not in allsynapses["pre"].values:
            continue

        clustered_neuron = prepare_clustering(allsynapses, swc_id, config=cfg)
        if clustered_neuron is None:
            continue
        clustered_frames.append(clustered_neuron)

    if not clustered_frames:
        raise ValueError("No neurons were clustered. Check the SWC directory and synapse table.")

    combined_df = pd.concat(clustered_frames, ignore_index=True)
    combined_df["neuron"] = combined_df["neuron"].astype("Int64")
    combined_df = append_matched_synapses(
        combined_df,
        allsynapses,
        distance_threshold=cfg.matching_distance_threshold,
    )
    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MSB clustering pipeline")
    parser.add_argument("--project-root", type=str, default=None, help="Path to the project directory")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clustered_df = run_pipeline(cfg)
    clustered_df.to_feather(output_path)
    print(f"Wrote {len(clustered_df)} clustered synapses to {output_path}")


if __name__ == "__main__":
    main()
