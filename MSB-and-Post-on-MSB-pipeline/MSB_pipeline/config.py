from __future__ import annotations

from pathlib import Path


class Config:
    """Configuration values for the MSB clustering pipeline."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        package_dir = Path(__file__).resolve().parent
        default_root = package_dir.parent
        self.project_root = Path(project_root or default_root).resolve()

        data_dir = package_dir.parent / "data"

        self.swc_dir = self._resolve_existing_dir(
            data_dir / "swc",
            self.project_root / "data" / "swc",
            package_dir / "swc",  # fallback fallback
        )
        self.synapse_table_path = self._resolve_existing_file(
            data_dir / "sample_synapses.ftr",
            self.project_root / "data" / "sample_synapses.ftr",
            package_dir / "sample_synapses.ftr",  # fallback
        )
        self.output_path = data_dir / "cluster_info_output.ftr"

        self.min_score = 30
        self.min_samples = 3
        self.dbscan_eps = 900
        self.merge_distance_threshold = 450
        self.matching_distance_threshold = 450
        self.neuron_distance_threshold = 10_000
        self.attach_distance_threshold = 10_000

    @staticmethod
    def _resolve_existing_dir(*paths: Path) -> Path:
        for path in paths:
            if path.exists():
                return path
        return paths[0]

    @staticmethod
    def _resolve_existing_file(*paths: Path) -> Path:
        for path in paths:
            if path.exists():
                return path
        return paths[0]
