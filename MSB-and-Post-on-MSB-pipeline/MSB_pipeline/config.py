from __future__ import annotations

from pathlib import Path


class Config:
    """Configuration values for the MSB clustering pipeline."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        package_dir = Path(__file__).resolve().parent
        default_root = package_dir.parent
        self.project_root = Path(project_root or default_root).resolve()
        self.swc_dir = self._resolve_existing_dir(
            self.project_root / "swc",
            package_dir / "swc",
            package_dir.parent / "swc",
        )
        self.synapse_table_path = self._resolve_existing_file(
            self.project_root / "sample_synapses.ftr",
            package_dir.parent / "sample_synapses.ftr",
            package_dir / "sample_synapses.ftr",
        )
        self.output_path = package_dir.parent / "cluster_info_output.ftr"

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
