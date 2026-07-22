from __future__ import annotations

from pathlib import Path


class Config:
    """Configuration values for the Post-on-MSB filopodia pipeline.

    Resolves the input/output paths that the pipeline reads and writes. The
    inputs live in the shared ``data`` directory one level above this package
    (alongside the ``MSB_pipeline`` package), matching the layout used by
    ``MSB_pipeline``.
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        package_dir = Path(__file__).resolve().parent
        default_root = package_dir.parent
        self.project_root = Path(project_root or default_root).resolve()

        # Shared data directory at the repository root (one level up), the same
        # folder that MSB_pipeline reads from and writes to.
        data_dir = package_dir.parent / "data"
        self.data_dir = data_dir

        self.swc_dir = self._resolve_existing_dir(
            data_dir / "swc",
            self.project_root / "data" / "swc",
        )
        self.synapse_table_path = self._resolve_existing_file(
            data_dir / "sample_synapses.ftr",
            self.project_root / "data" / "sample_synapses.ftr",
        )
        self.cluster_info_path = self._resolve_existing_file(
            data_dir / "cluster_info_output.ftr",
            self.project_root / "data" / "cluster_info_output.ftr",
        )
        self.output_path = data_dir / "post_on_MSB.ftr"

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
