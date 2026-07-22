from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def iter_swc_files(swc_dir: str | Path | None = None) -> Iterator[Path]:
    base_dir = Path(swc_dir or (get_project_root().parent / "swc")).resolve()
    return sorted(base_dir.rglob("*.swc"))


def load_synapse_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path).resolve()
    if not table_path.exists():
        raise FileNotFoundError(f"Synapse table not found: {table_path}")
    return pd.read_feather(table_path)
