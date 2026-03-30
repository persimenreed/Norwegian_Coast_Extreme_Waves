from pathlib import Path

from src.extreme_value_modelling.common import dataset_name, evt_root
from src.settings import format_path


def resolve_input_path(location: str, mode: str, corr_method: str = "pqm", transfer_source: str | None = None) -> Path:
    mode = str(mode).strip().lower()
    if mode == "raw":
        return Path(format_path("hindcast_raw", location=location))
    if mode != "corrected":
        raise ValueError("mode must be 'raw' or 'corrected'")

    return Path(
        format_path(
            "corrected",
            location=location,
            corr_method=dataset_name("corrected", corr_method=corr_method, transfer_source=transfer_source),
        )
    )


def resolve_preprocessing_dir(location: str) -> Path:
    out = evt_root() / location / "preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_output_dir(location: str, dataset: str) -> Path:
    return evt_root() / location / dataset


def resolve_diagnostics_dir(location: str) -> Path:
    return evt_root() / location / "diagnostics"
