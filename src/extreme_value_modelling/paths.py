from pathlib import Path
from src.settings import format_path, get_path_template
from src.extreme_value_modelling.common import dataset_name


def normalize_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in {"raw", "corrected"}:
        raise ValueError("mode must be 'raw' or 'corrected'")
    return m


def resolve_input_path(
    location: str,
    mode: str,
    corr_method: str = "pqm",
    pooling: bool = False,
    transfer_source: str | None = None,
) -> Path:
    mode = normalize_mode(mode)

    if mode == "raw":
        return Path(format_path("hindcast_raw", location=location))

    ds = dataset_name(
        mode="corrected",
        corr_method=corr_method,
        pooling=pooling,
        transfer_source=transfer_source,
    )
    return Path(f"data/output/{location}/hindcast_corrected_{ds}.csv")


def resolve_output_dir(location: str, dataset: str) -> Path:
    root = Path(get_path_template("evt_results_root"))
    return root / location / dataset


def resolve_diagnostics_dir(location: str) -> Path:
    root = Path(get_path_template("evt_results_root"))
    return root / location / "diagnostics"