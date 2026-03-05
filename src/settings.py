import json
from functools import lru_cache
from pathlib import Path

SETTINGS_PATH = Path("config/settings.yaml")


@lru_cache(maxsize=1)
def load_settings(path: str = str(SETTINGS_PATH)):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Settings file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_columns():
    return load_settings().get("columns", {})


def get_thresholds():
    return load_settings().get("thresholds", {})


def get_locations():
    return load_settings().get("locations", {})


def get_buoy_locations():
    return list(get_locations().get("buoys", []))


def get_study_area_locations():
    return list(get_locations().get("study_areas", []))


def get_all_locations():
    out = []
    for loc in get_buoy_locations() + get_study_area_locations():
        if loc not in out:
            out.append(loc)
    return out


def get_methods():
    methods = load_settings().get("methods", [])
    if methods:
        return list(methods)
    return ["linear", "qm", "rf"]


def get_path_template(name: str) -> str:
    paths = load_settings().get("paths", {})
    if name not in paths:
        raise KeyError(f"Missing path template '{name}' in settings")
    return str(paths[name])


def format_path(name: str, **kwargs) -> str:
    return get_path_template(name).format(**kwargs)

def get_evt_return_periods():
    return get_thresholds().get("evt_return_periods", [10, 20, 50])


def get_evt_bootstrap_samples():
    return int(get_thresholds().get("evt_bootstrap_samples", 2000))
