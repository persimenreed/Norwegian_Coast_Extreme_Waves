import numpy as np
import pandas as pd

from src.settings import get_columns, get_dagqm_settings
from src.bias_correction.methods.common import HS_MODEL, HS_OBS, clip_nonnegative

_COLUMNS = get_columns()
DIR_MODEL = _COLUMNS.get("dir_model", "Pdir")


def _clean_positive(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return np.sort(x)


def _gumbel_quantile_grid(n=401):
    p = np.linspace(0.001, 0.999, n)
    g = -np.log(-np.log(p))
    g = (g - g.min()) / (g.max() - g.min())
    p_tail = 0.001 + g * (0.999 - 0.001)
    return np.unique(np.clip(p_tail, 0.001, 0.999))


def _build_mapping(source, target):
    source = _clean_positive(source)
    target = _clean_positive(target)

    if len(source) < 40 or len(target) < 40:
        raise ValueError("Too few positive samples for DAGQM sector mapping.")

    p = _gumbel_quantile_grid()
    qs = np.quantile(source, p)
    qt = np.quantile(target, p)

    return {"p": p, "qs": qs, "qt": qt}


def _interp_extrap(x, xp, fp):
    x = np.asarray(x, float)
    y = np.interp(x, xp, fp)

    left_mask = x < xp[0]
    right_mask = x > xp[-1]

    if np.any(left_mask):
        dx = xp[1] - xp[0]
        slope = 0.0 if dx == 0 else (fp[1] - fp[0]) / dx
        y[left_mask] = fp[0] + slope * (x[left_mask] - xp[0])

    if np.any(right_mask):
        dx = xp[-1] - xp[-2]
        slope = 0.0 if dx == 0 else (fp[-1] - fp[-2]) / dx
        y[right_mask] = fp[-1] + slope * (x[right_mask] - xp[-1])

    return y


def _sector_ids(direction_deg, n_sectors):
    ang = np.mod(np.asarray(direction_deg, float), 360.0)
    width = 360.0 / n_sectors
    sid = np.floor(ang / width).astype(int)
    sid = np.clip(sid, 0, n_sectors - 1)
    return sid


def fit(df):
    if DIR_MODEL not in df.columns:
        raise ValueError(f"DAGQM requires direction column '{DIR_MODEL}'.")

    cfg = get_dagqm_settings()
    n_sectors = int(cfg.get("n_direction_sectors", 8))
    min_samples = int(cfg.get("min_samples_per_sector", 80))
    blend_global_weight = float(cfg.get("blend_global_weight", 0.35))

    global_map = _build_mapping(df[HS_MODEL].values, df[HS_OBS].values)

    dirs = pd.to_numeric(df[DIR_MODEL], errors="coerce").values
    hs_m = pd.to_numeric(df[HS_MODEL], errors="coerce").values
    hs_o = pd.to_numeric(df[HS_OBS], errors="coerce").values

    valid = np.isfinite(dirs) & np.isfinite(hs_m) & np.isfinite(hs_o) & (hs_m > 0) & (hs_o > 0)
    sector_ids = _sector_ids(dirs[valid], n_sectors)

    sector_maps = {}
    for s in range(n_sectors):
        m = sector_ids == s
        if np.sum(m) < min_samples:
            continue
        sector_maps[s] = _build_mapping(hs_m[valid][m], hs_o[valid][m])

    return {
        "global_map": global_map,
        "sector_maps": sector_maps,
        "n_sectors": n_sectors,
        "blend_global_weight": blend_global_weight,
    }


def apply(df, model):
    out = df.copy()
    x = pd.to_numeric(out[HS_MODEL], errors="coerce").values
    y = np.full(len(out), np.nan, dtype=float)

    global_pred = np.full(len(out), np.nan, dtype=float)
    m_x = np.isfinite(x) & (x > 0)
    if np.any(m_x):
        global_pred[m_x] = _interp_extrap(
            x[m_x],
            model["global_map"]["qs"],
            model["global_map"]["qt"],
        )

    if DIR_MODEL not in out.columns:
        out[HS_MODEL] = clip_nonnegative(global_pred)
        return out

    dirs = pd.to_numeric(out[DIR_MODEL], errors="coerce").values
    valid_dir = np.isfinite(dirs)
    sector_ids = np.full(len(out), -1, dtype=int)
    sector_ids[valid_dir] = _sector_ids(dirs[valid_dir], model["n_sectors"])

    alpha = model["blend_global_weight"]

    for i in range(len(out)):
        if not (np.isfinite(x[i]) and x[i] > 0):
            continue

        gp = global_pred[i]
        sid = sector_ids[i]

        if sid in model["sector_maps"]:
            sec_map = model["sector_maps"][sid]
            sp = _interp_extrap(
                np.array([x[i]], dtype=float),
                sec_map["qs"],
                sec_map["qt"],
            )[0]
            y[i] = alpha * gp + (1.0 - alpha) * sp
        else:
            y[i] = gp

    out[HS_MODEL] = clip_nonnegative(y)
    return out
