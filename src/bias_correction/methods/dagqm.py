import numpy as np
import pandas as pd

from src.bias_correction.methods.common import (
    DIR_MODEL,
    HS_MODEL,
    HS_OBS,
    clip_nonnegative,
    gumbel_quantile_grid,
)

N_DIRECTION_SECTORS = 8
MIN_SAMPLES_PER_SECTOR = 80
BLEND_GLOBAL_WEIGHT = 0.50


def _clean_positive(values):
    values = np.asarray(values, float)
    return np.sort(values[np.isfinite(values) & (values > 0)])


def _build_mapping(source, target):
    source = _clean_positive(source)
    target = _clean_positive(target)

    if len(source) < 40 or len(target) < 40:
        raise ValueError("Too few positive samples for DAGQM sector mapping.")

    probabilities = gumbel_quantile_grid()
    source_quantiles = np.quantile(source, probabilities)
    target_quantiles = np.quantile(target, probabilities)
    return {
        "source_quantiles": source_quantiles,
        "target_quantiles": target_quantiles,
        "left_slope": (
            0.0
            if source_quantiles[1] == source_quantiles[0]
            else (target_quantiles[1] - target_quantiles[0]) / (source_quantiles[1] - source_quantiles[0])
        ),
        "right_tail_bias": float(target_quantiles[-1] - source_quantiles[-1]),
    }


def _interp_extrap(values, mapping):
    values = np.asarray(values, float)
    xp = mapping["source_quantiles"]
    fp = mapping["target_quantiles"]
    out = np.interp(values, xp, fp)

    left_mask = values < xp[0]
    if np.any(left_mask):
        out[left_mask] = fp[0] + mapping["left_slope"] * (values[left_mask] - xp[0])

    right_mask = values > xp[-1]
    if np.any(right_mask):
        out[right_mask] = values[right_mask] + mapping["right_tail_bias"]

    return out


def _sector_ids(direction_deg, n_sectors):
    width = 360.0 / n_sectors
    return np.clip(np.floor(np.mod(np.asarray(direction_deg, float), 360.0) / width).astype(int), 0, n_sectors - 1)


def fit(df, settings_name=None):
    if DIR_MODEL not in df.columns:
        raise ValueError(f"DAGQM requires direction column '{DIR_MODEL}'.")

    dirs = pd.to_numeric(df[DIR_MODEL], errors="coerce").to_numpy(float)
    hs_model = pd.to_numeric(df[HS_MODEL], errors="coerce").to_numpy(float)
    hs_obs = pd.to_numeric(df[HS_OBS], errors="coerce").to_numpy(float)

    valid = np.isfinite(dirs) & np.isfinite(hs_model) & np.isfinite(hs_obs) & (hs_model > 0) & (hs_obs > 0)
    sector_ids = _sector_ids(dirs[valid], N_DIRECTION_SECTORS)

    sector_maps = {}
    for sector in range(N_DIRECTION_SECTORS):
        mask = sector_ids == sector
        if np.sum(mask) >= MIN_SAMPLES_PER_SECTOR:
            sector_maps[sector] = _build_mapping(hs_model[valid][mask], hs_obs[valid][mask])

    global_map = _build_mapping(hs_model[valid], hs_obs[valid])
    return {
        "global_map": global_map,
        "sector_maps": sector_maps,
        "n_sectors": N_DIRECTION_SECTORS,
        "blend_global_weight": BLEND_GLOBAL_WEIGHT,
    }


def apply(df, model):
    out = df.copy()
    hs = pd.to_numeric(out[HS_MODEL], errors="coerce").to_numpy(float)
    corrected = np.full(len(out), np.nan, dtype=float)

    valid_hs = np.isfinite(hs) & (hs > 0)
    if np.any(valid_hs):
        corrected[valid_hs] = _interp_extrap(hs[valid_hs], model["global_map"])

    if DIR_MODEL in out.columns:
        dirs = pd.to_numeric(out[DIR_MODEL], errors="coerce").to_numpy(float)
        sector_ids = np.full(len(out), -1, dtype=int)
        valid_dir = valid_hs & np.isfinite(dirs)
        sector_ids[valid_dir] = _sector_ids(dirs[valid_dir], model["n_sectors"])

        alpha = model["blend_global_weight"]
        for sector, mapping in model["sector_maps"].items():
            mask = sector_ids == sector
            if not np.any(mask):
                continue
            sector_pred = _interp_extrap(hs[mask], mapping)
            corrected[mask] = alpha * corrected[mask] + (1.0 - alpha) * sector_pred

    out[HS_MODEL] = clip_nonnegative(corrected)
    return out
