import math
import random

import numpy as np
import pandas as pd

from src.settings import get_method_settings
from src.bias_correction.methods.common import (
    TIME,
    HS_MODEL,
    HS_OBS,
    HS_QUANTILE_BASELINE,
    cfg_int,
    cfg_float,
    prepare_ml_dataframe,
    resolve_feature_columns,
    clip_nonnegative,
    quantile_feature_columns,
    augment_quantile_features,
    build_target_transform,
    invert_target,
    build_tail_sample_weights,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = nn = DataLoader = TensorDataset = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is not installed. Install it with: pip install torch")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device():
    _require_torch()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU requested but not available")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    return device


# ── feature scaling ──────────────────────────────────────────────────────────

def _fit_scaler(df, feature_cols, mask):
    """Per-feature median fill, mean, and std — fitted on rows where *mask* is True."""
    fill, mean, std = {}, {}, {}
    for col in feature_cols:
        vals = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(float)
        med = float(np.nanmedian(vals))
        fill[col] = med if np.isfinite(med) else 0.0

        filled = np.where(np.isfinite(vals), vals, fill[col])
        mean[col] = float(np.mean(filled))
        s = float(np.std(filled))
        std[col] = s if (np.isfinite(s) and s > 0) else 1.0
    return fill, mean, std


def _scale_features(df, feature_cols, fill, mean, std):
    """Fill missing values and z-score normalise → (n_rows, n_features) float32."""
    X = np.empty((len(df), len(feature_cols)), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        v = np.where(np.isfinite(v), v, fill[col])
        X[:, j] = ((v - mean[col]) / std[col]).astype(np.float32)
    return X


# ── sequence construction ────────────────────────────────────────────────────

def _segment_ids(df, time_col, source_col=None, step_hours=1.0):
    """Assign segment ids; new segment at time gaps or source changes."""
    time = pd.to_datetime(df[time_col], errors="coerce")
    n = len(df)
    seg = np.zeros(n, dtype=int)
    cur = 0
    for i in range(1, n):
        t0, t1 = time.iloc[i - 1], time.iloc[i]
        new = pd.isna(t0) or pd.isna(t1)
        if not new:
            dt = (t1 - t0).total_seconds() / 3600.0
            new = not np.isfinite(dt) or abs(dt - step_hours) > 1e-9
        if source_col and source_col in df.columns:
            new = new or (df[source_col].iloc[i] != df[source_col].iloc[i - 1])
        if new:
            cur += 1
        seg[i] = cur
    return seg


def _make_sequences(X, y, seg, seq_len, target_mask=None):
    """Sliding windows of *seq_len* that stay within contiguous segments."""
    n, n_feat = X.shape
    if target_mask is None:
        target_mask = np.ones(n, dtype=bool)

    xs, ys, idx = [], [], []
    for end in range(seq_len - 1, n):
        start = end - seq_len + 1
        if seg[start] != seg[end]:
            continue
        if not target_mask[end]:
            continue
        xs.append(X[start:end + 1])
        ys.append(y[end])
        idx.append(end)

    if not xs:
        return (
            np.zeros((0, seq_len, n_feat), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=int),
        )
    return np.array(xs, np.float32), np.array(ys, np.float32), np.array(idx, int)

# ── model ────────────────────────────────────────────────────────────────────

if nn is not None:
    class _PosEncoding(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10_000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class _TransformerModel(nn.Module):
        def __init__(self, n_features, d_model, nhead, num_layers,
                     dim_feedforward, dropout, seq_len):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            self.pos = _PosEncoding(d_model, seq_len)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, 1),
            )

        def forward(self, x):
            z = self.pos(self.proj(x).contiguous()).contiguous()
            # No explicit causal mask — windows contain only past/current steps,
            # and an explicit mask triggers unstable CUDA paths on V100.
            z = self.encoder(z)
            return self.head(self.norm(z[:, -1])).squeeze(-1)
else:
    class _TransformerModel:
        def __init__(self, *a, **kw):
            _require_torch()


# ── data-loaders / prediction ────────────────────────────────────────────────

def _loader(
    x,
    y,
    batch_size,
    shuffle,
    use_cuda,
    seed,
    weights=None,
    obs_values=None,
    base_values=None,
):
    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)
    if obs_values is None:
        obs_values = np.zeros(len(y), dtype=np.float32)
    if base_values is None:
        base_values = np.zeros(len(y), dtype=np.float32)

    ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(np.asarray(weights, dtype=np.float32)),
        torch.from_numpy(np.asarray(obs_values, dtype=np.float32)),
        torch.from_numpy(np.asarray(base_values, dtype=np.float32)),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=shuffle,
        pin_memory=use_cuda,
        generator=gen,
        num_workers=6
    )


def _predict(model, x, batch_size, device):
    use_cuda = device.type == "cuda"
    parts = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = (
                torch.from_numpy(x[i:i + batch_size])
                .to(device, non_blocking=use_cuda)
                .contiguous()
            )
            parts.append(model(xb).cpu().numpy())
    return (
        np.concatenate(parts).astype(np.float32)
        if parts else np.array([], dtype=np.float32)
    )

def _weighted_mean(values, weights):
    weighted = values * weights
    denom = torch.clamp(weights.sum(), min=1e-8)
    return weighted.sum() / denom


def _invert_target_torch(pred_target, base_values, transform_cfg):
    mode = str(transform_cfg.get("mode", "log_ratio")).strip().lower()
    eps = float(transform_cfg.get("eps", 1e-4))

    if mode in {"additive_residual", "quantile_residual"}:
        corrected = base_values + pred_target
        return torch.clamp(corrected, min=0.0)

    if mode == "log_ratio":
        safe_base = torch.clamp(base_values + eps, min=eps)
        corrected = torch.exp(torch.log(safe_base) + pred_target) - eps
        return torch.clamp(corrected, min=0.0)

    raise ValueError(f"Unsupported inverse target transform: {mode}")


def _mixed_loss(pred, target, sample_weights, obs_values, base_values, transform_cfg, cfg):
    """
    Tail-aware dual-space loss:
      1. keep the transformed-target fit stable
      2. directly penalize errors in corrected Hs, where EVT sensitivity lives
    """
    target_loss = _weighted_mean((pred - target) ** 2, sample_weights)

    pred_hs = _invert_target_torch(pred, base_values, transform_cfg)
    true_hs = torch.clamp(obs_values, min=0.0)
    physical_loss = _weighted_mean((pred_hs - true_hs) ** 2, sample_weights)

    target_w = cfg_float(cfg, "target_space_loss_weight", 0.5)
    physical_w = cfg_float(cfg, "physical_space_loss_weight", 1.0)
    return target_w * target_loss + physical_w * physical_loss


# ── public API ───────────────────────────────────────────────────────────────

def fit(df, trial=None, trial_step_offset=0, settings_name=None):
    _require_torch()
    if not settings_name:
        raise ValueError("settings_name must be provided for transformer training.")

    cfg = get_method_settings(settings_name)
    if not cfg:
        raise ValueError(f"Missing transformer settings block '{settings_name}'.")
    seed = cfg_int(cfg, "random_state", 1)
    _set_seed(seed)

    work = df.copy()

    if TIME in work.columns:
        work[TIME] = pd.to_datetime(work[TIME], errors="coerce")
        sort_cols = ["source", TIME] if "source" in work.columns else [TIME]
        work = work.sort_values(sort_cols).reset_index(drop=True)

    work = prepare_ml_dataframe(work)

    obs_values = pd.to_numeric(work[HS_OBS], errors="coerce").to_numpy(np.float32)
    raw_values = pd.to_numeric(work[HS_MODEL], errors="coerce").to_numpy(np.float32)

    y, valid_target, transform_cfg = build_target_transform(obs_values, raw_values, cfg)

    if transform_cfg["mode"] == "quantile_residual":
        work, quantile_extras = augment_quantile_features(
            work,
            raw_values,
            transform_cfg,
            reference_values=raw_values,
        )
    else:
        quantile_extras = None

    features = resolve_feature_columns(work, cfg.get("features", []))
    if transform_cfg["mode"] == "quantile_residual":
        features = quantile_feature_columns(features)

    seq_len = cfg_int(cfg, "sequence_length", 24)
    pos = np.flatnonzero(valid_target)

    min_train = cfg_int(cfg, "min_train_samples", 80)
    min_val = cfg_int(cfg, "min_val_samples", 20)
    val_frac = cfg_float(cfg, "validation_fraction", 0.2)

    if len(pos) < min_train:
        raise ValueError("Too few valid samples for transformer.")

    n_val = min(
        max(min_val, int(round(len(pos) * val_frac))),
        max(0, len(pos) - min_train),
    )

    train_pos = pos[:-n_val] if n_val > 0 else pos
    val_pos = pos[-n_val:] if n_val > 0 else np.array([], dtype=int)

    fit_mask = np.zeros(len(work), dtype=bool)
    fit_mask[train_pos] = True

    fill, mean, std = _fit_scaler(work, features, fit_mask)
    X = _scale_features(work, features, fill, mean, std)

    src_col = "source" if "source" in work.columns else None
    seg = _segment_ids(work, TIME, source_col=src_col)

    def _target_mask(positions):
        m = np.zeros(len(work), dtype=bool)
        m[positions] = True
        return m & valid_target

    X_train, y_train, train_end_idx = _make_sequences(
        X, y, seg, seq_len, _target_mask(train_pos)
    )

    if len(X_train) < min_train:
        raise ValueError("Too few full-window sequences for transformer training.")

    X_val, y_val, val_end_idx = _make_sequences(
        X, y, seg, seq_len, _target_mask(val_pos)
    )

    if transform_cfg["mode"] == "quantile_residual":
        base_values = quantile_extras[HS_QUANTILE_BASELINE]
    else:
        base_values = raw_values

    train_obs = obs_values[train_end_idx]
    train_base = base_values[train_end_idx]
    val_obs = obs_values[val_end_idx] if len(val_end_idx) else np.zeros(0, dtype=np.float32)
    val_base = base_values[val_end_idx] if len(val_end_idx) else np.zeros(0, dtype=np.float32)

    train_weights = build_tail_sample_weights(train_obs, cfg)
    val_weights = build_tail_sample_weights(val_obs, cfg) if len(val_obs) else np.zeros(0, dtype=np.float32)

    device = _resolve_device()
    use_cuda = device.type == "cuda"
    print(f"[Transformer] Training device: {device}")
    print(f"[Transformer] Target transform: {transform_cfg['mode']} (eps={transform_cfg['eps']})")

    model_cfg = dict(
        d_model=cfg_int(cfg, "d_model", 64),
        nhead=cfg_int(cfg, "nhead", 4),
        num_layers=cfg_int(cfg, "num_layers", 2),
        dim_feedforward=cfg_int(cfg, "dim_feedforward", 128),
        dropout=cfg_float(cfg, "dropout", 0.1),
    )

    model = _TransformerModel(
        n_features=X_train.shape[2],
        seq_len=seq_len,
        **model_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_float(cfg, "learning_rate", 1e-3),
        weight_decay=cfg_float(cfg, "weight_decay", 1e-4),
    )

    batch_size = cfg_int(cfg, "batch_size", 64)

    train_dl = _loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
        use_cuda=use_cuda,
        seed=seed,
        weights=train_weights,
        obs_values=train_obs,
        base_values=train_base,
    )

    val_dl = (
        _loader(
            X_val,
            y_val,
            batch_size,
            shuffle=False,
            use_cuda=use_cuda,
            seed=seed,
            weights=val_weights,
            obs_values=val_obs,
            base_values=val_base,
        )
        if len(X_val)
        else None
    )

    best_state = None
    best_val = np.inf
    no_improve = 0

    max_epochs = cfg_int(cfg, "max_epochs", 200)
    patience = cfg_int(cfg, "patience", 20)

    for epoch in range(max_epochs):

        model.train()

        for xb, yb, wb, ob, bb in train_dl:
            xb = xb.to(device, non_blocking=use_cuda).contiguous()
            yb = yb.to(device, non_blocking=use_cuda)
            wb = wb.to(device, non_blocking=use_cuda)
            ob = ob.to(device, non_blocking=use_cuda)
            bb = bb.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = _mixed_loss(pred, yb, wb, ob, bb, transform_cfg, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if val_dl is not None:

            model.eval()
            with torch.no_grad():
                val_losses = []
                for xb, yb, wb, ob, bb in val_dl:
                    xb = xb.to(device, non_blocking=use_cuda).contiguous()
                    yb = yb.to(device, non_blocking=use_cuda)
                    wb = wb.to(device, non_blocking=use_cuda)
                    ob = ob.to(device, non_blocking=use_cuda)
                    bb = bb.to(device, non_blocking=use_cuda)

                    pred = model(xb)
                    vloss = _mixed_loss(pred, yb, wb, ob, bb, transform_cfg, cfg)
                    val_losses.append(vloss.item())

                vloss = float(np.mean(val_losses)) if val_losses else np.inf

            # ---------------------------
            # Optuna pruning (optional)
            # ---------------------------

            if trial is not None:
                trial.report(vloss, int(trial_step_offset) + epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.exceptions.TrialPruned()

            if vloss < best_val:
                best_val = vloss
                no_improve = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        else:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to("cpu").eval()

    return {
        "features": features,
        "fill": fill,
        "mean": mean,
        "std": std,
        "seq_len": seq_len,
        "config": model_cfg,
        "device": device.type,
        "model": model,
        "target_transform": transform_cfg,
    }


def apply(df, bundle):
    _require_torch()

    out = df.copy()
    if TIME in out.columns:
        out[TIME] = pd.to_datetime(out[TIME], errors="coerce")
        sort_cols = ["source", TIME] if "source" in out.columns else [TIME]
        out = out.sort_values(sort_cols).reset_index(drop=False).rename(
            columns={"index": "_orig_index"}
        )
    else:
        out["_orig_index"] = np.arange(len(out))

    prep = prepare_ml_dataframe(out.copy())
    hs = pd.to_numeric(prep[HS_MODEL], errors="coerce").to_numpy(np.float32)
    transform_cfg = bundle["target_transform"]

    if transform_cfg["mode"] == "quantile_residual":
        prep, extras = augment_quantile_features(
            prep,
            hs,
            transform_cfg,
            reference_values=hs,
        )
        base_values = extras[HS_QUANTILE_BASELINE]
    else:
        base_values = hs

    X = _scale_features(
        prep,
        bundle["features"],
        bundle["fill"],
        bundle["mean"],
        bundle["std"],
    )

    src_col = "source" if "source" in prep.columns else None
    seg = _segment_ids(prep, TIME, source_col=src_col)

    X_seq, _, end_idx = _make_sequences(
        X,
        np.zeros(len(prep), dtype=np.float32),
        seg,
        bundle["seq_len"],
        np.isfinite(hs),
    )

    pred_target = np.full(len(prep), np.nan, dtype=np.float32)
    if len(X_seq) > 0:
        device = _resolve_device()
        model = bundle["model"].to(device).eval()
        pred_target[end_idx] = _predict(model, X_seq, batch_size=256, device=device)

    corrected = hs.copy()
    restored = invert_target(pred_target, base_values, transform_cfg)
    m = np.isfinite(restored)
    corrected[m] = restored[m]
    prep[HS_MODEL] = clip_nonnegative(corrected)

    return (
        prep.sort_values("_orig_index")
        .drop(columns=["_orig_index"])
        .reset_index(drop=True)
    )
