import math
import random

import numpy as np
import pandas as pd

from src.settings import get_method_settings
from src.bias_correction.methods.common import (
    TIME, HS_MODEL, HS_OBS,
    prepare_ml_dataframe, resolve_feature_columns, clip_nonnegative,
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
    """Pick CUDA when available; enable cudnn benchmark + TF32."""
    _require_torch()
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        device = torch.device("cuda")
        torch.zeros(1, device=device)
    except Exception:
        return torch.device("cpu")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = True
    return device


def _cfg_int(cfg, key, default):
    return int(cfg.get(key, default))


def _cfg_float(cfg, key, default):
    return float(cfg.get(key, default))


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
        return (np.zeros((0, seq_len, n_feat), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=int))
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

def _loader(x, y, batch_size, shuffle, use_cuda, seed):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)),
                      shuffle=shuffle, pin_memory=use_cuda, generator=gen)


def _predict(model, x, batch_size, device):
    use_cuda = device.type == "cuda"
    parts = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = (torch.from_numpy(x[i:i + batch_size])
                  .to(device, non_blocking=use_cuda).contiguous())
            parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts).astype(np.float32) if parts else np.array([], dtype=np.float32)


# ── public API ───────────────────────────────────────────────────────────────

def fit(df, trial=None):
    _require_torch()
    cfg = get_method_settings("transformer")
    seed = _cfg_int(cfg, "random_state", 1)
    _set_seed(seed)

    work = df.copy()

    if TIME in work.columns:
        work[TIME] = pd.to_datetime(work[TIME], errors="coerce")
        sort_cols = ["source", TIME] if "source" in work.columns else [TIME]
        work = work.sort_values(sort_cols).reset_index(drop=True)

    work = prepare_ml_dataframe(work)

    features = resolve_feature_columns(work, cfg.get("features", []))
    seq_len = _cfg_int(cfg, "sequence_length", 24)

    y = (
        pd.to_numeric(work[HS_OBS], errors="coerce").to_numpy(np.float32)
        - pd.to_numeric(work[HS_MODEL], errors="coerce").to_numpy(np.float32)
    )

    valid = np.isfinite(y)
    pos = np.flatnonzero(valid)

    min_train = _cfg_int(cfg, "min_train_samples", 80)
    min_val = _cfg_int(cfg, "min_val_samples", 20)
    val_frac = _cfg_float(cfg, "validation_fraction", 0.2)

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
        return m & valid

    X_train, y_train, _ = _make_sequences(
        X, y, seg, seq_len, _target_mask(train_pos)
    )

    if len(X_train) < min_train:
        raise ValueError("Too few full-window sequences for transformer training.")

    X_val, y_val, _ = _make_sequences(
        X, y, seg, seq_len, _target_mask(val_pos)
    )

    device = _resolve_device()
    use_cuda = device.type == "cuda"

    model_cfg = dict(
        d_model=_cfg_int(cfg, "d_model", 64),
        nhead=_cfg_int(cfg, "nhead", 4),
        num_layers=_cfg_int(cfg, "num_layers", 2),
        dim_feedforward=_cfg_int(cfg, "dim_feedforward", 128),
        dropout=_cfg_float(cfg, "dropout", 0.1),
    )

    model = _TransformerModel(
        n_features=X_train.shape[2],
        seq_len=seq_len,
        **model_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_cfg_float(cfg, "learning_rate", 1e-3),
        weight_decay=_cfg_float(cfg, "weight_decay", 1e-4),
    )

    loss_fn = nn.HuberLoss(delta=1.0)
    batch_size = _cfg_int(cfg, "batch_size", 64)

    train_dl = _loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
        use_cuda=use_cuda,
        seed=seed,
    )

    val_dl = (
        _loader(
            X_val,
            y_val,
            batch_size,
            shuffle=False,
            use_cuda=use_cuda,
            seed=seed,
        )
        if len(X_val)
        else None
    )

    best_state = None
    best_val = np.inf
    no_improve = 0

    max_epochs = _cfg_int(cfg, "max_epochs", 200)
    patience = _cfg_int(cfg, "patience", 20)

    for epoch in range(max_epochs):

        model.train()

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=use_cuda).contiguous()
            yb = yb.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if val_dl is not None:

            model.eval()
            with torch.no_grad():
                vloss = np.mean(
                    [
                        loss_fn(
                            model(xb.to(device, non_blocking=use_cuda).contiguous()),
                            yb.to(device, non_blocking=use_cuda),
                        ).item()
                        for xb, yb in val_dl
                    ]
                )

            # ---------------------------
            # Optuna pruning (optional)
            # ---------------------------

            if trial is not None:
                trial.report(vloss, epoch)
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
    }


def apply(df, bundle):
    _require_torch()

    out = df.copy()
    if TIME in out.columns:
        out[TIME] = pd.to_datetime(out[TIME], errors="coerce")
        sort_cols = ["source", TIME] if "source" in out.columns else [TIME]
        out = out.sort_values(sort_cols).reset_index(drop=False).rename(
            columns={"index": "_orig_index"})
    else:
        out["_orig_index"] = np.arange(len(out))

    prep = prepare_ml_dataframe(out.copy())
    X = _scale_features(prep, bundle["features"], bundle["fill"],
                        bundle["mean"], bundle["std"])
    hs = pd.to_numeric(prep[HS_MODEL], errors="coerce").to_numpy(np.float32)

    src_col = "source" if "source" in prep.columns else None
    seg = _segment_ids(prep, TIME, source_col=src_col)

    X_seq, _, end_idx = _make_sequences(
        X, np.zeros(len(prep), dtype=np.float32),
        seg, bundle["seq_len"], np.isfinite(hs),
    )

    residual = np.full(len(prep), np.nan, dtype=np.float32)
    if len(X_seq) > 0:
        device = _resolve_device()
        model = bundle["model"].to(device).eval()
        residual[end_idx] = _predict(model, X_seq, batch_size=256, device=device)

    corrected = hs.copy()
    m = np.isfinite(residual)
    corrected[m] = hs[m] + residual[m]
    prep[HS_MODEL] = clip_nonnegative(corrected)

    return prep.sort_values("_orig_index").drop(columns=["_orig_index"]).reset_index(drop=True)
