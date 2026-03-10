import math
import random

import numpy as np
import pandas as pd

from src.settings import get_method_settings
from src.bias_correction.methods.common import (
    TIME,
    HS_MODEL,
    HS_OBS,
    prepare_ml_dataframe,
    resolve_feature_columns,
    clip_nonnegative,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def _require_torch():
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install torch"
        )


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _resolve_device():
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

    matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = True

    return device


def _state_dict_to_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _fit_fill_and_scaler(df, feature_cols, fit_mask):
    fill = {}
    mean = {}
    std = {}

    for col in feature_cols:
        vals = pd.to_numeric(df.loc[fit_mask, col], errors="coerce").values.astype(float)
        med = float(np.nanmedian(vals))
        if not np.isfinite(med):
            med = 0.0
        fill[col] = med

        filled = np.where(np.isfinite(vals), vals, med)
        mu = float(np.mean(filled))
        sigma = float(np.std(filled))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        mean[col] = mu
        std[col] = sigma

    return fill, mean, std


def _transform_features(df, feature_cols, fill, mean, std):
    X = np.zeros((len(df), len(feature_cols)), dtype=np.float32)

    for j, col in enumerate(feature_cols):
        vals = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        vals = np.where(np.isfinite(vals), vals, fill[col])
        vals = (vals - mean[col]) / std[col]
        X[:, j] = vals.astype(np.float32)

    return X


def _build_segment_ids(df, time_col, source_col=None, expected_step_hours=1.0):
    """
    Build contiguous segment ids so sequences never cross:
      - missing/invalid timestamps
      - non-hourly gaps
      - source/buoy changes (for pooled data)
    """
    time = pd.to_datetime(df[time_col], errors="coerce")
    n = len(df)

    seg = np.zeros(n, dtype=int)
    current = 0

    for i in range(1, n):
        new_segment = False

        t_prev = time.iloc[i - 1]
        t_curr = time.iloc[i]

        if pd.isna(t_prev) or pd.isna(t_curr):
            new_segment = True
        else:
            dt_hours = (t_curr - t_prev).total_seconds() / 3600.0
            if not np.isfinite(dt_hours) or abs(dt_hours - expected_step_hours) > 1e-9:
                new_segment = True

        if source_col is not None and source_col in df.columns:
            if df[source_col].iloc[i] != df[source_col].iloc[i - 1]:
                new_segment = True

        if new_segment:
            current += 1

        seg[i] = current

    return seg


def _build_full_sequences(X, y, segment_ids, seq_len, valid_target_mask=None):
    """
    Build only full-length windows of size seq_len.
    Each window must stay entirely within one contiguous segment.
    The target is y[end] for the window ending at end.
    """
    n, n_features = X.shape

    X_seq = []
    y_seq = []
    target_idx = []

    if valid_target_mask is None:
        valid_target_mask = np.ones(n, dtype=bool)

    for end in range(seq_len - 1, n):
        start = end - seq_len + 1

        if np.any(segment_ids[start:end + 1] != segment_ids[end]):
            continue

        if not valid_target_mask[end]:
            continue

        X_seq.append(X[start:end + 1])
        y_seq.append(y[end])
        target_idx.append(end)

    if not X_seq:
        return (
            np.zeros((0, seq_len, n_features), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=int),
        )

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(target_idx, dtype=int),
    )


if nn is not None:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]


    class TemporalTransformerRegressor(nn.Module):
        def __init__(
            self,
            n_features,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
            seq_len,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x):
            z = self.input_proj(x).contiguous()
            z = self.pos_encoder(z).contiguous()
            # The window already contains only past/current timesteps, so an
            # explicit causal mask is unnecessary here and triggers unstable
            # CUDA encoder paths on the target V100 environment.
            z = self.encoder(z)
            z = self.norm(z[:, -1, :])
            return self.head(z).squeeze(-1)
else:
    class PositionalEncoding:
        def __init__(self, *args, **kwargs):
            _require_torch()


    class TemporalTransformerRegressor:
        def __init__(self, *args, **kwargs):
            _require_torch()


def _make_loader(x, y, batch_size, shuffle, use_cuda, seed):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        pin_memory=use_cuda,
        generator=generator,
    )


def _predict_batches(model, x, batch_size, device):
    preds = []
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:stop]).to(device, non_blocking=use_cuda).contiguous()
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())

    if not preds:
        return np.array([], dtype=np.float32)

    return np.concatenate(preds).astype(np.float32, copy=False)


def fit(df):
    _require_torch()

    cfg = get_method_settings("transformer")
    seed = int(cfg.get("random_state", 1))
    _set_seed(seed)

    work = df.copy()
    if TIME in work.columns:
        work[TIME] = pd.to_datetime(work[TIME], errors="coerce")
        sort_cols = [TIME]
        if "source" in work.columns:
            sort_cols = ["source", TIME]
        work = work.sort_values(sort_cols).reset_index(drop=True)

    work = prepare_ml_dataframe(work)
    features = resolve_feature_columns(work, cfg.get("features", []))

    seq_len = int(cfg.get("sequence_length", 24))
    y = (
        pd.to_numeric(work[HS_OBS], errors="coerce").values.astype(np.float32)
        - pd.to_numeric(work[HS_MODEL], errors="coerce").values.astype(np.float32)
    )

    valid_target = np.isfinite(y)
    valid_positions = np.flatnonzero(valid_target)

    min_train = int(cfg.get("min_train_samples", 80))
    min_val = int(cfg.get("min_val_samples", 20))
    val_fraction = float(cfg.get("validation_fraction", 0.2))

    if len(valid_positions) < min_train:
        raise ValueError("Too few valid samples for transformer.")

    n_val = max(min_val, int(round(len(valid_positions) * val_fraction)))
    n_val = min(n_val, max(0, len(valid_positions) - min_train))

    if n_val > 0:
        train_target_positions = valid_positions[:-n_val]
        val_target_positions = valid_positions[-n_val:]
    else:
        train_target_positions = valid_positions
        val_target_positions = np.array([], dtype=int)

    fit_mask = np.zeros(len(work), dtype=bool)
    fit_mask[train_target_positions] = True

    fill, mean, std = _fit_fill_and_scaler(work, features, fit_mask)
    X = _transform_features(work, features, fill, mean, std)

    source_col = "source" if "source" in work.columns else None
    segment_ids = _build_segment_ids(
        work,
        time_col=TIME,
        source_col=source_col,
        expected_step_hours=1.0,
    )

    train_target_mask = np.zeros(len(work), dtype=bool)
    train_target_mask[train_target_positions] = True
    train_target_mask &= valid_target

    val_target_mask = np.zeros(len(work), dtype=bool)
    val_target_mask[val_target_positions] = True
    val_target_mask &= valid_target

    X_train, y_train, _ = _build_full_sequences(
        X=X,
        y=y,
        segment_ids=segment_ids,
        seq_len=seq_len,
        valid_target_mask=train_target_mask,
    )

    if len(X_train) < min_train:
        raise ValueError("Too few valid full-window sequence samples for transformer.")

    X_val, y_val, _ = _build_full_sequences(
        X=X,
        y=y,
        segment_ids=segment_ids,
        seq_len=seq_len,
        valid_target_mask=val_target_mask,
    )

    device = _resolve_device()
    use_cuda = device.type == "cuda"
    print("transformer device:", device)

    model = TemporalTransformerRegressor(
        n_features=X_train.shape[2],
        d_model=int(cfg.get("d_model", 64)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dim_feedforward=int(cfg.get("dim_feedforward", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    loss_fn = nn.HuberLoss(delta=1.0)
    batch_size = int(cfg.get("batch_size", 64))

    train_loader = _make_loader(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
        use_cuda=use_cuda,
        seed=seed,
    )

    use_val = len(X_val) > 0
    if use_val:
        val_loader = _make_loader(
            x=X_val,
            y=y_val,
            batch_size=batch_size,
            shuffle=False,
            use_cuda=use_cuda,
            seed=seed,
        )

    best_state = None
    best_val = np.inf
    epochs_no_improve = 0

    max_epochs = int(cfg.get("max_epochs", 200))
    patience = int(cfg.get("patience", 20))

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda).contiguous()
            yb = yb.to(device, non_blocking=use_cuda)

            # Keep transformer training in float32 on CUDA; fp16 autocast was
            # triggering CUBLAS_STATUS_INVALID_VALUE on V100 for this path.
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if use_val:
            model.eval()
            losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=use_cuda).contiguous()
                    yb = yb.to(device, non_blocking=use_cuda)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    losses.append(float(loss.item()))

            val_loss = float(np.mean(losses)) if losses else np.inf

            if val_loss < best_val:
                best_val = val_loss
                best_state = _state_dict_to_cpu(model)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
        else:
            best_state = _state_dict_to_cpu(model)

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to("cpu").eval()

    return {
        "features": features,
        "fill": fill,
        "mean": mean,
        "std": std,
        "seq_len": seq_len,
        "config": {
            "d_model": int(cfg.get("d_model", 64)),
            "nhead": int(cfg.get("nhead", 4)),
            "num_layers": int(cfg.get("num_layers", 2)),
            "dim_feedforward": int(cfg.get("dim_feedforward", 128)),
            "dropout": float(cfg.get("dropout", 0.1)),
        },
        "device": device.type,
        "model": model,
    }


def apply(df, bundle):
    _require_torch()

    out = df.copy()
    if TIME in out.columns:
        out[TIME] = pd.to_datetime(out[TIME], errors="coerce")
        sort_cols = [TIME]
        if "source" in out.columns:
            sort_cols = ["source", TIME]
        out = out.sort_values(sort_cols).reset_index(drop=False).rename(
            columns={"index": "_orig_index"}
        )
    else:
        out["_orig_index"] = np.arange(len(out))

    prepared = prepare_ml_dataframe(out.copy())
    X = _transform_features(
        prepared,
        bundle["features"],
        bundle["fill"],
        bundle["mean"],
        bundle["std"],
    )
    hs = pd.to_numeric(prepared[HS_MODEL], errors="coerce").values.astype(np.float32)

    source_col = "source" if "source" in prepared.columns else None
    segment_ids = _build_segment_ids(
        prepared,
        time_col=TIME,
        source_col=source_col,
        expected_step_hours=1.0,
    )

    valid_target_mask = np.isfinite(hs)

    X_seq, _, end_idx = _build_full_sequences(
        X=X,
        y=np.zeros(len(prepared), dtype=np.float32),
        segment_ids=segment_ids,
        seq_len=bundle["seq_len"],
        valid_target_mask=valid_target_mask,
    )

    residual = np.full(len(prepared), np.nan, dtype=np.float32)

    if len(X_seq) > 0:
        device = _resolve_device()
        model = bundle["model"].to(device).eval()
        pred_vals = _predict_batches(
            model=model,
            x=X_seq,
            batch_size=256,
            device=device,
        )
        residual[end_idx] = pred_vals

    corrected = hs.copy()
    m = np.isfinite(residual)
    corrected[m] = hs[m] + residual[m]
    prepared[HS_MODEL] = clip_nonnegative(corrected)

    prepared = prepared.sort_values("_orig_index").drop(columns=["_orig_index"])
    prepared.index = range(len(prepared))
    return prepared
