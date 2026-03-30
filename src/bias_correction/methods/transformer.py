import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd

from src.model_profiles import resolve_profile
from src.bias_correction.methods.common import (
    HS_MODEL,
    HS_OBS,
    HS_QUANTILE,
    HS_QUANTILE_BASELINE,
    TIME,
    augment_quantile_features,
    build_tail_sample_weights,
    build_target_transform,
    cfg_float,
    cfg_int,
    clip_nonnegative,
    feature_matrix,
    fit_standard_scaler,
    invert_target,
    numeric_values,
    prepare_ml_dataframe,
    protect_tail_residuals,
    quantile_feature_columns,
    resolve_feature_columns,
    restore_frame_order,
    sort_frame,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = nn = DataLoader = TensorDataset = None

DEFAULT_TRANSFORMER_CONFIG = {
    "features": [],
    "sequence_length": 16,
    "d_model": 16,
    "nhead": 2,
    "num_layers": 2,
    "dim_feedforward": 64,
    "dropout": 0.15,
    "learning_rate": 8e-4,
    "weight_decay": 5e-6,
    "batch_size": 32,
    "min_train_samples": 80,
    "min_val_samples": 20,
    "validation_fraction": 0.2,
    "max_epochs": 200,
    "patience": 20,
    "quantile_bias_mode": "additive",
    "target_eps": 1e-5,
    "tail_weight_q90": 3.0,
    "tail_weight_q95": 6.0,
    "tail_weight_q99": 10.0,
    "tail_pool_enabled": True,
    "tail_pool_start": 0.95,
    "tail_pool_end": 0.995,
    "tail_blend_start": 0.95,
    "tail_residual_protection_enabled": True,
    "tail_residual_protection_mode": "sign_aware",
    "tail_residual_start": 0.95,
    "tail_residual_end": 0.999,
    "tail_residual_min_scale": 0.25,
    "target_space_loss_weight": 0.25,
    "physical_space_loss_weight": 1.0,
    "random_state": 1,
}

def _resolve_config(settings_name=None):
    return resolve_profile(deepcopy(DEFAULT_TRANSFORMER_CONFIG), "transformer", settings_name)


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

    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def _segment_ids(df, time_col, source_col=None, step_hours=1.0):
    time = pd.to_datetime(df[time_col], errors="coerce")
    segment_ids = np.zeros(len(df), dtype=int)
    current = 0

    for index in range(1, len(df)):
        prev_time = time.iloc[index - 1]
        current_time = time.iloc[index]
        new_segment = pd.isna(prev_time) or pd.isna(current_time)

        if not new_segment:
            delta_hours = (current_time - prev_time).total_seconds() / 3600.0
            new_segment = not np.isfinite(delta_hours) or abs(delta_hours - step_hours) > 1e-9

        if source_col and source_col in df.columns:
            new_segment = new_segment or (df[source_col].iloc[index] != df[source_col].iloc[index - 1])

        if new_segment:
            current += 1
        segment_ids[index] = current

    return segment_ids


def _make_sequences(features, target, segments, sequence_length, target_mask=None):
    if target_mask is None:
        target_mask = np.ones(len(target), dtype=bool)

    xs = []
    ys = []
    end_idx = []
    for end in range(sequence_length - 1, len(features)):
        start = end - sequence_length + 1
        if segments[start] != segments[end] or not target_mask[end]:
            continue
        xs.append(features[start : end + 1])
        ys.append(target[end])
        end_idx.append(end)

    if not xs:
        n_features = features.shape[1]
        return (
            np.zeros((0, sequence_length, n_features), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=int),
        )

    return np.array(xs, np.float32), np.array(ys, np.float32), np.array(end_idx, int)


if nn is not None:
    class _PosEncoding(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10_000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]


    class _TransformerModel(nn.Module):
        def __init__(self, n_features, d_model, nhead, num_layers, dim_feedforward, dropout, sequence_length):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            self.pos = _PosEncoding(d_model, sequence_length)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
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
            return self.head(self.norm(self.encoder(self.pos(self.proj(x).contiguous()))[:, -1])).squeeze(-1)
else:
    class _TransformerModel:
        def __init__(self, *args, **kwargs):
            _require_torch()


def _loader(x, y, batch_size, shuffle, use_cuda, seed, weights=None, obs_values=None, base_values=None):
    weights = np.ones(len(y), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    obs_values = np.zeros(len(y), dtype=np.float32) if obs_values is None else np.asarray(obs_values, dtype=np.float32)
    base_values = np.zeros(len(y), dtype=np.float32) if base_values is None else np.asarray(base_values, dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(weights),
        torch.from_numpy(obs_values),
        torch.from_numpy(base_values),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        pin_memory=use_cuda,
        generator=generator,
        num_workers=6,
    )


def _predict(model, x, batch_size, device):
    parts = []
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device, non_blocking=use_cuda).contiguous()
            parts.append(model(batch).cpu().numpy())
    return np.concatenate(parts).astype(np.float32) if parts else np.array([], dtype=np.float32)


def _weighted_mean(values, weights):
    return (values * weights).sum() / torch.clamp(weights.sum(), min=1e-8)


def _invert_target_torch(pred_target, base_values, transform_cfg):
    return torch.clamp(base_values + pred_target, min=0.0)


def _mixed_loss(pred, target, sample_weights, obs_values, base_values, transform_cfg, cfg):
    target_loss = _weighted_mean((pred - target) ** 2, sample_weights)
    pred_hs = _invert_target_torch(pred, base_values, transform_cfg)
    true_hs = torch.clamp(obs_values, min=0.0)
    physical_loss = _weighted_mean((pred_hs - true_hs) ** 2, sample_weights)
    return cfg_float(cfg, "target_space_loss_weight", 0.5) * target_loss + cfg_float(
        cfg,
        "physical_space_loss_weight",
        1.0,
    ) * physical_loss


def fit(df, trial=None, trial_step_offset=0, settings_name=None):
    _require_torch()

    cfg = _resolve_config(settings_name)
    seed = cfg_int(cfg, "random_state", 1)
    _set_seed(seed)

    work = prepare_ml_dataframe(sort_frame(df))
    obs_values = numeric_values(work, HS_OBS, dtype=np.float32)
    raw_values = numeric_values(work, HS_MODEL, dtype=np.float32)
    target, valid_target, transform_cfg = build_target_transform(obs_values, raw_values, cfg)
    work, quantile_extras = augment_quantile_features(
        work,
        raw_values,
        transform_cfg,
        reference_values=raw_values,
    )

    features = resolve_feature_columns(work, cfg.get("features"))
    features = quantile_feature_columns(features)

    sequence_length = cfg_int(cfg, "sequence_length", 24)
    valid_positions = np.flatnonzero(valid_target)
    min_train = cfg_int(cfg, "min_train_samples", 80)
    min_val = cfg_int(cfg, "min_val_samples", 20)
    val_fraction = cfg_float(cfg, "validation_fraction", 0.2)

    if len(valid_positions) < min_train:
        raise ValueError("Too few valid samples for transformer.")

    n_val = min(
        max(min_val, int(round(len(valid_positions) * val_fraction))),
        max(0, len(valid_positions) - min_train),
    )
    train_positions = valid_positions[:-n_val] if n_val > 0 else valid_positions
    val_positions = valid_positions[-n_val:] if n_val > 0 else np.array([], dtype=int)

    fit_mask = np.zeros(len(work), dtype=bool)
    fit_mask[train_positions] = True

    fill, mean, std = fit_standard_scaler(work, features, mask=fit_mask)
    X, _ = feature_matrix(work, features, fill=fill, mean=mean, std=std)

    source_column = "source" if "source" in work.columns else None
    segments = _segment_ids(work, TIME, source_col=source_column)

    def _target_mask(positions):
        mask = np.zeros(len(work), dtype=bool)
        mask[positions] = True
        return mask & valid_target

    X_train, y_train, train_end_idx = _make_sequences(
        X,
        target,
        segments,
        sequence_length,
        _target_mask(train_positions),
    )
    if len(X_train) < min_train:
        raise ValueError("Too few full-window sequences for transformer training.")

    X_val, y_val, val_end_idx = _make_sequences(
        X,
        target,
        segments,
        sequence_length,
        _target_mask(val_positions),
    )

    base_values = quantile_extras[HS_QUANTILE_BASELINE] if quantile_extras else raw_values
    train_obs = obs_values[train_end_idx]
    train_base = base_values[train_end_idx]
    val_obs = obs_values[val_end_idx] if len(val_end_idx) else np.zeros(0, dtype=np.float32)
    val_base = base_values[val_end_idx] if len(val_end_idx) else np.zeros(0, dtype=np.float32)
    train_weights = build_tail_sample_weights(train_obs, cfg)
    val_weights = build_tail_sample_weights(val_obs, cfg) if len(val_obs) else np.zeros(0, dtype=np.float32)

    device = _resolve_device()
    use_cuda = device.type == "cuda"
    print(f"[Transformer] Training device: {device}")
    print(
        f"[Transformer] Target transform: {transform_cfg['mode']} "
        f"(quantile_bias_mode={transform_cfg['quantile_bias_mode']}, eps={transform_cfg['eps']})"
    )

    model = _TransformerModel(
        n_features=X_train.shape[2],
        sequence_length=sequence_length,
        d_model=cfg_int(cfg, "d_model", 64),
        nhead=cfg_int(cfg, "nhead", 4),
        num_layers=cfg_int(cfg, "num_layers", 2),
        dim_feedforward=cfg_int(cfg, "dim_feedforward", 128),
        dropout=cfg_float(cfg, "dropout", 0.1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_float(cfg, "learning_rate", 1e-3),
        weight_decay=cfg_float(cfg, "weight_decay", 1e-4),
    )
    batch_size = cfg_int(cfg, "batch_size", 64)

    train_loader = _loader(
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
    val_loader = (
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

    for epoch in range(cfg_int(cfg, "max_epochs", 200)):
        model.train()
        for xb, yb, wb, ob, bb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda).contiguous()
            yb = yb.to(device, non_blocking=use_cuda)
            wb = wb.to(device, non_blocking=use_cuda)
            ob = ob.to(device, non_blocking=use_cuda)
            bb = bb.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            loss = _mixed_loss(model(xb), yb, wb, ob, bb, transform_cfg, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if val_loader is None:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            continue

        model.eval()
        with torch.no_grad():
            losses = []
            for xb, yb, wb, ob, bb in val_loader:
                xb = xb.to(device, non_blocking=use_cuda).contiguous()
                yb = yb.to(device, non_blocking=use_cuda)
                wb = wb.to(device, non_blocking=use_cuda)
                ob = ob.to(device, non_blocking=use_cuda)
                bb = bb.to(device, non_blocking=use_cuda)
                losses.append(_mixed_loss(model(xb), yb, wb, ob, bb, transform_cfg, cfg).item())

        val_loss = float(np.mean(losses)) if losses else np.inf
        if trial is not None:
            trial.report(val_loss, int(trial_step_offset) + epoch)
            if trial.should_prune():
                import optuna

                raise optuna.exceptions.TrialPruned()

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg_int(cfg, "patience", 20):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "features": features,
        "fill": fill,
        "mean": mean,
        "std": std,
        "sequence_length": sequence_length,
        "model": model.to("cpu").eval(),
        "target_transform": transform_cfg,
    }


def apply(df, bundle):
    _require_torch()

    prepared = prepare_ml_dataframe(sort_frame(df, preserve_order=True))
    hs = numeric_values(prepared, HS_MODEL, dtype=np.float32)
    transform_cfg = bundle["target_transform"]
    prepared, extras = augment_quantile_features(
        prepared,
        hs,
        transform_cfg,
        reference_values=hs,
    )
    base_values = extras[HS_QUANTILE_BASELINE]
    quantiles = extras[HS_QUANTILE]

    X, _ = feature_matrix(
        prepared,
        bundle["features"],
        fill=bundle["fill"],
        mean=bundle["mean"],
        std=bundle["std"],
    )
    source_column = "source" if "source" in prepared.columns else None
    segments = _segment_ids(prepared, TIME, source_col=source_column)
    X_seq, _, end_idx = _make_sequences(
        X,
        np.zeros(len(prepared), dtype=np.float32),
        segments,
        bundle["sequence_length"],
        np.isfinite(hs),
    )

    pred_target = np.full(len(prepared), np.nan, dtype=np.float32)
    if len(X_seq):
        device = _resolve_device()
        pred_target[end_idx] = _predict(
            bundle["model"].to(device).eval(),
            X_seq,
            batch_size=256,
            device=device,
        )

    pred_target = protect_tail_residuals(pred_target, quantiles, transform_cfg)

    corrected = hs.copy()
    restored = invert_target(pred_target, base_values, transform_cfg)
    mask = np.isfinite(restored)
    corrected[mask] = restored[mask]
    prepared[HS_MODEL] = clip_nonnegative(corrected)
    return restore_frame_order(prepared)
