from contextlib import contextmanager
from copy import deepcopy

MODEL_PROFILES = {
    "xgboost_fedjeosen": {
        "tail_weight_q90": 2.0,
        "tail_weight_q95": 5.0,
        "tail_weight_q99": 13.0,
        "n_estimators": 544,
        "max_depth": 4,
        "learning_rate": 0.04500643668944804,
        "subsample": 0.8150366480747694,
        "colsample_bytree": 0.7003490840108048,
        "gamma": 4.78911928209132,
        "min_child_weight": 6.913979262417389,
        "reg_alpha": 0.03104853768658029,
        "reg_lambda": 0.02312678477285944,
        "target_eps": 0.001,
    },
    "xgboost_fauskane": {
        "tail_weight_q90": 3.0,
        "tail_weight_q95": 6.0,
        "tail_weight_q99": 7.0,
        "n_estimators": 666,
        "max_depth": 3,
        "learning_rate": 0.17326525924425426,
        "subsample": 0.8682010457760649,
        "colsample_bytree": 0.9401477693054193,
        "gamma": 3.5632281107343986,
        "min_child_weight": 0.3459140262535759,
        "reg_alpha": 0.9939675581491936,
        "reg_lambda": 0.42812436488697486,
        "target_eps": 1e-06,
    },
    "transformer_fedjeosen": {
        "sequence_length": 14,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 96,
        "dropout": 0.022707794228104636,
        "learning_rate": 0.00047756529133542703,
        "weight_decay": 0.0002730809780914489,
        "batch_size": 32,
        "target_eps": 1e-05,
        "tail_weight_q90": 1.5,
        "tail_weight_q95": 4.5,
        "tail_weight_q99": 6.5,
    },
    "transformer_fauskane": {
        "sequence_length": 13,
        "d_model": 32,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 96,
        "dropout": 0.04397545124330803,
        "learning_rate": 0.0012765382440310617,
        "weight_decay": 6.672524361038236e-06,
        "batch_size": 64,
        "target_eps": 0.0001,
        "tail_weight_q90": 2.5,
        "tail_weight_q95": 5.5,
        "tail_weight_q99": 6.5,
    },
    "ensemble_xgboost": {
        "tail_weight_q90": 3.0,
        "tail_weight_q95": 6.0,
        "tail_weight_q99": 10.0,
        "n_estimators": 329,
        "max_depth": 8,
        "learning_rate": 0.06806108735563433,
        "subsample": 0.8580474209660353,
        "colsample_bytree": 0.9979666107350993,
        "gamma": 0.018484049342889122,
        "min_child_weight": 1.9426354946377948,
        "reg_alpha": 4.45088393525389,
        "reg_lambda": 0.1795998167091841,
        "tail_aware": True,
        "tail_strength_q95": 0.39657458470491386,
        "tail_strength_q99": 0.29779847207711807,
        "random_state": 1,
    },
    "ensemble_xgboost_fedjeosen": {
        "tail_weight_q90": 2.5,
        "tail_weight_q95": 5.5,
        "tail_weight_q99": 11.5,
        "n_estimators": 625,
        "max_depth": 4,
        "learning_rate": 0.06126274599758409,
        "subsample": 0.8792535116125372,
        "colsample_bytree": 0.5134249806502195,
        "gamma": 0.07171974970003875,
        "min_child_weight": 3.193045061034521,
        "reg_alpha": 4.525452370788657,
        "reg_lambda": 3.912258284837049,
        "tail_aware": True,
        "tail_strength_q95": 0.3921347430093849,
        "tail_strength_q99": 0.20062871404868463,
        "random_state": 1,
    },
    "ensemble_xgboost_fauskane": {
        "tail_weight_q90": 2.0,
        "tail_weight_q95": 4.0,
        "tail_weight_q99": 6.0,
        "n_estimators": 369,
        "max_depth": 7,
        "learning_rate": 0.024460150670228255,
        "subsample": 0.9734069944906701,
        "colsample_bytree": 0.6927675622832247,
        "gamma": 0.0493303887402711,
        "min_child_weight": 2.0048996321337262,
        "reg_alpha": 1.8300028816813942,
        "reg_lambda": 0.9597084376656845,
        "tail_aware": True,
        "tail_strength_q95": 0.3410804907066404,
        "tail_strength_q99": 0.7987265824536579,
        "random_state": 1,
    },
}


def resolve_profile(defaults, *profile_names):
    profile = deepcopy(defaults)
    for name in profile_names:
        if name:
            profile.update(MODEL_PROFILES.get(str(name), {}))
    return profile


@contextmanager
def override_profile(profile_name, params):
    profile_name = str(profile_name)
    existed = profile_name in MODEL_PROFILES
    original = deepcopy(MODEL_PROFILES.get(profile_name, {}))

    MODEL_PROFILES.setdefault(profile_name, {})
    MODEL_PROFILES[profile_name].update(params)

    try:
        yield
    finally:
        if existed:
            MODEL_PROFILES[profile_name] = original
        else:
            MODEL_PROFILES.pop(profile_name, None)
