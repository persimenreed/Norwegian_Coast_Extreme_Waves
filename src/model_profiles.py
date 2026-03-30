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
        "tail_weight_q99": 8.0,
        "n_estimators": 274,
        "max_depth": 3,
        "learning_rate": 0.07278757815903632,
        "subsample": 0.7847940632749715,
        "colsample_bytree": 0.8788646010732099,
        "gamma": 0.03481193550773202,
        "min_child_weight": 1.2098172168901085,
        "reg_alpha": 1.138698687979474,
        "reg_lambda": 4.767161352689669,
        "tail_aware": True,
        "tail_strength_q95": 0.3560785191227247,
        "tail_strength_q99": 0.5530310017279734,
        "random_state": 1,
    },
    "ensemble_xgboost_fedjeosen": {
        "tail_weight_q90": 2.5,
        "tail_weight_q95": 5.5,
        "tail_weight_q99": 11.5,
        "n_estimators": 798,
        "max_depth": 8,
        "learning_rate": 0.051596459425892166,
        "subsample": 0.6560929669098633,
        "colsample_bytree": 0.5822253939967578,
        "gamma": 0.28941921609963167,
        "min_child_weight": 0.29637002963954573,
        "reg_alpha": 0.11025048924193015,
        "reg_lambda": 5.443676589384466,
        "tail_aware": True,
        "tail_strength_q95": 0.3826219200171952,
        "tail_strength_q99": 0.7717619486985879,
        "random_state": 1,
    },
    "ensemble_xgboost_fauskane": {
        "tail_weight_q90": 2.0,
        "tail_weight_q95": 3.0,
        "tail_weight_q99": 4.0,
        "n_estimators": 148,
        "max_depth": 8,
        "learning_rate": 0.06460572436299825,
        "subsample": 0.6521407689949626,
        "colsample_bytree": 0.8477922664112483,
        "gamma": 0.0617013270338804,
        "min_child_weight": 6.5929789068127445,
        "reg_alpha": 0.24270792431453708,
        "reg_lambda": 0.001218871884366766,
        "tail_aware": True,
        "tail_strength_q95": 0.29410382249842376,
        "tail_strength_q99": 0.7883565181207807,
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
