from src.bias_correction.methods import linear
from src.bias_correction.methods import pqm
from src.bias_correction.methods import dagqm
from src.bias_correction.methods import gam
from src.bias_correction.methods import gpr
from src.bias_correction.methods import xgboost
from src.bias_correction.methods import transformer

METHODS = {
    "linear": linear,
    "pqm": pqm,
    "dagqm": dagqm,
    "gam": gam,
    "gpr": gpr,
    "xgboost": xgboost,
    "transformer": transformer,
}


def get_method(name):
    if name not in METHODS:
        raise ValueError(f"Unknown BC method {name}")
    return METHODS[name]