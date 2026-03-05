from src.bias_correction.methods import linear
from src.bias_correction.methods import qm
from src.bias_correction.methods import rf


METHODS = {
    "linear": linear,
    "qm": qm,
    "rf": rf,
}

def get_method(name):

    if name not in METHODS:
        raise ValueError(f"Unknown BC method {name}")

    return METHODS[name]