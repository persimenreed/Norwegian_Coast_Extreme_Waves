from pathlib import Path

from src.settings import (
    get_buoy_locations,
    get_methods,
    format_path,
)

from src.bias_correction.data import (
    load_pairs,
    load_hindcast,
    load_pooled_pairs,
)

from src.bias_correction.registry import get_method


def _run_methods(df_train, df_hind, location, prefix=""):

    methods = get_methods()

    saved = {}

    for name in methods:

        method = get_method(name)

        print(f"Training {prefix}{name}")
        model = method.fit(df_train)

        print(f"Applying {prefix}{name}")
        df_pred = method.apply(df_hind, model)

        out_path = format_path(
            "corrected",
            location=location,
            corr_method=f"{prefix}{name}",
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_pred.to_csv(out_path, index=False)

        saved[f"{prefix}{name}"] = out_path

    return saved


def run_bias_correction(location):

    buoys = get_buoy_locations()

    df_hind = load_hindcast(location)

    saved = {}

    # -------------------------
    # LOCAL BC (only if buoy)
    # -------------------------

    if location in buoys:

        print(f"Running LOCAL correction for {location}")

        df_train_local = load_pairs(location)

        saved.update(
            _run_methods(
                df_train_local,
                df_hind,
                location,
                prefix="local_",
            )
        )

    # -------------------------
    # TRANSFER BC (train on other buoy)
    # -------------------------

    if location in buoys and len(buoys) > 1:

        other_buoys = [b for b in buoys if b != location]

        for other in other_buoys:

            print(f"Running TRANSFER correction: train={other} apply={location}")

            df_train_transfer = load_pairs(other)

            saved.update(
                _run_methods(
                    df_train_transfer,
                    df_hind,
                    location,
                    prefix="transfer_",
                )
            )

    # -------------------------
    # POOLED BC (always)
    # -------------------------

    print(f"Running POOLED correction for {location}")

    df_train_pool = load_pooled_pairs()

    saved.update(
        _run_methods(
            df_train_pool,
            df_hind,
            location,
            prefix="pooled_",
        )
    )

    print("\nSaved corrected hindcasts:")
    for k in sorted(saved):
        print(f"  {k}: {saved[k]}")

    return saved