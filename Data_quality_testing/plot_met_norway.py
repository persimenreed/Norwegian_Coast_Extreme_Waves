import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
from datetime import datetime
from calendar import monthrange


def parse_yymm(yymm, is_start=True):
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])

    if is_start:
        return datetime(year, month, 1, 0, 0, 0)
    else:
        last_day = monthrange(year, month)[1]
        return datetime(year, month, last_day, 23, 59, 59)


def plot_significant_wave_height_combined(csv_files, start_date=None, end_date=None):

    # Read and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['time'] = pd.to_datetime(df['time'], format="mixed")
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.sort_values('time').reset_index(drop=True)

    # Apply optional time filtering
    if start_date is not None:
        df_combined = df_combined[df_combined['time'] >= start_date]
    if end_date is not None:
        df_combined = df_combined[df_combined['time'] <= end_date]

    # === Plot: two subplots ===
    fig, (ax_hs, ax_tp) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(16, 8),
        sharex=True
    )

    # --- Hs ---
    ax_hs.plot(
        df_combined['time'],
        df_combined['Significant_Wave_Height_Hm0'],
        linewidth=1.2,
        label='Significant Wave Height (Hm0)'
    )
    ax_hs.set_ylabel('Hm0 (m)')
    ax_hs.set_title(
        f'Wave Height and Spectral Peak Period Timeseries - {args.location}',
        fontsize=14,
        fontweight='bold'
    )
    ax_hs.grid(True, alpha=0.3)
    ax_hs.legend()

    # --- Tp ---
    ax_tp.plot(
        df_combined['time'],
        df_combined['Wave_Peak_Period'],
        linewidth=1.2,
        color='orange',  # Change color to orange
        label='Wave Spectral Peak Period (Tp)'
    )
    ax_tp.set_ylabel('Tp (s)')
    ax_tp.set_xlabel('Time')
    ax_tp.grid(True, alpha=0.3)
    ax_tp.legend()

    fig.autofmt_xdate(rotation=45, ha='right')
    fig.tight_layout()

    output_file = f"output/{args.location}_combined_wave_height_tp.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # === Statistics ===

    # Hs statistics
    hs = df_combined['Significant_Wave_Height_Hm0']
    n_total = len(hs)
    n_missing_hs = hs.isna().sum()

    print("\nSignificant Wave Height (Hs) Statistics (Filtered):")
    print(f"  Mean: {hs.mean():.2f} m")
    print(f"  Min: {hs.min():.2f} m")
    print(f"  Max: {hs.max():.2f} m")
    print(f"  Std Dev: {hs.std():.2f} m")
    print(f"  Missing values: {n_missing_hs} / {n_total} ({100*n_missing_hs/n_total:.2f}%)")

    # Tp statistics
    tp = df_combined['Wave_Peak_Period']
    n_missing_tp = tp.isna().sum()

    print("\nWave Spectral Peak Period (Tp) Statistics (Filtered):")
    print(f"  Mean: {tp.mean():.2f} s")
    print(f"  Min: {tp.min():.2f} s")
    print(f"  Max: {tp.max():.2f} s")
    print(f"  Std Dev: {tp.std():.2f} s")
    print(f"  Missing values: {n_missing_tp} / {len(tp)} ({100*n_missing_tp/len(tp):.2f}%)")

    # Wave direction – missing values only
    direction = df_combined['Wave_Peak_Direction']
    n_missing_dir = direction.isna().sum()

    print("\nWave Direction (Peak) Missing Data:")
    print(f"  Missing values: {n_missing_dir} / {len(direction)} ({100*n_missing_dir/len(direction):.2f}%)")

    print(f"\nTime range: {df_combined['time'].min()} to {df_combined['time'].max()}")

    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot wave parameters from MET Norway buoy CSV files")
    parser.add_argument("--location", required=True, help="Buoy location name")
    parser.add_argument("--start", help="Start date in YYMM format (e.g. 2411)")
    parser.add_argument("--end", help="End date in YYMM format (e.g. 2501)")

    args = parser.parse_args()

    start_date = parse_yymm(args.start, True) if args.start else None
    end_date = parse_yymm(args.end, False) if args.end else None

    csv_files = sorted(glob.glob(f"../data/met_norway/{args.location}/*.csv"))

    if not csv_files:
        print(f"No CSV files found in ../data/met_norway/{args.location}/")
    else:
        print(f"Found {len(csv_files)} CSV files")
        plot_significant_wave_height_combined(csv_files, start_date, end_date)
        print("\nPlot complete!")
