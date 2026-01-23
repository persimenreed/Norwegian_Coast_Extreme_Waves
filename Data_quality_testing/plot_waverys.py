import pandas as pd
import matplotlib.pyplot as plt
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

def plot_waverys_at_location(csv_file, latitude, longitude, start_date=None, end_date=None):
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter for specific coordinates
    df_filtered = df[
        (df['latitude'] == latitude) & 
        (df['longitude'] == longitude)
    ].copy()
    
    if df_filtered.empty:
        print(f"No data found for coordinates ({latitude}, {longitude})")
        print(f"Available coordinates: {df[['latitude', 'longitude']].drop_duplicates()}")
        return
    
    # Apply optional time filtering
    if start_date is not None:
        df_filtered = df_filtered[df_filtered['time'] >= start_date]
    if end_date is not None:
        df_filtered = df_filtered[df_filtered['time'] <= end_date]
    
    df_filtered = df_filtered.sort_values('time').reset_index(drop=True)
    
    print(f"Found {len(df_filtered)} records at ({latitude}, {longitude})")
    
    # Plot: two subplots 
    fig, (ax_vhm0, ax_vtpk) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(16, 8),
        sharex=True
    )
    
    # VHM0 (Significant Wave Height)
    ax_vhm0.plot(
        df_filtered['time'],
        df_filtered['VHM0'],
        linewidth=1.2,
        label='Significant Wave Height (VHM0)'
    )
    ax_vhm0.set_ylabel('VHM0 (m)')
    ax_vhm0.set_title(
        f'Wave Parameters at Latitude {latitude}, Longitude {longitude}',
        fontsize=14,
        fontweight='bold'
    )
    ax_vhm0.grid(True, alpha=0.3)
    ax_vhm0.legend()
    
    # VTPK (Spectral Peak Period)
    ax_vtpk.plot(
        df_filtered['time'],
        df_filtered['VTPK'],
        linewidth=1.2,
        color='orange',
        label='Spectral Peak Period (VTPK)'
    )
    ax_vtpk.set_ylabel('VTPK (s)')
    ax_vtpk.set_xlabel('Time')
    ax_vtpk.grid(True, alpha=0.3)
    ax_vtpk.legend()
    
    fig.autofmt_xdate(rotation=45, ha='right')
    fig.tight_layout()
    
    output_file = f"Data_quality_testing/output/waverys_{latitude}_{longitude}_timeseries.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Missing values summary
    n_total = len(df_filtered)
    print("\nMissing values per column:")
    for col in df_filtered.columns:
        missing = df_filtered[col].isna().sum()
        available = n_total - missing
        pct = 100 * missing / n_total if n_total else 0
        print(f"{col},{available},({pct:.2f}% missing)")
    
    print(f"\nTime range: {df_filtered['time'].min()} to {df_filtered['time'].max()}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot wave parameters from waverys CSV file")
    parser.add_argument("--start", help="Start date in YYMM format (e.g. 2411)")
    parser.add_argument("--end", help="End date in YYMM format (e.g. 2501)")
    
    args = parser.parse_args()
    
    start_date = parse_yymm(args.start, True) if args.start else None
    end_date = parse_yymm(args.end, False) if args.end else None
    
    csv_file = "data/waverys/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1768390036755.csv"
    latitude = 60.4
    longitude = 4.399994
    
    plot_waverys_at_location(csv_file, latitude, longitude, start_date, end_date)
    print("\nPlot complete!")