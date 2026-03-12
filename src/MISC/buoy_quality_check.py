from __future__ import annotations

import csv, glob, math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results/MISC/buoy_quality_check"

CORE_COLUMNS = [
    "Significant_Wave_Height_Hm0",
    "Wave_Height_Hmax",
    "Wave_Period_Tz",
    "Wave_Peak_Period",
    "Wave_Mean_Period_Tm02",
]

PERIOD_COLUMNS = {
    "Wave_Period_Tz",
    "Wave_Peak_Period",
    "Wave_Mean_Period_Tm02",
}

COLOR = dict(
    all="gray",
    isolated_spike="red",
    extreme_but_coherent="orange",
    physics_check="blue",
)

@dataclass
class Dataset:
    location: str
    times: list[datetime]
    series: dict[str, np.ndarray]

@dataclass
class Flag:
    idx: int
    column: str
    value: float
    classification: str
    z: float | None


# -------------------------
# IO
# -------------------------

def load_dataset(path: Path):

    times, cols = [], {}

    with path.open() as f:
        reader = csv.DictReader(f)
        cols = {c: [] for c in reader.fieldnames if c != "time"}

        for r in reader:
            try:
                times.append(datetime.strptime(r["time"], TIME_FORMAT))
            except:
                continue

            for c in cols:
                try: cols[c].append(float(r[c]))
                except: cols[c].append(math.nan)

    return Dataset(
        location=path.stem.replace("buoy_","").replace("_max",""),
        times=times,
        series={k: np.array(v,float) for k,v in cols.items()}
    )


# -------------------------
# QC metrics
# -------------------------

def robust_z(x):

    finite = x[np.isfinite(x)]
    if len(finite)==0: return np.full_like(x,np.nan)

    med = np.median(finite)
    mad = np.median(np.abs(finite-med))
    if mad==0: return np.full_like(x,np.nan)

    z = np.full_like(x,np.nan)
    mask = np.isfinite(x)
    z[mask] = 0.6745*(x[mask]-med)/mad
    return z


def local_spike(x):

    s = np.full_like(x,np.nan)

    for i in range(1,len(x)-1):
        a,b,c = x[i-1],x[i],x[i+1]
        if np.isfinite([a,b,c]).all():
            s[i] = abs(b - np.median([a,c]))

    return s


# -------------------------
# Flagging
# -------------------------

def build_flags(ds: Dataset):

    flags=[]
    hs = ds.series.get("Significant_Wave_Height_Hm0")

    for col in CORE_COLUMNS:

        v = ds.series.get(col)
        if v is None: continue

        rz = robust_z(v)
        spike = local_spike(v)

        for i,val in enumerate(v):

            if not np.isfinite(val): continue

            cls=None

            if np.isfinite(rz[i]) and rz[i]>6:
                cls = "isolated_spike" if spike[i]>1 else "extreme_but_coherent"

            if col=="Wave_Height_Hmax" and hs is not None and val<hs[i]:
                cls="physics_check"

            if cls:
                flags.append(Flag(i,col,float(val),cls,
                    float(rz[i]) if np.isfinite(rz[i]) else None))

    return flags


# -------------------------
# Plot helpers
# -------------------------

def legend(include_cloud=False, include_line=False):

    h=[]

    if include_cloud:
        h.append(Line2D([],[],marker="o",linestyle="",color=COLOR["all"],alpha=0.25,label="All"))

    if include_line:
        h.append(Line2D([],[],linestyle="--",color="black",label="1:1"))

    h += [
        Line2D([],[],marker="o",linestyle="",color=COLOR["isolated_spike"],label="Spike"),
        Line2D([],[],marker="o",linestyle="",color=COLOR["extreme_but_coherent"],label="Extreme")
    ]

    return h


# -------------------------
# Plots
# -------------------------

def plot_wave_height(ds,flags,path):

    hs = ds.series["Significant_Wave_Height_Hm0"]

    plt.figure(figsize=(12,4))
    plt.plot(ds.times,hs,color="gray")

    for f in flags:
        if f.column!="Significant_Wave_Height_Hm0": continue
        plt.scatter(ds.times[f.idx],f.value,color=COLOR[f.classification])

    plt.title(f"{ds.location} Wave Height QC")
    plt.ylabel("Hs (m)")
    plt.legend(handles=legend())
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_periods(ds,flags,path):

    hs = ds.series["Significant_Wave_Height_Hm0"]
    per = ds.series.get("Wave_Mean_Period_Tm02")
    tz = ds.series.get("Wave_Period_Tz")

    fig,ax=plt.subplots(2,1,figsize=(10,8))

    ax[0].scatter(hs,per,s=2,alpha=0.2,color="gray")
    ax[1].scatter(hs,tz,s=2,alpha=0.2,color="gray")

    for f in flags:
        if f.column not in PERIOD_COLUMNS: continue

        h=hs[f.idx]
        c=COLOR[f.classification]

        if f.column=="Wave_Mean_Period_Tm02":
            ax[0].scatter(h,f.value,c=c,s=20,zorder=5)

        if f.column=="Wave_Period_Tz":
            ax[1].scatter(h,f.value,c=c,s=20,zorder=5)

    ax[0].set_title("Hs vs Tm02")
    ax[1].set_title("Hs vs Tz")

    for a in ax:
        a.set_xlabel("Hs (m)")
        a.set_ylabel("Period (s)")
        a.legend(handles=legend(True))

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_tz_tm02(ds,flags,path):

    tz=ds.series.get("Wave_Period_Tz")
    tm02=ds.series.get("Wave_Mean_Period_Tm02")

    if tz is None or tm02 is None: return

    mask=np.isfinite(tz)&np.isfinite(tm02)

    fig,ax=plt.subplots(figsize=(8,7))

    ax.scatter(tz[mask],tm02[mask],s=4,alpha=0.18,color="gray")

    lo=min(tz[mask].min(),tm02[mask].min())
    hi=max(tz[mask].max(),tm02[mask].max())
    ax.plot([lo,hi],[lo,hi],"--",color="black",alpha=0.6)

    for f in flags:
        if f.column not in {"Wave_Period_Tz","Wave_Mean_Period_Tm02"}: continue
        if not np.isfinite([tz[f.idx],tm02[f.idx]]).all(): continue

        ax.scatter(tz[f.idx],tm02[f.idx],
                   c=COLOR[f.classification],s=30,
                   edgecolors="black",linewidths=0.4)

    ax.set_title(f"{ds.location} Tz vs Tm02")
    ax.set_xlabel("Tz (s)")
    ax.set_ylabel("Tm02 (s)")
    ax.legend(handles=legend(True,True))
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def save_flags(ds, flags):

    out = OUTPUT_DIR / f"{ds.location}_flags.csv"

    with out.open("w", newline="") as f:

        w = csv.writer(f)
        w.writerow(["time","column","value","classification","robust_z"])

        for fl in flags:

            w.writerow([
                ds.times[fl.idx].strftime(TIME_FORMAT),
                fl.column,
                fl.value,
                fl.classification,
                fl.z
            ])


# -------------------------
# Summary
# -------------------------

def summarize(ds, flags):

    summary_rows = []

    print(f"\n=== {ds.location} ===")
    print("Rows:",len(ds.times))

    for c in CORE_COLUMNS:

        x = ds.series.get(c)
        if x is None:
            continue

        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            continue

        rz = robust_z(x)

        mean = np.mean(finite)
        std = np.std(finite)
        maxv = np.max(finite)
        maxz = np.nanmax(abs(rz))

        summary_rows.append([
            ds.location, c, len(finite), mean, std, maxv, maxz
        ])

        print(
            f"{c}: "
            f"mean={mean:.2f} "
            f"std={std:.2f} "
            f"max={maxv:.2f} "
            f"max|z|={maxz:.2f}"
        )

    print("flags:", len(flags))

    out = OUTPUT_DIR / f"{ds.location}_summary.csv"

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["location","parameter","n","mean","std","max","max_abs_robust_z"])
        w.writerows(summary_rows)


# -------------------------
# Main
# -------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    files=glob.glob("data/**/buoy_*_max.csv",recursive=True)
    if not files:
        raise RuntimeError("No buoy files found")

    for f in files:

        path=Path(f)
        ds=load_dataset(path)
        flags=build_flags(ds)

        plot_wave_height(ds,flags,OUTPUT_DIR/f"{ds.location}_wave_height.png")
        plot_periods(ds,flags,OUTPUT_DIR/f"{ds.location}_periods.png")
        plot_tz_tm02(ds,flags,OUTPUT_DIR/f"{ds.location}_tz_tm02_relationship.png")

        summarize(ds,flags)
        save_flags(ds,flags)


if __name__=="__main__":
    main()