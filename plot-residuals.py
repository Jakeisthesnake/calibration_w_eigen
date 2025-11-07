#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/home/jake/calibration_w_eigen"
OUT_DIR = "./residual_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Load all CSVs named iter_###.csv
# --------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(DATA_DIR, "iter_*.csv")))
if not files:
    raise RuntimeError("No iter_*.csv files found.")
dfs = []
for f in files:
    # read as raw text
    df = pd.read_csv(f, header=None)
    df.columns = ["cam", "frame", "point", "res_u", "res_v", "res_norm"]

    # strip whitespace on string columns
    df["cam"] = df["cam"].str.strip()

    # force numeric parsing
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    df["res_u"] = pd.to_numeric(df["res_u"], errors="coerce")
    df["res_v"] = pd.to_numeric(df["res_v"], errors="coerce")
    df["res_norm"] = pd.to_numeric(df["res_norm"], errors="coerce")

    # add iteration index
    df["iter"] = int(os.path.basename(f).split("_")[1].split(".")[0])

    dfs.append(df)


data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=["res_norm"])

iterations = sorted(data["iter"].unique())
cams = sorted(data["cam"].unique())
frames = sorted(data["frame"].unique())

# --------------------------------------------------------------------
# Compute summary stats
# --------------------------------------------------------------------
stats = data.groupby("iter")["res_norm"].agg(["mean", "median", "std", "max"])

# --------------------------------------------------------------------
# 1) Mean, median, std, max vs iteration
# --------------------------------------------------------------------
plt.figure()
stats[["mean", "median", "std", "max"]].plot()
plt.title("Residual Summary Statistics vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Residual (pixels)")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/summary_stats.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 2) Overlaid histograms
# --------------------------------------------------------------------
plt.figure()
for it in iterations:
    subset = data[data.iter == it]
    plt.hist(subset.res_norm, bins=40, alpha=0.3, label=f"iter {it}")
plt.title("Residual Histograms Overlay")
plt.xlabel("res_norm")
plt.ylabel("count")
plt.legend()
plt.savefig(f"{OUT_DIR}/histograms.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 3) Boxplot per iteration
# --------------------------------------------------------------------
plt.figure()
data.boxplot(column="res_norm", by="iter")
plt.title("Residual Boxplot by Iteration")
plt.suptitle("")
plt.xlabel("Iteration")
plt.ylabel("res_norm")
plt.savefig(f"{OUT_DIR}/boxplot.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 4) Per-camera mean residual trend
# --------------------------------------------------------------------
cam_stats = data.groupby(["iter", "cam"])["res_norm"].mean().unstack()
plt.figure()
cam_stats.plot()
plt.title("Mean Residual per Camera vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("res_norm")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/per_camera_mean.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 5) Per-frame mean residual trend
# --------------------------------------------------------------------
frame_stats = data.groupby(["iter", "frame"])["res_norm"].mean().unstack()
plt.figure()
frame_stats.plot()
plt.title("Mean Residual per Frame vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("res_norm")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/per_frame_mean.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 6) res_u / res_v std over iteration
# --------------------------------------------------------------------
uv_stats = data.groupby("iter")[["res_u", "res_v"]].std()
plt.figure()
uv_stats.plot()
plt.title("Std of res_u/res_v vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("std (pixels)")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/uv_std.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 7) res_u vs res_v scatter (grid per iteration)
# --------------------------------------------------------------------
n = len(iterations)
cols = 4
rows = int(np.ceil(n / cols))
plt.figure(figsize=(4 * cols, 4 * rows))
for i, it in enumerate(iterations):
    subset = data[data.iter == it]
    plt.subplot(rows, cols, i + 1)
    plt.scatter(subset.res_u, subset.res_v, s=5, alpha=0.5)
    plt.title(f"iter {it}")
    plt.xlabel("res_u")
    plt.ylabel("res_v")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/uv_scatter_grid.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 8) Residual vs radius
# --------------------------------------------------------------------
# Estimate image plane radius from residual offsets
data["radius"] = np.sqrt(data.res_u**2 + data.res_v**2)

plt.figure()
plt.scatter(data.radius, data.res_norm, s=4, alpha=0.3)
plt.title("Residual Norm vs Radius")
plt.xlabel("Pixel radius (approx)")
plt.ylabel("res_norm")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/radius_correlation.png", dpi=200)
plt.close()

# --------------------------------------------------------------------
# 9) Outlier counts vs iteration
# --------------------------------------------------------------------
thresholds = [200, 500, 800, 1000]  # adjust as needed
outlier_counts = {}

for t in thresholds:
    # More modern Pandas: use GroupBy.sum directly
    outlier_counts[t] = data.groupby("iter")["res_norm"].apply(lambda x: np.sum(x > t))

plt.figure()

for t in thresholds:
    # Convert Series to aligned array of values
    plt.plot(
        iterations,
        outlier_counts[t].reindex(iterations).values,
        label=f">{t}px"
    )

plt.title("Outlier Counts vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_DIR}/outliers.png", dpi=200)
plt.close()

print(f"Done! Plots saved to {OUT_DIR}")
