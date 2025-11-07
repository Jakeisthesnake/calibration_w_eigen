#!/usr/bin/env python3
import json
import glob
import os
import re
import matplotlib.pyplot as plt

# -------------- Helpers --------------
def extract_iteration(fname):
    m = re.search(r"calib_iter_(\d+)\.json", fname)
    return int(m.group(1)) if m else -1

# -------------- Load all files --------------
files = glob.glob("calib_iter_*.json")
files = sorted(files, key=extract_iteration)

if not files:
    print("No calib_iter_*.json files found!")
    exit(0)

iterations = []
intrinsics = {0: [], 1: [], 2: []}
distortion = {0: [], 1: [], 2: []}
qvec = {1: [], 2: []}
tvec = {1: [], 2: []}

for f in files:
    with open(f, "r") as fp:
        data = json.load(fp)

    it = extract_iteration(f)
    iterations.append(it)

    for cam in [0, 1, 2]:
        intrinsics[cam].append(data[f"camera{cam}"]["intrinsics"])
        distortion[cam].append(data[f"camera{cam}"]["distortion"])

    # inter-camera transforms
    qvec[1].append(data["inter_camera"]["camera1_to_camera0"]["quaternion"])
    qvec[2].append(data["inter_camera"]["camera2_to_camera0"]["quaternion"])
    tvec[1].append(data["inter_camera"]["camera1_to_camera0"]["translation_vector"])
    tvec[2].append(data["inter_camera"]["camera2_to_camera0"]["translation_vector"])

# -------------- Convert lists->columns --------------
import numpy as np
def col(v): return np.array(v)

iters = np.array(iterations)

# -------------- Plotting --------------
def plot_group(values, title, labels):
    arr = col(values)
    plt.figure()
    for i in range(arr.shape[1]):
        plt.plot(iters, arr[:, i], marker='o', label=labels[i])
    plt.title(title)
    plt.xlabel("Iteration")
    plt.legend()
    plt.grid(True)

# # intrinsics fx, fy, cx, cy
# for cam in [0, 1, 2]:
#     plot_group(intrinsics[cam],
#                f"Camera {cam} Intrinsics",
#                ["fx", "fy", "cx", "cy"])

# # distortion k0..k3
# for cam in [0, 1, 2]:
#     plot_group(distortion[cam],
#                f"Camera {cam} Distortion",
#                ["k0", "k1", "k2", "k3"])

# quaternions
plot_group(qvec[1], "Camera1→Cam0 Quaternion", ["qw", "qx", "qy", "qz"])
plot_group(qvec[2], "Camera2→Cam0 Quaternion", ["qw", "qx", "qy", "qz"])

# translations
plot_group(tvec[1], "Camera1→Cam0 Translation", ["tx", "ty", "tz"])
plot_group(tvec[2], "Camera2→Cam0 Translation", ["tx", "ty", "tz"])

plt.show()
