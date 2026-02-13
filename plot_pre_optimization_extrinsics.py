#!/usr/bin/env python3
"""
Plot pre-optimization per-frame extrinsics from pre_optimization_extrinsics_per_frame.json
to visualize consistency of cam1→cam0, cam2→cam0, and cam2→cam1 across frames.
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def quat_to_euler_xyz(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw) in radians, XYZ order."""
    # Roll (x), Pitch (y), Yaw (z)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def load_pre_opt_extrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot pre-optimization per-frame extrinsics consistency")
    parser.add_argument(
        "json_file",
        nargs="?",
        default="pre_optimization_extrinsics_per_frame.json",
        help="Path to pre_optimization_extrinsics_per_frame.json",
    )
    parser.add_argument("-o", "--output", help="Save figure to this path instead of showing")
    args = parser.parse_args()

    data = load_pre_opt_extrinsics(args.json_file)
    frames = data["frames"]

    # Keys to extract: (json_key, label)
    transform_keys = [
        ("camera1_to_camera0", "cam1→cam0"),
        ("camera2_to_camera0", "cam2→cam0"),
        ("camera2_to_camera1", "cam2→cam1"),
    ]

    # Collect per-transform data: list of (frame_index, timestamp_id, quat[4], tvec[3])
    collected = {label: [] for _, label in transform_keys}

    for fr in frames:
        idx = fr["frame_index"]
        ts = fr["timestamp_id"]
        for key, label in transform_keys:
            if fr.get(key) is not None:
                q = fr[key]["quaternion"]
                t = fr[key]["translation"]
                collected[label].append((idx, ts, q[0], q[1], q[2], q[3], t[0], t[1], t[2]))

    if not any(collected.values()):
        print("No extrinsics data found in", args.json_file)
        return

    def to_arrays(cam_data):
        if not cam_data:
            return None
        idx = np.array([x[0] for x in cam_data])
        ts = np.array([x[1] for x in cam_data])
        qw = np.array([x[2] for x in cam_data])
        qx = np.array([x[3] for x in cam_data])
        qy = np.array([x[4] for x in cam_data])
        qz = np.array([x[5] for x in cam_data])
        tx = np.array([x[6] for x in cam_data])
        ty = np.array([x[7] for x in cam_data])
        tz = np.array([x[8] for x in cam_data])
        roll, pitch, yaw = quat_to_euler_xyz(qw, qx, qy, qz)
        return {
            "frame_index": idx,
            "timestamp_id": ts,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": tx, "ty": ty, "tz": tz,
            "roll": roll, "pitch": pitch, "yaw": yaw,
        }

    datasets = [(label, to_arrays(collected[label])) for _, label in transform_keys]
    datasets = [(label, d) for label, d in datasets if d is not None]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(datasets) * 3, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ci = 0

    # Row 0: Translation (tx, ty, tz) vs frame index
    ax_t = axes[0, 0]
    for label, d in datasets:
        ax_t.plot(d["frame_index"], d["tx"], "o-", label=label + " tx", markersize=4, color=colors[ci % len(colors)])
        ax_t.plot(d["frame_index"], d["ty"], "s-", label=label + " ty", markersize=4, color=colors[(ci + 1) % len(colors)])
        ax_t.plot(d["frame_index"], d["tz"], "^-", label=label + " tz", markersize=4, color=colors[(ci + 2) % len(colors)])
        ax_t.axhline(np.mean(d["tx"]), color=colors[ci % len(colors)], linestyle="--", alpha=0.6)
        ax_t.axhline(np.mean(d["ty"]), color=colors[(ci + 1) % len(colors)], linestyle="--", alpha=0.6)
        ax_t.axhline(np.mean(d["tz"]), color=colors[(ci + 2) % len(colors)], linestyle="--", alpha=0.6)
        ci += 3
    ax_t.set_xlabel("Frame index")
    ax_t.set_ylabel("Translation (m)")
    ax_t.set_title("Pre-optimization extrinsics: translation vs frame")
    ax_t.legend(loc="best", fontsize=7)
    ax_t.grid(True, alpha=0.3)

    # Row 0, col 1: Rotation (roll, pitch, yaw) vs frame index
    ax_r = axes[0, 1]
    ci = 0
    for label, d in datasets:
        ax_r.plot(d["frame_index"], np.degrees(d["roll"]), "o-", label=label + " roll", markersize=4, color=colors[ci % len(colors)])
        ax_r.plot(d["frame_index"], np.degrees(d["pitch"]), "s-", label=label + " pitch", markersize=4, color=colors[(ci + 1) % len(colors)])
        ax_r.plot(d["frame_index"], np.degrees(d["yaw"]), "^-", label=label + " yaw", markersize=4, color=colors[(ci + 2) % len(colors)])
        ci += 3
    ax_r.set_xlabel("Frame index")
    ax_r.set_ylabel("Angle (deg)")
    ax_r.set_title("Pre-optimization extrinsics: rotation (Euler) vs frame")
    ax_r.legend(loc="best", fontsize=7)
    ax_r.grid(True, alpha=0.3)

    # Row 1: Quaternion components vs frame index
    ax_q = axes[1, 0]
    ci = 0
    for label, d in datasets:
        ax_q.plot(d["frame_index"], d["qw"], "o-", label=label + " qw", markersize=4, color=colors[ci % len(colors)])
        ax_q.plot(d["frame_index"], d["qx"], "s-", label=label + " qx", markersize=4, color=colors[(ci + 1) % len(colors)])
        ax_q.plot(d["frame_index"], d["qy"], "^-", label=label + " qy", markersize=4, color=colors[(ci + 2) % len(colors)])
        ax_q.plot(d["frame_index"], d["qz"], "d-", label=label + " qz", markersize=4, color=colors[(ci + 3) % len(colors)])
        ci += 4
    ax_q.set_xlabel("Frame index")
    ax_q.set_ylabel("Quaternion component")
    ax_q.set_title("Pre-optimization extrinsics: quaternion vs frame")
    ax_q.legend(loc="best", fontsize=7)
    ax_q.grid(True, alpha=0.3)

    # Row 1, col 1: Translation mean ± std per transform
    ax_std = axes[1, 1]
    for i, (label, d) in enumerate(datasets):
        x_off = i * 1.0
        ax_std.errorbar(x_off, np.mean(d["tx"]), yerr=np.std(d["tx"]), fmt="o", capsize=5, label=label + " tx", color=colors[i * 3 % len(colors)])
        ax_std.errorbar(x_off + 0.25, np.mean(d["ty"]), yerr=np.std(d["ty"]), fmt="s", capsize=5, label=label + " ty", color=colors[(i * 3 + 1) % len(colors)])
        ax_std.errorbar(x_off + 0.5, np.mean(d["tz"]), yerr=np.std(d["tz"]), fmt="^", capsize=5, label=label + " tz", color=colors[(i * 3 + 2) % len(colors)])
    ax_std.set_xlabel("(transform offset)")
    ax_std.set_ylabel("Translation (m) ± std")
    ax_std.set_title("Pre-optimization: translation mean ± std across frames")
    ax_std.legend(loc="best", fontsize=7)
    ax_std.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print("Saved", args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
