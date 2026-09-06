#!/usr/bin/env python3
"""Raw thermal logger for the TOPDON TC001-A (P2 modality), runs on the Pi.

The TC001 streams a 256x384 YUYV frame at 25 fps: the top 256x192 is the
8-bit false-colour image, the bottom 256x192 carries the raw 16-bit
radiometric data. We open the device with CONVERT_RGB disabled to get the
untouched YUYV bytes, and per frame we record CLOCK_MONOTONIC_RAW + UTC
plus summary statistics over the radiometric half (mean/max/centre-ROI),
which is what tracks GPU temperature. Full raw frames are saved
periodically for exact offline decoding.

Cross-machine sync: the Pi and the GPU node have independent monotonic
clocks, so alignment to the node's telemetry is via the UTC anchors
(both NTP-disciplined) plus the workload's own thermal rise. Stops on a
duration limit or when the --stop-file appears.
"""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def open_camera(device: int, w: int, h: int):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def radiometric_stats(frame: np.ndarray, w=256, h=384):
    """Return (mean, max, centre-ROI mean) of the raw 16-bit thermal half.

    The bottom half (rows h/2..h) holds the radiometric data; interpret its
    bytes as little-endian uint16. Falls back to whole-buffer stats if the
    frame shape is unexpected, so a signal is always produced.
    """
    buf = np.asarray(frame).reshape(-1).view(np.uint8)
    expected = w * h * 2
    if buf.size < expected:
        vals = buf.astype(np.uint16)
        return float(vals.mean()), float(vals.max()), float(vals.mean())
    raw16 = buf[:expected].view("<u2").reshape(h, w)
    thermal = raw16[h // 2:, :]            # bottom half = radiometric plane
    cy, cx = thermal.shape[0] // 2, thermal.shape[1] // 2
    roi = thermal[cy - 32:cy + 32, cx - 32:cx + 32]
    return float(thermal.mean()), float(thermal.max()), float(roi.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--duration-s", type=float, default=1800)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--full-frame-every-s", type=float, default=5.0)
    ap.add_argument("--stop-file", type=Path, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    cap = open_camera(args.device, args.width, args.height)
    if not cap.isOpened():
        print("ERROR: could not open camera", flush=True)
        return 2

    t0 = raw_now()
    utc0 = datetime.now(timezone.utc).isoformat()
    last_full = 0.0
    n = 0
    dropped = 0
    with (args.out_dir / "thermal_series.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_raw_s", "utc", "therm_mean", "therm_max", "roi_mean"])
        w.writerow([-1, t0, utc0, "anchor", "anchor", "anchor"])
        while raw_now() - t0 < args.duration_s:
            if args.stop_file and args.stop_file.exists():
                break
            ok, frame = cap.read()
            t = raw_now()
            if not ok or frame is None:
                dropped += 1
                time.sleep(0.01)
                continue
            mean, mx, roi = radiometric_stats(frame, args.width, args.height)
            w.writerow([n, f"{t:.6f}", datetime.now(timezone.utc).isoformat(),
                        f"{mean:.2f}", f"{mx:.2f}", f"{roi:.2f}"])
            if t - last_full >= args.full_frame_every_s:
                np.save(frames_dir / f"frame_{n:06d}_{t:.3f}.npy", frame)
                last_full = t
            n += 1
            if n % 200 == 0:
                fh.flush()
    cap.release()
    print(f"thermal_logger: {n} frames, {dropped} dropped, out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
