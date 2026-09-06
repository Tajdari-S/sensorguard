#!/usr/bin/env python3
"""Thermal logger via the TC001 web MJPEG stream (fallback path).

The camera is held by a root-owned `thermal-web.service` we cannot stop
without sudo, so raw radiometric capture is unavailable. This logger reads
the service's colormapped MJPEG stream instead and records a
temperature-CORRELATED proxy per frame (mean luminance, per-channel means,
centre-ROI luminance) with CLOCK_MONOTONIC_RAW + UTC timestamps.

Caveat: the stream is JPEG-compressed and false-colour mapped, so these
are QUALITATIVE thermal signals, not calibrated temperatures. They suffice
to show that thermal tracks GPU load and to identify which GPU is in frame.
For calibrated radiometry, stop thermal-web.service and use
thermal_logger.py (raw YUYV). Runs anywhere that can reach the stream;
intended to run on the Pi against http://localhost:8090/stream.
"""

import argparse
import csv
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def frames(url, timeout=10):
    """Yield (jpeg_bytes) from a multipart MJPEG stream."""
    stream = urllib.request.urlopen(url, timeout=timeout)
    buf = b""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2)
            if s < 0 or e < 0:
                break
            yield buf[s:e + 2]
            buf = buf[e + 2:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8090/stream")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--duration-s", type=float, default=1800)
    ap.add_argument("--full-frame-every-s", type=float, default=15.0)
    ap.add_argument("--stop-file", type=Path, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fdir = args.out_dir / "frames"
    fdir.mkdir(exist_ok=True)
    t0 = raw_now()
    last_full = 0.0
    n = 0
    with (args.out_dir / "thermal_series.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_raw_s", "utc", "lum_mean", "r_mean", "g_mean",
                    "b_mean", "roi_lum_mean", "lum_max"])
        w.writerow([-1, t0, datetime.now(timezone.utc).isoformat(),
                    "anchor", "anchor", "anchor", "anchor", "anchor", "anchor"])
        for jpg in frames(args.url):
            t = raw_now()
            if raw_now() - t0 >= args.duration_s:
                break
            if args.stop_file and args.stop_file.exists():
                break
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            lum = (0.299 * r + 0.587 * g + 0.114 * b)
            hh, ww = lum.shape
            roi = lum[hh // 2 - hh // 6:hh // 2 + hh // 6, ww // 2 - ww // 6:ww // 2 + ww // 6]
            w.writerow([n, f"{t:.6f}", datetime.now(timezone.utc).isoformat(),
                        f"{lum.mean():.3f}", f"{r.mean():.3f}", f"{g.mean():.3f}",
                        f"{b.mean():.3f}", f"{roi.mean():.3f}", f"{lum.max():.0f}"])
            if t - last_full >= args.full_frame_every_s:
                (fdir / f"frame_{n:06d}_{t:.3f}.jpg").write_bytes(jpg)
                last_full = t
            n += 1
            if n % 100 == 0:
                fh.flush()
    print(f"thermal_mjpeg_logger: {n} frames, out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
