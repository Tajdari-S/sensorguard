# Thermal camera pairing — findings (2026-09-06)

Camera: TOPDON TC001-A, on the Raspberry Pi (`pi3bplus`, 100.105.210.5),
`/dev/video0`, 256x192 IR @ 25 fps. Pointed at node2.

## Access

- The camera works and is reachable via the Pi. Confirmed by decoding live
  frames.
- It is held exclusively by a **root-owned `thermal-web.service`**
  (`/home/pi/thermal_web.py`) serving an MJPEG stream on port 8090.
- We have `admin` SSH on the Pi but **no sudo**, so we cannot stop that
  service to get exclusive `/dev/video0` access for raw radiometric capture.

## The MJPEG stream is NOT usable as a thermal measurement

`thermal_web.py` auto-normalizes every frame before colormapping:

```python
tmin, tmax = float(tempC.min()), float(tempC.max())
norm = (tempC - tmin) / max(tmax - tmin, 1e-3) * 255
img  = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
```

Each frame is rescaled to its own min/max, so the stream carries only
*relative within-frame* temperature; absolute heating is erased. A paired
collection on 2026-09-06 (5-GPU gemm sweep + ResNet train/infer + idle on
node2, ~24 min thermal) confirmed this empirically:

- Run-averaged centre-ROI luminance was flat across all loads
  (gemm gpu0-4, training, inference all ~60; idle actually highest at 61.4).
- Within-run rise over a 180 s ResNet training run: +0.99 luminance units
  (out of 255); gemm +0.57. Both are noise.

No GPU could be identified as "in frame" from the MJPEG signal, and load
produced no detectable response.

## What is required for a usable thermal channel

1. **Raw radiometric capture** (`scripts/loggers/thermal_logger.py`, already
   written): reads the raw 16-bit temperature plane BEFORE normalization.
   Requires exclusive `/dev/video0`, i.e. **stop `thermal-web.service`**,
   which needs **sudo on the Pi** (technician).
2. **Longer runs**: GPU case/heatsink temperature (what the camera sees)
   lags load by tens of seconds due to thermal mass; 2-3 min runs are
   marginal. Use >=5-10 min runs for a clear thermal signature.

## Committed artifacts

- `data/runs/*thermalpair*` — 8 valid NVML+DCGM runs on node2 (these are
  fine as telemetry corpus; only their *thermal* pairing is unusable).
- `data/thermal/20260906_thermalpair/thermal_series.csv` + sample frames —
  the MJPEG series, kept as evidence of the negative result. **Do not use
  as a thermal feature source.**
