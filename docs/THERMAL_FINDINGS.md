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

## Raw radiometric access CONFIRMED, but camera aim is the blocker (2026-09-06, later)

With temporary `admin` sudo on the Pi, `thermal-web.service` was stopped and
`thermal_logger.py` captured the raw 16-bit plane directly. Access works and
the raw frames carry real spatial structure (per-pixel std ~4684, not flat).

**However, a load test shows the camera as-aimed does not see GPU heat.**
Raw thermal was captured across an idle -> 5-GPU-gemm (5x350 W, 180 s) ->
transition on node2:

- idle roi_mean 32887.4 vs load(late) roi_mean 32887.3 (delta -0.2 raw units).
- Pixelwise diff of a pre-load frame vs an end-of-load frame: mean +4.9, max
  +31, min -22 raw units; **zero pixels changed by >500** over the whole scene.

So the flatness is NOT a stream artifact and NOT fixed by raw access: 3 minutes
of maximal GPU load produced no thermal change anywhere in frame. The most
likely causes, both needing physical/verification work (not access):

1. **Camera aim/placement** — it is pointed at a surface that does not reflect
   GPU load (e.g. the closed node2 chassis exterior; GPU heat exits via exhaust
   fans, not the case wall). It must view a GPU heatsink/backplate or the
   exhaust, ideally with an unobstructed thermal path.
2. **Radiometric decode** — the raw16 values (mean ~32166; a formula like
   raw/64-273 gives implausible temperatures) suggest the exact TC001 byte
   layout/plane may differ from `radiometric_stats()`'s assumption. Even so, a
   monotonic encoding would still shift under real heating, and it did not — so
   aim is the leading suspect. The correct TC001 decode should be confirmed
   against a known hot/cold reference before trusting absolute values.

**Net:** raw thermal is now ACCESSIBLE, but the thermal channel is not yet
USABLE for GPU-workload sensing. Next step is physical: reposition the camera
onto a GPU heat-exposed surface and repeat the idle->load test.

## Committed artifacts

- `data/runs/*thermalpair*` — 8 valid NVML+DCGM runs on node2 (these are
  fine as telemetry corpus; only their *thermal* pairing is unusable).
- `data/thermal/20260906_thermalpair/thermal_series.csv` + sample frames —
  the MJPEG series, kept as evidence of the negative result. **Do not use
  as a thermal feature source.**
