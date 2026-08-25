#!/usr/bin/env python3
"""PicoScope 2000-series (legacy ps2000 API) fast-streaming logger.

The verifier's two scopes (2204A/2205A class, USB 0ce9:1007) speak the
old ps2000 API, not ps2000a. Fast streaming per unit, both channels.

Modes:
  --list      enumerate connected units and exit
  --dry-run   drive the built-in signal generator (1 kHz sine) and capture;
              validates the acquisition path before probes are wired
              (requires the AWG output looped to channel A with a BNC lead;
              without the lead, expect a flat trace and use --list only)
  (default)   stream to .npy + meta JSON with CLOCK_MONOTONIC_RAW anchors
              per driver poll, so alignment is measured against the run's
              power-edge marker.

Rail probing is configured per the approved wiring diagram, never here.
"""

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
from picosdk.ps2000 import ps2000 as ps

RANGE_2V = 7          # PS2000 range enum: 7 == +/-2 V
RANGE_MV = 2000
MAX_ADC = 32767


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def open_units(max_units=4):
    units = []
    for _ in range(max_units):
        h = ps.ps2000_open_unit()
        hv = h.value if hasattr(h, "value") else int(h)
        if hv <= 0:
            break
        units.append(hv)
    return units


def unit_serial(handle) -> str:
    buf = ctypes.create_string_buffer(64)
    ps.ps2000_get_unit_info(handle, buf, 64, 4)  # 4 = batch/serial
    return buf.value.decode()


# GetOverviewBuffersMaxMin: buffers = [A max, A min, B max, B min] pointers.
# picosdk's ps2000 wrapper does not export the callback type, so define it.
STREAMING_CB = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),
                                ctypes.c_int16, ctypes.c_uint32, ctypes.c_int16,
                                ctypes.c_int16, ctypes.c_uint32)


def stream_unit(handle, seconds: float, sample_interval_us: int, out_prefix: Path):
    for ch in (0, 1):  # A, B
        assert ps.ps2000_set_channel(handle, ch, 1, 1, RANGE_2V) != 0

    chunks_a, chunks_b, anchors = [], [], []
    state = {"overflow": 0}

    def py_cb(buffers, overflow, trig_at, trig, auto_stop, n):
        if n > 0:
            state["overflow"] |= overflow
            chunks_a.append(np.ctypeslib.as_array(buffers[0], shape=(n,)).copy())
            chunks_b.append(np.ctypeslib.as_array(buffers[2], shape=(n,)).copy())
            anchors.append((raw_now(), int(n)))

    cb = STREAMING_CB(py_cb)

    # PS2000_TIME_UNITS: 0=fs 1=ps 2=ns 3=us 4=ms 5=s
    ok = ps.ps2000_run_streaming_ns(
        handle, sample_interval_us, 3, 60000, 0, 1, 30000)
    assert ok != 0, "run_streaming_ns failed"

    t_end = raw_now() + seconds
    while raw_now() < t_end:
        ps.ps2000_get_streaming_last_values(handle, cb)
        time.sleep(0.01)
    ps.ps2000_stop(handle)

    a = np.concatenate(chunks_a) if chunks_a else np.array([], dtype=np.int16)
    b = np.concatenate(chunks_b) if chunks_b else np.array([], dtype=np.int16)
    np.save(f"{out_prefix}_chA.npy", a)
    np.save(f"{out_prefix}_chB.npy", b)
    to_mv = lambda x: x.astype(np.float64) * RANGE_MV / MAX_ADC
    meta = {
        "serial": unit_serial(handle),
        "api": "ps2000-fast-streaming",
        "sample_interval_us": sample_interval_us,
        "range_mv": RANGE_MV,
        "samples": int(a.size),
        "polls": len(anchors),
        "overflow_flags": int(state["overflow"]),
        "first_anchor_raw_s": anchors[0][0] if anchors else None,
        "last_anchor_raw_s": anchors[-1][0] if anchors else None,
        "clipping_fraction_a": float((np.abs(a) >= MAX_ADC - 700).mean()) if a.size else None,
        "mean_mv_a": float(np.mean(to_mv(a))) if a.size else None,
        "p2p_mv_a": float(np.ptp(to_mv(a))) if a.size else None,
    }
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--duration-s", type=float, default=5)
    parser.add_argument("--sample-interval-us", type=int, default=100, help="100 us = 10 kS/s")
    parser.add_argument("--output-prefix", type=Path, default=Path("data/pico/dryrun"))
    args = parser.parse_args()

    units = open_units()
    if not units:
        print("No ps2000-series units found", file=sys.stderr)
        return 2
    print(f"{len(units)} unit(s): {[unit_serial(h) for h in units]}")
    if args.list:
        for h in units:
            ps.ps2000_close_unit(h)
        return 0

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    for idx, handle in enumerate(units):
        if args.dry_run:
            # 1 kHz, 1.5 V p-p sine (offset 0): args are offset_uV, pkToPk_uV,
            # wavetype (0=sine), start/stop freq, increment, dwell, sweep, sweeps
            ps.ps2000_set_sig_gen_built_in(
                handle, 0, 1_500_000, 0, 1000.0, 1000.0, 0.0, 0.0, 0, 0)
        meta = stream_unit(handle, args.duration_s, args.sample_interval_us,
                           Path(f"{args.output_prefix}_u{idx}"))
        p2p = "n/a" if meta["p2p_mv_a"] is None else f"{meta['p2p_mv_a']:.1f}"
        print(f"unit {idx} ({meta['serial']}): {meta['samples']} samples "
              f"in {meta['polls']} polls, p2p_A={p2p} mV, "
              f"clip_A={meta['clipping_fraction_a']}, overflow={meta['overflow_flags']}")
        if not meta["samples"]:
            rc = 1
        ps.ps2000_close_unit(handle)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
