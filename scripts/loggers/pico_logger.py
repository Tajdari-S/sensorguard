#!/usr/bin/env python3
"""PicoScope 2000A-series streaming logger (electrical tier).

Modes:
  --list      enumerate connected units and exit
  --dry-run   route the unit's built-in signal generator to a capture and
              verify amplitude/clipping — validates the acquisition path
              before any probe is wired (safety sign-off pending)
  (default)   stream both channels to a binary .npy + sidecar meta, with
              CLOCK_MONOTONIC_RAW anchors per driver callback so alignment
              is measured against the run's power-edge marker.

Voltage range and coupling are conservative defaults; real rail probing is
configured per the approved wiring diagram, never here.
"""

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.ps2000a import ps2000a as ps


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


RANGE_KEY = "PS2000A_2V"
RANGE_MV = 2000


def open_units(max_units=4):
    units = []
    for _ in range(max_units):
        handle = ctypes.c_int16()
        status = ps.ps2000aOpenUnit(ctypes.byref(handle), None)
        if status != 0:
            break
        units.append(handle)
    return units


def close_units(units):
    for h in units:
        ps.ps2000aCloseUnit(h)


def unit_serial(handle) -> str:
    buf = ctypes.create_string_buffer(64)
    required = ctypes.c_int16()
    # PICO_BATCH_AND_SERIAL = 4
    ps.ps2000aGetUnitInfo(handle, buf, 64, ctypes.byref(required), 4)
    return buf.value.decode()


def setup_channels(handle):
    rng = ps.PS2000A_RANGE[RANGE_KEY]
    for ch_name in ("PS2000A_CHANNEL_A", "PS2000A_CHANNEL_B"):
        ch = ps.PS2000A_CHANNEL[ch_name]
        # enabled=1, DC coupling=1, analogue offset 0
        assert_pico_ok(ps.ps2000aSetChannel(handle, ch, 1, 1, rng, 0.0))
    return rng


def stream(handle, seconds: float, sample_interval_us: int, out_prefix: Path, rng):
    n_per_cb = 8192
    buf_a = np.zeros(n_per_cb, dtype=np.int16)
    buf_b = np.zeros(n_per_cb, dtype=np.int16)
    for ch_name, buf in (("PS2000A_CHANNEL_A", buf_a), ("PS2000A_CHANNEL_B", buf_b)):
        ch = ps.PS2000A_CHANNEL[ch_name]
        assert_pico_ok(ps.ps2000aSetDataBuffers(
            handle, ch,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), None,
            n_per_cb, 0, ps.PS2000A_RATIO_MODE["PS2000A_RATIO_MODE_NONE"]))

    interval = ctypes.c_int32(sample_interval_us)
    assert_pico_ok(ps.ps2000aRunStreaming(
        handle, ctypes.byref(interval),
        ps.PS2000A_TIME_UNITS["PS2000A_US"],
        0, 0, 0, 1, ps.PS2000A_RATIO_MODE["PS2000A_RATIO_MODE_NONE"], n_per_cb))

    chunks_a, chunks_b, anchors = [], [], []
    state = {"start": None, "n": None, "overflow": 0}

    def cb(handle_, n_samples, start_index, overflow, trigger_at, triggered, auto_stop, param):
        state["start"], state["n"] = start_index, n_samples
        state["overflow"] |= overflow

    c_cb = ps.StreamingReadyType(cb)
    t_end = raw_now() + seconds
    total = 0
    while raw_now() < t_end:
        state["n"] = None
        ps.ps2000aGetStreamingLatestValues(handle, c_cb, None)
        if state["n"]:
            s, n = state["start"], state["n"]
            anchors.append((raw_now(), total, n))
            chunks_a.append(buf_a[s:s + n].copy())
            chunks_b.append(buf_b[s:s + n].copy())
            total += n
        else:
            time.sleep(0.001)
    ps.ps2000aStop(handle)

    a = np.concatenate(chunks_a) if chunks_a else np.array([], dtype=np.int16)
    b = np.concatenate(chunks_b) if chunks_b else np.array([], dtype=np.int16)
    max_adc = ctypes.c_int16(32767)
    a_mv = np.asarray(adc2mV(a.astype(np.int16), rng, max_adc)) if a.size else a
    b_mv = np.asarray(adc2mV(b.astype(np.int16), rng, max_adc)) if b.size else b
    np.save(f"{out_prefix}_chA.npy", a)
    np.save(f"{out_prefix}_chB.npy", b)
    meta = {
        "serial": unit_serial(handle),
        "sample_interval_us": interval.value,
        "range_mv": RANGE_MV,
        "samples": int(total),
        "overflow_flags": int(state["overflow"]),
        "anchors_raw_s": [(t, off, n) for t, off, n in anchors[:5]] + [(anchors[-1])] if anchors else [],
        "clipping_fraction_a": float((np.abs(a) >= 32000).mean()) if a.size else None,
        "clipping_fraction_b": float((np.abs(b) >= 32000).mean()) if b.size else None,
        "mean_mv_a": float(np.mean(a_mv)) if a.size else None,
        "p2p_mv_a": float(np.ptp(a_mv)) if a.size else None,
    }
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="signal-generator loopback validation (no probes needed: "
                             "AWG output is readable internally on many units; else "
                             "connect AWG->ch A with a BNC lead)")
    parser.add_argument("--duration-s", type=float, default=5)
    parser.add_argument("--sample-interval-us", type=int, default=100, help="100 us = 10 kS/s")
    parser.add_argument("--output-prefix", type=Path, default=Path("data/pico/dryrun"))
    args = parser.parse_args()

    units = open_units()
    if not units:
        print("No PicoScope 2000A units found (or ps2000a driver mismatch)", file=sys.stderr)
        return 2
    print(f"{len(units)} unit(s): {[unit_serial(h) for h in units]}")
    if args.list:
        close_units(units)
        return 0

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    for idx, handle in enumerate(units):
        rng = setup_channels(handle)
        if args.dry_run:
            # 1 kHz, 1.5 V p-p sine from the built-in generator
            assert_pico_ok(ps.ps2000aSetSigGenBuiltIn(
                handle, 0, 1_500_000, 0, 1000.0, 1000.0, 0, 0,
                0, 0, 0, 0, 0, 0, 0))
        meta = stream(handle, args.duration_s, args.sample_interval_us,
                      Path(f"{args.output_prefix}_u{idx}"), rng)
        print(f"unit {idx} ({meta['serial']}): {meta['samples']} samples, "
              f"p2p_A={meta['p2p_mv_a']} mV, clip_A={meta['clipping_fraction_a']}, "
              f"overflow={meta['overflow_flags']}")
        if args.dry_run and (not meta["samples"] or meta["clipping_fraction_a"] is None):
            rc = 1
    close_units(units)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
