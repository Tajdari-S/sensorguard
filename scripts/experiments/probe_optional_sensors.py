#!/usr/bin/env python3
"""Non-invasive readiness probe for optional SensorGuard modalities.

The probe never opens a PicoScope or capture device, so it is safe to run while
an experiment is collecting GPU current.  It records presence and software
readiness separately; physical placement/mapping is never inferred from USB or
PCI enumeration alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""


def packet_capture_permission() -> tuple[bool, str]:
    if not hasattr(socket, "AF_PACKET"):
        return False, "AF_PACKET is unavailable on this platform"
    try:
        handle = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
        handle.close()
        return True, "raw packet socket opens"
    except PermissionError:
        return False, "CAP_NET_RAW or an approved capture helper is required"
    except OSError as exc:
        return False, f"raw packet socket failed: {exc}"


def saige_status(url: str) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "sensorguard-readiness/1"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except OSError as exc:
        return False, str(exc)


def matching_lines(text: str, pattern: str) -> list[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [line.strip() for line in text.splitlines() if regex.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--saige-health-url", default="https://gpu.felkru.com/sensors/api/health")
    parser.add_argument("--network-interface", default="")
    parser.add_argument("--network-tap-verified", action="store_true")
    parser.add_argument(
        "--verified-mapping", action="append", default=[], metavar="SENSOR=DESCRIPTION",
        help="record an independently verified physical map; enumeration alone never sets this",
    )
    args = parser.parse_args()

    verified_mapping = {}
    for item in args.verified_mapping:
        if "=" not in item:
            parser.error("--verified-mapping must be SENSOR=DESCRIPTION")
        sensor, description = item.split("=", 1)
        verified_mapping[sensor] = description

    usb = command_output(["lsusb"])
    links = command_output(["ip", "-brief", "link"])
    video = sorted(str(path) for path in Path("/dev").glob("video*"))
    serial = sorted(str(path) for pattern in ("ttyACM*", "ttyUSB*") for path in Path("/dev").glob(pattern))
    picos = matching_lines(usb, r"0ce9:|pico technology|picoscope")
    microphones = matching_lines(usb, r"dodotronic|ultramic|microphone|audio")
    thermal = matching_lines(usb, r"topdon|thermal|infrared|flir|seek")
    sdr = matching_lines(usb, r"nooelec|rtl283|rtl-sdr|software defined radio")
    packet_ok, packet_note = packet_capture_permission()
    saige_ok, saige_note = saige_status(args.saige_health_url)
    network_interface_present = bool(args.network_interface and args.network_interface in links)

    def mapped(sensor: str) -> bool:
        return sensor in verified_mapping

    def with_mapping(sensor: str, evidence: str) -> str:
        description = verified_mapping.get(sensor)
        return f"{evidence}; verified map: {description}" if description else evidence

    rows = [
        {
            "sensor": "gpu_current",
            "purpose": "electrical activity and temporal modulation",
            "hardware_present": bool(picos),
            "software_ready": bool(picos and Path("/home/felkru/picoenv/bin/python").exists()),
            "mapping_verified": mapped("gpu_current"),
            "paper_role": "headline candidate; already validated separately",
            "evidence": with_mapping("gpu_current", "; ".join(picos) or "no PicoScope USB device"),
            "needed": "approved clamp-to-GPU map; do not infer it from enumeration",
        },
        {
            "sensor": "ultrasound",
            "purpose": "coil-whine and switching-frequency temporal structure",
            "hardware_present": bool(microphones or saige_ok),
            "software_ready": bool(microphones and (shutil.which("arecord") or shutil.which("ffmpeg"))) or saige_ok,
            "mapping_verified": mapped("ultrasound"),
            "paper_role": "optional; retain only for incremental held-family gain",
            "evidence": "; ".join(microphones) or f"SAIGE health: {saige_note}",
            "needed": "UltraMic attachment, fixed placement, >=384 kS/s acquisition, channel calibration",
        },
        {
            "sensor": "thermal_camera",
            "purpose": "spatial hotspot, exhaust and throttling dynamics",
            "hardware_present": bool(thermal or video),
            "software_ready": bool(video and shutil.which("ffmpeg")),
            "mapping_verified": mapped("thermal_camera"),
            "paper_role": "optional; slow corroborating modality",
            "evidence": "; ".join(thermal + video) or "no thermal/UVC device",
            "needed": "TOPDON attachment, fixed GPU/exhaust ROI, emissivity and ambient calibration, 25 Hz frames",
        },
        {
            "sensor": "thermistors",
            "purpose": "low-cost inlet, exhaust and backplate temperature dynamics",
            "hardware_present": False,
            "software_ready": bool(picos),
            "mapping_verified": mapped("thermistors"),
            "paper_role": "optional; low-cost thermal alternative",
            "evidence": f"{len(picos)} PicoScope units, but wiring cannot be inferred",
            "needed": "10K 3590B probes, divider/reference circuit, three placement labels, calibration points, >=10 Hz sampling",
        },
        {
            "sensor": "network_tap",
            "purpose": "distributed-training packet timing, direction and size without payload",
            "hardware_present": args.network_tap_verified,
            "software_ready": bool(args.network_tap_verified and network_interface_present and shutil.which("tcpdump") and packet_ok),
            "mapping_verified": bool(args.network_tap_verified and mapped("network_tap")),
            "paper_role": "distributed workloads only; not useful for local fused-update semantics",
            "evidence": with_mapping(
                "network_tap",
                f"interface={args.network_interface or 'not specified'}; "
                f"interface_present={network_interface_present}; tap_verified={args.network_tap_verified}; {packet_note}",
            ),
            "needed": "tap-facing interface name, SPAN/TAP verification, CAP_NET_RAW, header-only capture policy and clock sync",
        },
        {
            "sensor": "rf_sdr",
            "purpose": "independent electromagnetic activity and temporal modulation",
            "hardware_present": bool(sdr),
            "software_ready": bool(sdr and (shutil.which("rtl_sdr") or shutil.which("rtl_power"))),
            "mapping_verified": mapped("rf_sdr"),
            "paper_role": "optional; retain only for incremental held-family gain",
            "evidence": "; ".join(sdr) or "no RTL-SDR USB device",
            "needed": "Nooelec attachment, fixed antenna geometry, spectrum scout, frozen center frequency/gain and >=2.4 MS/s IQ",
        },
        {
            "sensor": "visible_light",
            "purpose": "physical integrity, obstruction and tamper monitoring",
            "hardware_present": bool(video),
            "software_ready": bool(video and shutil.which("ffmpeg")),
            "mapping_verified": mapped("visible_light"),
            "paper_role": "health/tamper only; never a training detector",
            "evidence": "; ".join(video) or "no UVC video device",
            "needed": "camera attachment, fixed field of view, privacy mask, baseline frame and 5 Hz capture",
        },
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps({
        "host": socket.gethostname(),
        "saige_health_url": args.saige_health_url,
        "saige_reachable_without_credentials": saige_ok,
        "saige_status": saige_note,
        "network_links": links.splitlines(),
        "serial_devices": serial,
        "rows": rows,
    }, indent=2))
    print(f"wrote {args.output_csv} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
