#!/usr/bin/env python3
"""Generate a dependency-free SVG roofline plot from parsed JSON."""

import argparse
import html
import json
import math
from pathlib import Path


def map_log(value, low, high, start, end):
    value = min(max(value, low), high)
    return start + (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)) * (end - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--peak-gbps", type=float)
    parser.add_argument("--peaks-json", type=Path,
                        help="Unprofiled measured GEMM/copy benchmark JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = json.loads(args.summary.read_text())["points"]
    if args.peaks_json:
        peaks = json.loads(args.peaks_json.read_text())["results"]
        measured_tflops = max(float(row.get("tflops", 0.0)) for row in peaks)
        measured_gbps = max(float(row.get("minimum_gbps", 0.0)) for row in peaks)
        args.peak_tflops = args.peak_tflops or measured_tflops
        args.peak_gbps = args.peak_gbps or measured_gbps
    if args.peak_tflops is None or args.peak_gbps is None:
        parser.error("provide --peaks-json or both --peak-tflops and --peak-gbps")
    width, height = 900, 560
    left, right, top, bottom = 90, 850, 35, 500
    xmin, xmax, ymin, ymax = 0.01, 10000.0, 0.001, max(args.peak_tflops * 1.5, 1.0)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:system-ui,sans-serif;font-size:12px}.axis{stroke:#333}.grid{stroke:#ddd}.roof{stroke:#d9485f;stroke-width:3;fill:none}.point{fill:#1769aa}</style>']
    for exponent in range(-2, 5):
        xval = 10 ** exponent
        x = map_log(xval, xmin, xmax, left, right)
        lines += [f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                  f'<text x="{x:.1f}" y="{bottom+22}" text-anchor="middle">10^{exponent}</text>']
    for exponent in range(-3, int(math.ceil(math.log10(ymax))) + 1):
        yval = 10 ** exponent
        y = map_log(yval, ymin, ymax, bottom, top)
        lines += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                  f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end">10^{exponent}</text>']
    ridge = []
    for i in range(240):
        ai = xmin * (xmax / xmin) ** (i / 239)
        perf = min(args.peak_tflops, ai * args.peak_gbps / 1000.0)
        ridge.append(f'{map_log(ai,xmin,xmax,left,right):.1f},{map_log(max(perf,ymin),ymin,ymax,bottom,top):.1f}')
    lines.append(f'<polyline class="roof" points="{" ".join(ridge)}"/>')
    for point in points:
        if point.get("operation") == "copy":
            continue
        ai = point.get("arithmetic_intensity_measured") or point["arithmetic_intensity_min"] or xmin
        perf = max(point.get("tflops", 0), ymin)
        x, y = map_log(ai,xmin,xmax,left,right), map_log(perf,ymin,ymax,bottom,top)
        label = html.escape(point["case"])
        lines += [f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="5"/>',
                  f'<text x="{x+7:.1f}" y="{y-7:.1f}">{label}</text>']
    lines += [f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
              f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
              f'<text x="{(left+right)/2}" y="548" text-anchor="middle">Arithmetic intensity (FLOP/DRAM byte)</text>',
              f'<text transform="translate(20 {(top+bottom)/2}) rotate(-90)" text-anchor="middle">Performance (TFLOP/s)</text>',
              '<text x="450" y="20" text-anchor="middle" font-size="17">RTX 3090 measured microbenchmark roofline</text>',
              f'<text x="450" y="38" text-anchor="middle">ceilings: {args.peak_tflops:.2f} TFLOP/s, {args.peak_gbps:.1f} GB/s</text>', '</svg>']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
