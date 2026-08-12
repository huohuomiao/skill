#!/usr/bin/env python3
"""Summarize occupancy, resources and throughput from raw Nsight Compute CSV."""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


ALIASES = {
    "duration": ("gpu__time_duration.sum",),
    "registers": ("launch__registers_per_thread", "launch__registers_per_thread_allocated"),
    "shared": ("launch__shared_mem_per_block_allocated", "launch__shared_mem_per_block"),
    "shared_static": ("launch__shared_mem_per_block_static",),
    "shared_dynamic": ("launch__shared_mem_per_block_dynamic",),
    "block": ("launch__block_size",),
    "grid": ("launch__grid_size",),
    "waves": ("launch__waves_per_multiprocessor",),
    "achieved": ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    "theoretical": (
        "sm__maximum_warps_per_active_cycle_pct",
        "sm__maximum_warps_avg_per_active_cycle.pct_of_peak_sustained_active",
    ),
    "sm": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
    "combined": ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",),
    "dram": (
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "read": ("dram__bytes_read.sum",),
    "write": ("dram__bytes_write.sum",),
}


def norm(text):
    return re.sub(r"\s+", " ", text.strip().strip('"')).casefold()


def read_csv(path):
    """Ignore NCU chatter and accept repeated headers from multi-process reports."""
    records, header = [], None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            names = [norm(x) for x in row]
            if "metric name" in names and "metric value" in names:
                header = names
            elif header and len(row) == len(header):
                item = dict(zip(header, (x.strip() for x in row)))
                if item.get("metric name"):
                    records.append(item)
    if not records:
        raise ValueError("no metric rows; export with `ncu --import REPORT --page raw --csv`")
    return records


def number(text):
    text = text.strip().replace("\u202f", "").replace("\xa0", "").replace(",", "")
    if not text or text.casefold() in {"n/a", "nan", "not available"}:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", text, re.I)
    return float(match.group()) if match else None


def convert(value, unit, kind):
    if value is None:
        return None
    unit = unit.casefold().replace(" ", "").replace("bytes", "byte")
    if kind == "bytes":
        return value * {
            "byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9,
            "kib": 1024, "mib": 1024**2, "gib": 1024**3,
        }.get(unit, 1)
    return value * {"nsecond": 1, "usecond": 1e3, "msecond": 1e6, "second": 1e9}.get(unit, 1)


def pick(rows, key):
    lookup = {row.get("metric name"): row for row in rows}
    for name in ALIASES[key]:
        row = lookup.get(name)
        if row:
            value = number(row.get("metric value", ""))
            if value is not None:
                return value, row.get("metric unit", "")
    return None, ""


def group(records):
    groups = defaultdict(list)
    for row in records:
        key = (
            row.get("id", "?"), row.get("process id", "?"),
            row.get("kernel name", row.get("kernel", "unknown")),
            row.get("context", "?"), row.get("stream", "?"),
        )
        groups[key].append(row)
    return groups


def analyze(key, rows):
    values = {name: pick(rows, name) for name in ALIASES}
    registers = values["registers"][0]
    shared = convert(*values["shared"], "bytes")
    static = convert(*values["shared_static"], "bytes")
    dynamic = convert(*values["shared_dynamic"], "bytes")
    if shared is None and (static is not None or dynamic is not None):
        shared = (static or 0) + (dynamic or 0)
    achieved, theoretical = values["achieved"][0], values["theoretical"][0]
    sm, dram, waves = values["sm"][0], values["dram"][0], values["waves"][0]
    advice = []

    def add(level, evidence, action):
        advice.append({"level": level, "evidence": evidence, "action": action})

    if registers is not None and registers >= 200:
        add("high", f"{registers:.0f} registers/thread is near the sm_86 limit 255", "reduce live ranges, tile, unrolling or stages; check spills")
    elif registers is not None and registers >= 128:
        add("medium", f"{registers:.0f} registers/thread may restrict residency", "compare smaller tile/stages and inspect spills")
    if shared is not None and shared > 99 * 1024:
        add("high", f"{shared / 1024:.1f} KiB shared/block exceeds sm_86 limit 99 KiB", "reduce tile or stages")
    elif shared is not None and shared > 48 * 1024:
        add("medium", f"{shared / 1024:.1f} KiB shared/block needs opt-in and may allow one block/SM", "benchmark a smaller tile/stage count")
    if theoretical is not None and theoretical < 50:
        add("medium", f"theoretical occupancy is {theoretical:.1f}%", "read NCU occupancy limiters before changing warps or tile")
    if achieved is not None and theoretical is not None and theoretical - achieved >= 20:
        add("medium", f"achieved {achieved:.1f}% trails theoretical {theoretical:.1f}%", "inspect stalls, divergence and tail imbalance")
    elif achieved is not None and achieved < 35:
        add("medium", f"achieved occupancy is {achieved:.1f}%", "check latency hiding; occupancy alone is not the goal")
    if waves is not None and waves < 1:
        add("medium", f"only {waves:.2f} waves/SM", "inspect grid size; preserve grid-stride coverage in persistent kernels")
    elif waves is not None and abs(waves - round(waves)) >= .15:
        add("low", f"{waves:.2f} waves/SM leaves a partial tail wave", "compare tile/grid choices")
    if dram is not None and dram >= 75 and (sm is None or sm < 65):
        add("info", f"DRAM {dram:.1f}% is high relative to SM", "prioritize coalescing, reuse and bytes moved")
    elif sm is not None and sm >= 75 and (dram is None or dram < 65):
        add("info", f"SM {sm:.1f}% is high relative to DRAM", "inspect instruction mix and Tensor Core eligibility")
    elif sm is not None and dram is not None and max(sm, dram) < 40:
        add("medium", f"SM {sm:.1f}% and DRAM {dram:.1f}% are both low", "check small launch, serialization, stalls or replay artifacts")
    if not advice:
        add("info", "no threshold bottleneck identified", "compare with a correct baseline and inspect the full report")

    return {
        "launch": {"id": key[0], "process_id": key[1], "context": key[3], "stream": key[4]},
        "kernel_name": key[2],
        "metrics": {
            "duration_ns": convert(*values["duration"], "time"),
            "registers_per_thread": registers,
            "shared_memory_per_block_bytes": shared,
            "shared_memory_static_bytes": static,
            "shared_memory_dynamic_bytes": dynamic,
            "block_threads": values["block"][0], "grid_blocks": values["grid"][0],
            "waves_per_sm": waves, "achieved_occupancy_percent": achieved,
            "theoretical_occupancy_percent": theoretical,
            "sm_throughput_percent": sm, "combined_throughput_percent": values["combined"][0],
            "dram_throughput_percent": dram,
            "dram_read_bytes": convert(*values["read"], "bytes"),
            "dram_write_bytes": convert(*values["write"], "bytes"),
        },
        "suggestions": advice,
    }


def fmt(value, suffix="", digits=1):
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}{suffix}"


def show(kernels, source):
    print(f"Nsight Compute summary: {source}\nKernel launches parsed: {len(kernels)}")
    for index, kernel in enumerate(kernels, 1):
        m = kernel["metrics"]
        shared, duration = m["shared_memory_per_block_bytes"], m["duration_ns"]
        print(f"\n[{index}] {kernel['kernel_name']} (launch ID {kernel['launch']['id']})")
        print("  occupancy: achieved={} theoretical={} waves/SM={}".format(
            fmt(m["achieved_occupancy_percent"], "%"), fmt(m["theoretical_occupancy_percent"], "%"), fmt(m["waves_per_sm"], digits=2)))
        print("  resources: registers/thread={} shared/block={} block={} grid={}".format(
            fmt(m["registers_per_thread"], digits=0), "n/a" if shared is None else f"{shared/1024:.1f} KiB",
            fmt(m["block_threads"], digits=0), fmt(m["grid_blocks"], digits=0)))
        print("  throughput: SM={} DRAM={} combined={} duration={}".format(
            fmt(m["sm_throughput_percent"], "%"), fmt(m["dram_throughput_percent"], "%"),
            fmt(m["combined_throughput_percent"], "%"), "n/a" if duration is None else f"{duration/1000:.2f} us"))
        for item in kernel["suggestions"]:
            print(f"  - [{item['level']}] {item['evidence']}; {item['action']}")
    print("\nSuggestions are triage hints, not proof; inspect the .ncu-rep and benchmark.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not args.csv_path.is_file():
        parser.error(f"CSV does not exist: {args.csv_path}")
    try:
        kernels = [analyze(key, rows) for key, rows in sorted(group(read_csv(args.csv_path)).items())]
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    result = {"schema_version": 1, "source_csv": str(args.csv_path.resolve()),
              "kernel_launch_count": len(kernels), "kernels": kernels}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "", end="" if args.json else "")
    if not args.json:
        show(kernels, args.csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
