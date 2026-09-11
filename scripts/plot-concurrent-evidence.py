#!/usr/bin/env python3
"""Plot the frozen streaming receipt without estimating missing observations."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def plot(output):
    record = json.loads((ROOT / "receipts/concurrent-service-study.json").read_text())
    variants = ("leone_batch1", "leone_batch4", "llama_server")
    labels = ("Leone, batch limit 1", "Leone, batch limit 4", "llama.cpp, all slots")
    colors = ("#64748b", "#d95f02", "#2563eb")
    plt.rcParams.update({"svg.hashsalt": "leone-concurrent-v03", "font.size": 11,
                         "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    specifications = (("ttft_ms", "Median first content latency", "milliseconds, lower is better"),
                      ("aggregate_completion_tok_s", "Median aggregate completion throughput", "tokens per second, higher is better"))
    for axis, (metric, title, unit) in zip(axes, specifications):
        values = [record["summary"][variant][metric]["values"]["p50"] for variant in variants]
        axis.barh(labels, values, color=colors, height=0.55)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=12, loc="left", pad=16)
        axis.set_xlabel(unit, fontsize=9, color="#475569")
        axis.set_xlim(0, max(values) * 1.25)
        axis.tick_params(axis="y", length=0)
        axis.xaxis.grid(True, color="#e2e8f0")
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(value + max(values) * 0.025, index, f"{value:.2f}", va="center", fontsize=10)
    figure.suptitle("Frozen concurrent streaming workload", fontsize=16, weight="bold", x=0.02, ha="left")
    figure.savefig(output, metadata={"Date": None, "Description":
        "Generated from receipts/concurrent-service-study.json. See docs/concurrent-service-evidence.md for configuration and limits."})
    plt.close(figure)
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/concurrent-service.svg"
    plot(destination)
    print(destination)
