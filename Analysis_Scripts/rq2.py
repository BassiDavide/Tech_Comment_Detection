#!/usr/bin/env python3
# CONFIG
DATA_DIR = "data"
OUTPUT_DIR = "outputs/RQ2"
TOP_K = 5
GAP_DIRECTION = "Right-Left"
PROGRESS_EVERY = 200000
CUTOFF_TIMESTAMP = "2024-12-31T23:59:59Z"
TIME_GRAIN = "semester"
ALWAYS_INCLUDE = {"appeal to hypocrisy (to quoque)"}
FIGSIZE = (3.35, 2.6)
OUTPUT_FORMATS = ("pdf",)

import os
import json
import gzip
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import (
    TIME_GRAIN_DEFAULT,
    canonical_label,
    color_for_label,
    get_period_label_from_month,
    resolve_grain,
    shorten_period_label,
)

def iter_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith((".jsonl", ".json", ".gz")):
                yield os.path.join(dirpath, filename)


def iter_records_from_file(path, counters):
    def handle_jsonl(file_obj):
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counters["parse_errors"] += 1
                continue
            yield obj

    def handle_json(file_obj):
        try:
            data = json.load(file_obj)
        except json.JSONDecodeError:
            counters["parse_errors"] += 1
            return
        if isinstance(data, list):
            for obj in data:
                yield obj
        elif isinstance(data, dict):
            yield data
        else:
            counters["parse_errors"] += 1

    if path.endswith((".jsonl", ".jsonl.gz")):
        mode = "jsonl"
    elif path.endswith((".json", ".json.gz")):
        mode = "json"
    elif path.endswith(".gz"):
        mode = "jsonl"
    else:
        return

    open_func = gzip.open if path.endswith(".gz") else open
    with open_func(path, "rt", encoding="utf-8", errors="replace") as f:
        if mode == "jsonl":
            yield from handle_jsonl(f)
        else:
            yield from handle_json(f)


def extract_techniques(span_predictions):
    techniques = set()
    if not isinstance(span_predictions, list):
        return techniques
    for entry in span_predictions:
        if not isinstance(entry, dict):
            continue
        technique = entry.get("technique")
        spans = entry.get("spans")
        if technique is None:
            continue
        if isinstance(spans, list) and len(spans) > 0:
            techniques.add(technique)
    return techniques


def parse_timestamp_utc(ts):
    if not isinstance(ts, str) or len(ts) < 19:
        return None
    try:
        year = int(ts[0:4])
        month = int(ts[5:7])
        day = int(ts[8:10])
        hour = int(ts[11:13])
        minute = int(ts[14:16])
        second = int(ts[17:19])
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except (ValueError, TypeError, IndexError):
        return None


def compute_gap(prop_left, prop_right):
    if np.isnan(prop_left) or np.isnan(prop_right):
        return np.nan
    if GAP_DIRECTION == "Left-Right":
        return prop_left - prop_right
    return prop_right - prop_left


def main():
    counters = {
        "total_records": 0,
        "skipped_records": 0,
        "parse_errors": 0,
        "duplicate_records": 0,
    }
    grain_setting = TIME_GRAIN_DEFAULT if TIME_GRAIN is None else TIME_GRAIN
    months_per, period_prefix, grain_label = resolve_grain(grain_setting)
    cutoff_dt = parse_timestamp_utc(CUTOFF_TIMESTAMP)
    seen_ids = set()
    n_total_left = {}
    n_total_right = {}
    n_with_left = {}
    n_with_right = {}
    techniques_seen = set()
    trimester_keys = {}

    for path in iter_files(DATA_DIR):
        print(f"Processing file: {path}", flush=True)
        for obj in iter_records_from_file(path, counters):
            counters["total_records"] += 1
            if PROGRESS_EVERY and counters["total_records"] % PROGRESS_EVERY == 0:
                print(
                    "Progress: "
                    f"{counters['total_records']} parsed, "
                    f"{counters['skipped_records']} skipped, "
                    f"{counters['parse_errors']} parse errors, "
                    f"{counters['duplicate_records']} duplicates",
                    flush=True,
                )
            if not isinstance(obj, dict):
                counters["skipped_records"] += 1
                continue

            comment_id = obj.get("CommentID")
            timestamp = obj.get("Timestamp")
            span_predictions = obj.get("span_predictions")
            channel_leaning = obj.get("ChannelLeaning")

            if comment_id is None or timestamp is None or span_predictions is None:
                counters["skipped_records"] += 1
                continue
            if channel_leaning not in ("Left", "Right"):
                counters["skipped_records"] += 1
                continue
            if not isinstance(span_predictions, list):
                counters["skipped_records"] += 1
                continue
            if comment_id in seen_ids:
                counters["duplicate_records"] += 1
                continue

            dt = parse_timestamp_utc(timestamp)
            if dt is None:
                counters["skipped_records"] += 1
                continue
            if cutoff_dt is not None and dt > cutoff_dt:
                counters["skipped_records"] += 1
                continue

            trimester_label, key = get_period_label_from_month(
                dt.year, dt.month, months_per, period_prefix
            )
            techniques = extract_techniques(span_predictions)

            seen_ids.add(comment_id)
            trimester_keys[trimester_label] = key
            if channel_leaning == "Left":
                n_total_left[trimester_label] = n_total_left.get(trimester_label, 0) + 1
                for technique in techniques:
                    n_with_key = (trimester_label, technique)
                    n_with_left[n_with_key] = n_with_left.get(n_with_key, 0) + 1
            else:
                n_total_right[trimester_label] = n_total_right.get(trimester_label, 0) + 1
                for technique in techniques:
                    n_with_key = (trimester_label, technique)
                    n_with_right[n_with_key] = n_with_right.get(n_with_key, 0) + 1

            techniques_seen.update(techniques)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ordered_trimesters = sorted(trimester_keys.keys(), key=lambda x: trimester_keys[x])
    techniques = sorted(techniques_seen)

    rows = []
    for trimester in ordered_trimesters:
        total_left = n_total_left.get(trimester, 0)
        total_right = n_total_right.get(trimester, 0)
        for technique in techniques:
            nwl = n_with_left.get((trimester, technique), 0)
            nwr = n_with_right.get((trimester, technique), 0)
            prop_left = nwl / total_left if total_left > 0 else np.nan
            prop_right = nwr / total_right if total_right > 0 else np.nan
            gap = compute_gap(prop_left, prop_right)
            rows.append({
                "trimester": trimester,
                "technique": technique,
                "n_total_left": total_left,
                "n_with_left": nwl,
                "prop_left": prop_left,
                "n_total_right": total_right,
                "n_with_right": nwr,
                "prop_right": prop_right,
                "gap": gap,
            })

    result = pd.DataFrame(
        rows,
        columns=[
            "trimester",
            "technique",
            "n_total_left",
            "n_with_left",
            "prop_left",
            "n_total_right",
            "n_with_right",
            "prop_right",
            "gap",
        ],
    )

    csv_path = os.path.join(OUTPUT_DIR, "rq2_trimester_gap.csv")
    result.to_csv(csv_path, index=False)

    if not result.empty:
        gap_by_tech = result.groupby("technique")["gap"].apply(
            lambda s: s.loc[s.abs().idxmax()] if s.dropna().size > 0 else np.nan
        )
        gap_by_tech = gap_by_tech.dropna()
        gap_by_tech = gap_by_tech.reindex(
            gap_by_tech.abs().sort_values(ascending=False).index
        )
        top_techniques = gap_by_tech.head(TOP_K).index.tolist()
        always_include = {canonical_label(item) for item in ALWAYS_INCLUDE}
        for technique in techniques_seen:
            if canonical_label(technique) in always_include and technique not in top_techniques:
                top_techniques.append(technique)
    else:
        gap_by_tech = pd.Series(dtype=float)
        top_techniques = []

    plt.figure(figsize=FIGSIZE)
    if top_techniques and ordered_trimesters:
        other_techniques = [t for t in techniques if t not in top_techniques]
        if other_techniques:
            other_gaps = result[result["technique"].isin(other_techniques)]
            other_median = []
            other_low = []
            other_high = []
            for trimester in ordered_trimesters:
                gaps = (
                    other_gaps[other_gaps["trimester"] == trimester]["gap"]
                    .dropna()
                    .values
                )
                if gaps.size == 0:
                    other_median.append(np.nan)
                    other_low.append(np.nan)
                    other_high.append(np.nan)
                else:
                    other_median.append(float(np.nanmedian(gaps)))
                    other_low.append(float(np.nanquantile(gaps, 0.25)))
                    other_high.append(float(np.nanquantile(gaps, 0.75)))
            plt.fill_between(
                ordered_trimesters,
                other_low,
                other_high,
                color="lightgray",
                alpha=0.35,
                label="Other techniques (IQR)",
                zorder=1,
            )
            plt.plot(
                ordered_trimesters,
                other_median,
                color="gray",
                linewidth=1.2,
                label="Other techniques (median)",
                zorder=2,
            )
        for technique in top_techniques:
            tech_series = result[result["technique"] == technique].set_index("trimester")
            tech_series = tech_series.reindex(ordered_trimesters)
            plt.plot(
                ordered_trimesters,
                tech_series["gap"].values,
                linewidth=1.5,
                label=technique,
                color=color_for_label(technique),
                zorder=3,
            )
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.xlabel(grain_label)
        plt.ylabel("Gap", labelpad=6)
        if GAP_DIRECTION == "Left-Right":
            top_label = "Left"
            bottom_label = "Right"
        else:
            top_label = "Right"
            bottom_label = "Left"
        ax = plt.gca()
        ax.text(
            0.01,
            0.98,
            top_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="black",
        )
        ax.text(
            0.01,
            0.02,
            bottom_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="black",
        )
        ax = plt.gca()
        short_labels = [shorten_period_label(t) for t in ordered_trimesters]
        ax.set_xticks(ordered_trimesters)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        plt.subplots_adjust(bottom=0.34, left=0.18, right=0.98, top=0.97)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            fontsize=6,
            ncol=2,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.34),
            handlelength=1.4,
            columnspacing=0.8,
            labelspacing=0.3,
        )
    else:
        plt.text(0.5, 0.5, "No techniques to plot", ha="center", va="center")
        plt.axis("off")
    for fmt in OUTPUT_FORMATS:
        plot_path = os.path.join(OUTPUT_DIR, f"rq2_topK_gap_trends.{fmt}")
        if fmt.lower() == "png":
            plt.savefig(plot_path, dpi=300)
        else:
            plt.savefig(plot_path)
    plt.close()

    print(f"Total parsed records: {counters['total_records']}")
    print(f"Skipped records: {counters['skipped_records']}")
    print(f"Parse errors: {counters['parse_errors']}")
    print(f"Duplicate records: {counters['duplicate_records']}")
    print(f"Unique comments: {len(seen_ids)}")
    print(f"Top 10 technique gaps overall ({GAP_DIRECTION}):")
    if not gap_by_tech.empty:
        for technique, gap in gap_by_tech.head(10).items():
            print(f"  {technique}: {gap:.6f}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
