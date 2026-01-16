#!/usr/bin/env python3
# CONFIG
DATA_DIR = "data"
OUTPUT_DIR = "outputs/RQ0_semester"
OUTPUT_BASENAME = "rq0_video_mean_techniques_trends"
CSV_NAME = "rq0_video_mean_techniques.csv"
PROGRESS_EVERY = 200000
CUTOFF_TIMESTAMP = "2024-12-31T23:59:59Z"
TIME_GRAIN = "semester"
OUTPUT_FORMATS = ("pdf", "png")

FIGSIZE = (3.35, 2.6)
LEFT_MARGIN = 0.20
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.97
BOTTOM_MARGIN = 0.30
LEGEND_NCOL = 2
LEGEND_FONT = 6
LEGEND_Y = -0.30
LABEL_FONTSIZE = 7
TICK_FONTSIZE = 7
LINE_WIDTH = 1.5
MARKER_SIZE = 2.5

CHANNEL_COLORS = {
    "Left": "#1f78b4",
    "Right": "#e31a1c",
}

import os
import json
import gzip
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_utils import (
    TIME_GRAIN_DEFAULT,
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
    video_totals = {}
    video_sums = {}
    period_keys = {}

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
            video_id = obj.get("VideoID")

            if comment_id is None or timestamp is None or span_predictions is None:
                counters["skipped_records"] += 1
                continue
            if channel_leaning not in ("Left", "Right"):
                counters["skipped_records"] += 1
                continue
            if video_id is None:
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

            techniques = extract_techniques(span_predictions)
            technique_count = len(techniques)

            period_label, key = get_period_label_from_month(
                dt.year, dt.month, months_per, period_prefix
            )
            period_keys[period_label] = key
            seen_ids.add(comment_id)

            video_key = (period_label, channel_leaning, video_id)
            video_totals[video_key] = video_totals.get(video_key, 0) + 1
            video_sums[video_key] = video_sums.get(video_key, 0) + technique_count

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ordered_periods = sorted(period_keys.keys(), key=lambda x: period_keys[x])
    channels = ["Left", "Right"]

    mean_sum = {}
    video_counts = {}
    comment_totals = {}
    overall_mean_sum = {"Left": 0.0, "Right": 0.0}
    overall_video_counts = {"Left": 0, "Right": 0}

    for (period_label, channel, video_id), n_comments in video_totals.items():
        sum_techniques = video_sums.get((period_label, channel, video_id), 0)
        mean_video = sum_techniques / n_comments if n_comments > 0 else np.nan
        if np.isnan(mean_video):
            continue
        period_channel = (period_label, channel)
        mean_sum[period_channel] = mean_sum.get(period_channel, 0.0) + mean_video
        video_counts[period_channel] = video_counts.get(period_channel, 0) + 1
        comment_totals[period_channel] = comment_totals.get(period_channel, 0) + n_comments
        overall_mean_sum[channel] = overall_mean_sum.get(channel, 0.0) + mean_video
        overall_video_counts[channel] = overall_video_counts.get(channel, 0) + 1

    rows = []
    for period in ordered_periods:
        for channel in channels:
            key = (period, channel)
            n_videos = video_counts.get(key, 0)
            total_comments = comment_totals.get(key, 0)
            mean_techniques = mean_sum.get(key, 0.0) / n_videos if n_videos > 0 else np.nan
            rows.append({
                "trimester": period,
                "channel_leaning": channel,
                "n_videos": n_videos,
                "n_comments_total": total_comments,
                "mean_techniques_video_avg": mean_techniques,
            })

    result = pd.DataFrame(
        rows,
        columns=[
            "trimester",
            "channel_leaning",
            "n_videos",
            "n_comments_total",
            "mean_techniques_video_avg",
        ],
    )

    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    result.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if ordered_periods:
        for channel in channels:
            channel_df = result[result["channel_leaning"] == channel].set_index("trimester")
            channel_df = channel_df.reindex(ordered_periods)
            ax.plot(
                ordered_periods,
                channel_df["mean_techniques_video_avg"].values,
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=channel,
                color=CHANNEL_COLORS.get(channel, None),
            )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xlabel(grain_label, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(
            "Mean techniques per comment",
            fontsize=LABEL_FONTSIZE,
            labelpad=6,
        )
        short_labels = [shorten_period_label(p) for p in ordered_periods]
        ax.set_xticks(ordered_periods)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        fig.subplots_adjust(
            left=LEFT_MARGIN,
            right=RIGHT_MARGIN,
            top=TOP_MARGIN,
            bottom=BOTTOM_MARGIN,
        )
        legend = ax.legend(
            fontsize=LEGEND_FONT,
            ncol=LEGEND_NCOL,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, LEGEND_Y),
            handlelength=1.4,
            columnspacing=0.8,
            labelspacing=0.3,
        )
    else:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.axis("off")
        legend = None

    for fmt in OUTPUT_FORMATS:
        plot_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.{fmt}")
        if legend is None:
            fig.savefig(plot_path, bbox_inches="tight")
        else:
            fig.savefig(plot_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)

    print(f"Total loaded records: {counters['total_records']}")
    print(f"Total unique comments: {len(seen_ids)}")
    print(f"Total skipped records: {counters['skipped_records']}")
    for channel in channels:
        total_videos = overall_video_counts.get(channel, 0)
        mean_techniques = (
            overall_mean_sum.get(channel, 0.0) / total_videos if total_videos > 0 else np.nan
        )
        print(
            f"Overall mean techniques (video avg) ({channel}): {mean_techniques:.6f}"
        )


if __name__ == "__main__":
    main()
