#!/usr/bin/env python3
# CONFIG
DATA_DIR = "Analysis/data"
OUTPUT_DIR = "outputs/RQ4_semester"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
BATCH_SIZE = 64
PROGRESS_EVERY = 200000
CUTOFF_TIMESTAMP = "2024-12-31T23:59:59Z"
TIME_GRAIN = "semester"

import os
import json
import gzip
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from analysis_utils import (
    TIME_GRAIN_DEFAULT,
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


def cosine_distance(vec_a, vec_b):
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return np.nan
    return 1.0 - (np.dot(vec_a, vec_b) / denom)


def process_batch(
    texts,
    keys,
    model,
    sum_left,
    sum_right,
    count_left,
    count_right,
    counters,
):
    if not texts:
        return
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for emb, key in zip(embeddings, keys):
        technique, trimester, leaning = key
        base_key = (technique, trimester)
        if leaning == "Left":
            if base_key in sum_left:
                sum_left[base_key] += emb
            else:
                sum_left[base_key] = emb.astype(np.float64)
            count_left[base_key] = count_left.get(base_key, 0) + 1
        else:
            if base_key in sum_right:
                sum_right[base_key] += emb
            else:
                sum_right[base_key] = emb.astype(np.float64)
            count_right[base_key] = count_right.get(base_key, 0) + 1
    counters["encoded_spans"] += len(texts)
    texts.clear()
    keys.clear()


def main():
    counters = {
        "total_records": 0,
        "skipped_records": 0,
        "parse_errors": 0,
        "duplicate_records": 0,
        "total_spans": 0,
        "used_spans": 0,
        "encoded_spans": 0,
    }
    grain_setting = TIME_GRAIN_DEFAULT if TIME_GRAIN is None else TIME_GRAIN
    months_per, period_prefix, grain_label = resolve_grain(grain_setting)
    cutoff_dt = parse_timestamp_utc(CUTOFF_TIMESTAMP)
    seen_ids = set()
    sum_left = {}
    sum_right = {}
    count_left = {}
    count_right = {}
    techniques_seen = set()
    trimester_keys = {}

    batch_texts = []
    batch_keys = []

    model = SentenceTransformer(MODEL_NAME)

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
                    f"{counters['duplicate_records']} duplicates, "
                    f"{counters['encoded_spans']} spans encoded",
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
            trimester_keys[trimester_label] = key
            seen_ids.add(comment_id)

            for entry in span_predictions:
                if not isinstance(entry, dict):
                    continue
                technique = entry.get("technique")
                spans = entry.get("spans")
                if technique is None or not isinstance(spans, list):
                    continue
                for span in spans:
                    counters["total_spans"] += 1
                    if not isinstance(span, dict):
                        continue
                    text = span.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    counters["used_spans"] += 1
                    batch_texts.append(text)
                    batch_keys.append((technique, trimester_label, channel_leaning))
                    techniques_seen.add(technique)
                    if len(batch_texts) >= BATCH_SIZE:
                        process_batch(
                            batch_texts,
                            batch_keys,
                            model,
                            sum_left,
                            sum_right,
                            count_left,
                            count_right,
                            counters,
                        )

    process_batch(
        batch_texts,
        batch_keys,
        model,
        sum_left,
        sum_right,
        count_left,
        count_right,
        counters,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ordered_trimesters = sorted(trimester_keys.keys(), key=lambda x: trimester_keys[x])
    techniques = sorted(techniques_seen)

    rows = []
    for trimester in ordered_trimesters:
        for technique in techniques:
            key = (technique, trimester)
            n_left = count_left.get(key, 0)
            n_right = count_right.get(key, 0)
            if n_left > 0 and n_right > 0:
                centroid_left = sum_left[key] / n_left
                centroid_right = sum_right[key] / n_right
                dist = cosine_distance(centroid_left, centroid_right)
            else:
                dist = np.nan
            rows.append({
                "trimester": trimester,
                "technique": technique,
                "n_spans_left": n_left,
                "n_spans_right": n_right,
                "cosine_distance_left_right": dist,
            })

    result = pd.DataFrame(
        rows,
        columns=[
            "trimester",
            "technique",
            "n_spans_left",
            "n_spans_right",
            "cosine_distance_left_right",
        ],
    )

    csv_path = os.path.join(OUTPUT_DIR, "rq4_left_right_centroid_distances.csv")
    result.to_csv(csv_path, index=False)

    if not result.empty:
        avg_dist = result.groupby("technique")["cosine_distance_left_right"].mean()
        valid_counts = result.groupby("technique")["cosine_distance_left_right"].apply(
            lambda s: int(s.notna().sum())
        )
        avg_dist = avg_dist.sort_values(ascending=False)
        top_techniques = avg_dist.head(TOP_K).index.tolist()
    else:
        avg_dist = pd.Series(dtype=float)
        valid_counts = pd.Series(dtype=int)
        top_techniques = []

    plt.figure(figsize=(10, 5))
    if top_techniques and ordered_trimesters:
        for technique in top_techniques:
            tech_df = result[result["technique"] == technique].set_index("trimester")
            tech_df = tech_df.reindex(ordered_trimesters)
            plt.plot(
                ordered_trimesters,
                tech_df["cosine_distance_left_right"].values,
                marker="o",
                linewidth=1.5,
                label=technique,
                color=color_for_label(technique),
            )
        plt.xlabel(grain_label)
        plt.ylabel("Cosine Distance (Left-Right)")
        plt.title(f"Top {TOP_K} Techniques by Left-Right Distance")
        ax = plt.gca()
        short_labels = [shorten_period_label(t) for t in ordered_trimesters]
        ax.set_xticks(ordered_trimesters)
        ax.set_xticklabels(short_labels, rotation=45, ha="right")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No techniques to plot", ha="center", va="center")
        plt.axis("off")
    plot_path = os.path.join(OUTPUT_DIR, "rq4_topK_left_right_distance.pdf")
    plt.savefig(plot_path)
    plt.close()

    print(f"Total loaded records: {counters['total_records']}")
    print(f"Total unique comments: {len(seen_ids)}")
    print(f"Total skipped records: {counters['skipped_records']}")
    print("Top techniques by avg Left-Right distance:")
    if not avg_dist.empty:
        for technique in top_techniques:
            avg_val = avg_dist[technique]
            n_valid = int(valid_counts.get(technique, 0))
            print(f"  {technique}: {avg_val:.6f} (trimesters={n_valid})")
    else:
        print("  none")


if __name__ == "__main__":
    main()
