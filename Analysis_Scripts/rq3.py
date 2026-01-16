#!/usr/bin/env python3
# CONFIG
DATA_DIR = "data"
OUTPUT_DIR = "outputs/RQ3_semester"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 8
BATCH_SIZE = 128
PROGRESS_EVERY = 200000
CUTOFF_TIMESTAMP = "2024-12-31T23:59:59Z"
CHECKPOINT_EVERY = 500000
CHECKPOINT_PATH = "outputs/RQ3_semester/rq3_checkpoint.pkl"
RESUME_FROM_CHECKPOINT = True
TIME_GRAIN = "semester"

import os
import json
import gzip
import pickle
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


def save_checkpoint(path, state):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def process_batch(texts, keys, model, sum_vectors, count_spans, counters):
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
        if key in sum_vectors:
            sum_vectors[key] += emb
        else:
            sum_vectors[key] = emb.astype(np.float64)
        count_spans[key] = count_spans.get(key, 0) + 1
    counters["encoded_spans"] += len(texts)
    texts.clear()
    keys.clear()


def main():
    default_counters = {
        "total_records": 0,
        "skipped_records": 0,
        "parse_errors": 0,
        "duplicate_records": 0,
        "total_spans": 0,
        "used_spans": 0,
        "encoded_spans": 0,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        state = load_checkpoint(CHECKPOINT_PATH)
        counters = state.get("counters", {})
        for key, value in default_counters.items():
            counters.setdefault(key, value)
        seen_ids = state.get("seen_ids", set())
        sum_vectors = state.get("sum_vectors", {})
        count_spans = state.get("count_spans", {})
        techniques_seen = state.get("techniques_seen", set())
        trimester_keys = state.get("trimester_keys", {})
        last_checkpoint_record = state.get("last_checkpoint_record", counters["total_records"])
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}", flush=True)
    else:
        counters = default_counters.copy()
        seen_ids = set()
        sum_vectors = {}
        count_spans = {}
        techniques_seen = set()
        trimester_keys = {}
        last_checkpoint_record = 0

    grain_setting = TIME_GRAIN_DEFAULT if TIME_GRAIN is None else TIME_GRAIN
    months_per, period_prefix, grain_label = resolve_grain(grain_setting)
    cutoff_dt = parse_timestamp_utc(CUTOFF_TIMESTAMP)

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
            if CHECKPOINT_EVERY and (counters["total_records"] - last_checkpoint_record) >= CHECKPOINT_EVERY:
                process_batch(
                    batch_texts,
                    batch_keys,
                    model,
                    sum_vectors,
                    count_spans,
                    counters,
                )
                last_checkpoint_record = counters["total_records"]
                save_checkpoint(
                    CHECKPOINT_PATH,
                    {
                        "counters": counters,
                        "seen_ids": seen_ids,
                        "sum_vectors": sum_vectors,
                        "count_spans": count_spans,
                        "techniques_seen": techniques_seen,
                        "trimester_keys": trimester_keys,
                        "last_checkpoint_record": last_checkpoint_record,
                    },
                )
                print(f"Checkpoint saved at record {last_checkpoint_record}", flush=True)
            if not isinstance(obj, dict):
                counters["skipped_records"] += 1
                continue

            comment_id = obj.get("CommentID")
            timestamp = obj.get("Timestamp")
            span_predictions = obj.get("span_predictions")

            if comment_id is None or timestamp is None or span_predictions is None:
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
                techniques_seen.add(technique)
                for span in spans:
                    counters["total_spans"] += 1
                    if not isinstance(span, dict):
                        continue
                    text = span.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    counters["used_spans"] += 1
                    batch_texts.append(text)
                    batch_keys.append((technique, trimester_label))
                    if len(batch_texts) >= BATCH_SIZE:
                        process_batch(
                            batch_texts,
                            batch_keys,
                            model,
                            sum_vectors,
                            count_spans,
                            counters,
                        )

    process_batch(batch_texts, batch_keys, model, sum_vectors, count_spans, counters)
    if CHECKPOINT_EVERY:
        last_checkpoint_record = counters["total_records"]
        save_checkpoint(
            CHECKPOINT_PATH,
            {
                "counters": counters,
                "seen_ids": seen_ids,
                "sum_vectors": sum_vectors,
                "count_spans": count_spans,
                "techniques_seen": techniques_seen,
                "trimester_keys": trimester_keys,
                "last_checkpoint_record": last_checkpoint_record,
            },
        )
        print(f"Final checkpoint saved at record {last_checkpoint_record}", flush=True)

    ordered_trimesters = sorted(trimester_keys.keys(), key=lambda x: trimester_keys[x])
    techniques = sorted(techniques_seen)

    pairs = []
    pair_labels = []
    pair_labels_short = []
    for idx in range(len(ordered_trimesters) - 1):
        t = ordered_trimesters[idx]
        t1 = ordered_trimesters[idx + 1]
        pairs.append((t, t1))
        pair_labels.append(f"{t}->{t1}")
        pair_labels_short.append(f"{shorten_period_label(t)}->{shorten_period_label(t1)}")

    rows = []
    for technique in techniques:
        for t, t1 in pairs:
            key_t = (technique, t)
            key_t1 = (technique, t1)
            if key_t not in sum_vectors or key_t1 not in sum_vectors:
                continue
            n_t = count_spans.get(key_t, 0)
            n_t1 = count_spans.get(key_t1, 0)
            if n_t == 0 or n_t1 == 0:
                continue
            centroid_t = sum_vectors[key_t] / n_t
            centroid_t1 = sum_vectors[key_t1] / n_t1
            dist = cosine_distance(centroid_t, centroid_t1)
            rows.append({
                "technique": technique,
                "trimester_t": t,
                "trimester_t1": t1,
                "n_spans_t": n_t,
                "n_spans_t1": n_t1,
                "cosine_distance": dist,
            })

    result = pd.DataFrame(
        rows,
        columns=[
            "technique",
            "trimester_t",
            "trimester_t1",
            "n_spans_t",
            "n_spans_t1",
            "cosine_distance",
        ],
    )

    csv_path = os.path.join(OUTPUT_DIR, "rq3_drift_centroid_distances.csv")
    result.to_csv(csv_path, index=False)

    if not result.empty:
        avg_dist = result.groupby("technique")["cosine_distance"].mean()
        pair_counts = result.groupby("technique")["cosine_distance"].size()
        avg_dist = avg_dist.sort_values(ascending=False)
        top_techniques = avg_dist.head(TOP_K).index.tolist()
    else:
        avg_dist = pd.Series(dtype=float)
        pair_counts = pd.Series(dtype=int)
        top_techniques = []

    plt.figure(figsize=(10, 5))
    if top_techniques and pair_labels:
        for technique in top_techniques:
            tech_df = result[result["technique"] == technique]
            tech_map = dict(
                zip(
                    tech_df["trimester_t"] + "->" + tech_df["trimester_t1"],
                    tech_df["cosine_distance"],
                )
            )
            y_vals = [tech_map.get(label, np.nan) for label in pair_labels]
            plt.plot(
                pair_labels_short,
                y_vals,
                marker="o",
                linewidth=1.5,
                label=technique,
                color=color_for_label(technique),
            )
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.legend(fontsize=8, ncol=2)
        plt.xlabel(f"{grain_label} Pair")
        plt.ylabel("Cosine Distance")
        plt.title(f"Top {TOP_K} Techniques by Drift (Centroid Distance)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No techniques to plot", ha="center", va="center")
        plt.axis("off")
    plot_path = os.path.join(OUTPUT_DIR, "rq3_topK_drift.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Total parsed records: {counters['total_records']}")
    print(f"Skipped records: {counters['skipped_records']}")
    print(f"Parse errors: {counters['parse_errors']}")
    print(f"Duplicate records: {counters['duplicate_records']}")
    print(f"Unique comments: {len(seen_ids)}")
    print(f"Total spans seen: {counters['total_spans']}")
    print(f"Spans with text: {counters['used_spans']}")
    print("Top techniques by average drift:")
    if not avg_dist.empty:
        for technique in top_techniques:
            avg_val = avg_dist[technique]
            n_pairs = int(pair_counts.get(technique, 0))
            print(f"  {technique}: {avg_val:.6f} (pairs={n_pairs})")
    else:
        print("  none")


if __name__ == "__main__":
    main()
