#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Left vs Right lexical contrast for a selected technique within a period.

Inputs:
- Raw data directory with span_predictions
- A selected technique + period

Outputs:
- outputs/RQ4_Lexicon/<run_id>/distinctive_terms.csv
- outputs/RQ4_Lexicon/<run_id>/distinctive_terms.pdf
- outputs/RQ4_Lexicon/<run_id>/examples.txt
"""

# =========================
# CONFIG
# =========================
DATA_DIR = "data"
OUTPUT_ROOT = "outputs/RQ4_Lexicon"

# Select the period + technique you want to illustrate:
TECHNIQUE = "causal oversimplification"
PERIOD = "2023-S2"
TIME_GRAIN = "semester"

FIGSIZE = (3.35, 2.3)
LABEL_FONTSIZE = 8
VALUE_TICK_FONTSIZE = 8
TERM_FONTSIZE = 9
TITLE_FONTSIZE = 8
SHOW_TITLE = False
X_LABEL = "Left <-----> Right"
BAR_HEIGHT = 0.8

# Data controls
MIN_SPANS_PER_SIDE = 200
MAX_SPANS_PER_SIDE = 8000
RANDOM_SEED = 13

# Lexicon controls
TOP_TERMS_EACH_SIDE = 5          # will show TOP_TERMS_EACH_SIDE per side on the diverging plot
MIN_DF = 10                       # min document frequency in TF-IDF
NGRAM_RANGE = (1, 2)              # (1,1) if you want only unigrams

# POS filtering (recommended). If spaCy model unavailable, script will fallback.
USE_POS_FILTER = True
KEEP_POS = {"NOUN", "PROPN", "ADJ"}

# Output controls
SAVE_PNG_TOO = True

# Example spans (for write-up)
N_EXAMPLES_PER_SIDE = 5

# =========================
# IMPORTS
# =========================
import os
import re
import json
import gzip
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from analysis_utils import (
    TIME_GRAIN_DEFAULT,
    color_for_label,
    period_label_from_timestamp,
    resolve_grain,
)

# =========================
# FILE LOADING
# =========================
def iter_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith((".jsonl", ".json", ".gz")):
                yield os.path.join(dirpath, fn)


def iter_records_from_file(path, parse_error_counter):
    def handle_jsonl(f):
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                parse_error_counter[0] += 1

    def handle_json(f):
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            parse_error_counter[0] += 1
            return
        if isinstance(data, list):
            for obj in data:
                yield obj
        elif isinstance(data, dict):
            yield data
        else:
            parse_error_counter[0] += 1

    if path.endswith((".jsonl", ".jsonl.gz")) or (path.endswith(".gz") and not path.endswith(".json.gz")):
        mode = "jsonl"
    else:
        mode = "json"

    open_func = gzip.open if path.endswith(".gz") else open
    with open_func(path, "rt", encoding="utf-8", errors="replace") as f:
        if mode == "jsonl":
            yield from handle_jsonl(f)
        else:
            yield from handle_json(f)


def period_from_timestamp(ts: str, months_per, period_prefix):
    return period_label_from_timestamp(ts, months_per, period_prefix)


def load_span_texts_for(technique, period_label, leaning, months_per, period_prefix):
    """
    Load span texts for a given technique, period, and leaning.
    Deduplicate by CommentID (first occurrence wins), consistent with RQ4 logic.
    """
    parse_errors = [0]
    skipped = 0
    duplicates = 0
    seen_ids = set()
    texts = []

    for path in iter_files(DATA_DIR):
        for obj in iter_records_from_file(path, parse_errors):
            if not isinstance(obj, dict):
                skipped += 1
                continue
            cid = obj.get("CommentID")
            ts = obj.get("Timestamp")
            preds = obj.get("span_predictions")
            channel = obj.get("ChannelLeaning")

            if cid is None or ts is None or preds is None or not isinstance(preds, list):
                skipped += 1
                continue
            if channel != leaning:
                skipped += 1
                continue
            if cid in seen_ids:
                duplicates += 1
                continue

            period = period_from_timestamp(ts, months_per, period_prefix)
            if period != period_label:
                continue

            seen_ids.add(cid)

            for entry in preds:
                if not isinstance(entry, dict):
                    continue
                if entry.get("technique") != technique:
                    continue
                spans = entry.get("spans")
                if not isinstance(spans, list):
                    continue
                for sp in spans:
                    if not isinstance(sp, dict):
                        continue
                    txt = sp.get("text")
                    if isinstance(txt, str) and txt.strip():
                        texts.append(txt.strip())

    stats = {
        "parse_errors": parse_errors[0],
        "skipped": skipped,
        "duplicates": duplicates,
        "n_texts": len(texts),
    }
    return texts, stats


# =========================
# OPTIONAL POS FILTERING
# =========================
def try_load_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            return None
        return nlp
    except Exception:
        return None


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def pos_filter_to_content_words(texts, nlp, keep_pos):
    """
    Convert each span text into a space-joined string of selected POS tokens.
    """
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        toks = []
        for t in doc:
            if t.is_stop or t.is_punct or t.like_num:
                continue
            if t.pos_ in keep_pos and WORD_RE.fullmatch(t.text):
                toks.append(t.lemma_.lower())
        out.append(" ".join(toks))
    return out


# =========================
# TF-IDF DIFFERENCE + DIVERGING PLOT
# =========================
def compute_contrastive_terms(docs_left, docs_right):
    """
    Fit TF-IDF on combined docs; compute mean difference between groups.
    Returns dataframe with columns: term, diff (positive => characteristic of Right; negative => Left)
    """
    y = np.array([0] * len(docs_left) + [1] * len(docs_right))
    docs = docs_left + docs_right

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"[A-Za-z][A-Za-z']+",
        min_df=MIN_DF,
        max_df=0.90,
        ngram_range=NGRAM_RANGE,
    )

    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    Xl = X[y == 0]
    Xr = X[y == 1]

    mean_l = np.asarray(Xl.mean(axis=0)).ravel()
    mean_r = np.asarray(Xr.mean(axis=0)).ravel()

    diff = mean_r - mean_l  # >0 => more characteristic of Right; <0 => Left

    df = pd.DataFrame({"term": terms, "diff": diff})
    df = df.sort_values("diff")
    return df


def plot_diverging(df_terms, out_pdf, title, x_label, color=None):
    """
    Plot TOP_TERMS_EACH_SIDE most negative and most positive terms.
    """
    neg = df_terms.head(TOP_TERMS_EACH_SIDE).copy()
    pos = df_terms.tail(TOP_TERMS_EACH_SIDE).copy()

    plot_df = pd.concat([neg, pos], axis=0)
    plot_df = plot_df.sort_values("diff")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["diff"], color=color, height=BAR_HEIGHT)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["term"], fontsize=TERM_FONTSIZE)
    ax.axvline(0, color="black", linewidth=1)
    if SHOW_TITLE:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=VALUE_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def dedupe_texts(texts):
    seen = set()
    out = []
    for t in texts:
        norm = re.sub(r"\s+", " ", t.strip().lower())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(t.strip())
    return out


# =========================
# MAIN
# =========================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    grain_setting = TIME_GRAIN_DEFAULT if TIME_GRAIN is None else TIME_GRAIN
    months_per, period_prefix, grain_label = resolve_grain(grain_setting)

    run_id = f"{re.sub(r'[^A-Za-z0-9]+','_',TECHNIQUE)[:40]}_{PERIOD}_LeftRight"
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Load raw span texts
    texts_left, stats_left = load_span_texts_for(
        TECHNIQUE, PERIOD, "Left", months_per, period_prefix
    )
    texts_right, stats_right = load_span_texts_for(
        TECHNIQUE, PERIOD, "Right", months_per, period_prefix
    )

    # Dedupe within each side to avoid repetitive spans dominating examples
    texts_left = dedupe_texts(texts_left)
    texts_right = dedupe_texts(texts_right)

    # Sampling for speed (optional)
    if len(texts_left) > MAX_SPANS_PER_SIDE:
        texts_left = random.sample(texts_left, MAX_SPANS_PER_SIDE)
    if len(texts_right) > MAX_SPANS_PER_SIDE:
        texts_right = random.sample(texts_right, MAX_SPANS_PER_SIDE)

    if len(texts_left) < MIN_SPANS_PER_SIDE or len(texts_right) < MIN_SPANS_PER_SIDE:
        print("Insufficient span texts after dedupe/sampling.")
        print("Counts:", len(texts_left), len(texts_right))
        return

    # Prepare docs for TF-IDF
    nlp = None
    if USE_POS_FILTER:
        nlp = try_load_spacy()
        if nlp is None:
            print("spaCy POS filter requested, but en_core_web_sm not available. Falling back to stopword-based TF-IDF.")

    if nlp is not None:
        docs_left = pos_filter_to_content_words(texts_left, nlp, KEEP_POS)
        docs_right = pos_filter_to_content_words(texts_right, nlp, KEEP_POS)
    else:
        docs_left = texts_left
        docs_right = texts_right

    # Compute contrastive terms
    df_terms = compute_contrastive_terms(docs_left, docs_right)

    # Save full table
    csv_path = os.path.join(out_dir, "distinctive_terms.csv")
    df_terms.to_csv(csv_path, index=False)

    # Plot diverging bars and save PDF
    title = f"{TECHNIQUE} | Contrastive lexicon: {PERIOD} (Left vs Right)"
    pdf_path = os.path.join(out_dir, "distinctive_terms.pdf")
    plot_diverging(
        df_terms,
        pdf_path,
        title,
        X_LABEL,
        color=color_for_label(TECHNIQUE),
    )

    if SAVE_PNG_TOO:
        png_path = os.path.join(out_dir, "distinctive_terms.png")
        plot_diverging(
            df_terms,
            png_path,
            title,
            X_LABEL,
            color=color_for_label(TECHNIQUE),
        )

    # Write examples file (random samples from each side; deduped)
    ex_path = os.path.join(out_dir, "examples.txt")
    with open(ex_path, "w", encoding="utf-8") as f:
        f.write(f"Technique: {TECHNIQUE}\n")
        f.write(f"{grain_label}: {PERIOD}\n")
        f.write(f"Counts (deduped): Left={len(texts_left)} | Right={len(texts_right)}\n\n")

        f.write("=== Random span examples from Left (deduped) ===\n")
        for s in random.sample(texts_left, min(N_EXAMPLES_PER_SIDE, len(texts_left))):
            f.write(f"- {s}\n")
        f.write("\n")

        f.write("=== Random span examples from Right (deduped) ===\n")
        for s in random.sample(texts_right, min(N_EXAMPLES_PER_SIDE, len(texts_right))):
            f.write(f"- {s}\n")

    # Print summary
    print("Wrote outputs to:", out_dir)
    print("Stats Left:", stats_left)
    print("Stats Right:", stats_right)
    print("Saved:", csv_path)
    print("Saved:", pdf_path)
    print("Saved:", ex_path)

    # Print top terms each side for quick inspection
    neg = df_terms.head(TOP_TERMS_EACH_SIDE)
    pos = df_terms.tail(TOP_TERMS_EACH_SIDE)
    print("\nTop terms characteristic of Left (negative diffs):")
    for _, r in neg.iterrows():
        print(f"  {r['term']}\t{r['diff']:.6f}")
    print("\nTop terms characteristic of Right (positive diffs):")
    for _, r in pos.iterrows():
        print(f"  {r['term']}\t{r['diff']:.6f}")


if __name__ == "__main__":
    main()
