#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Remove stopwords and nouns/proper nouns (spaCy if available; fallback if not)
- Produce a single diverging bar chart (contrastive lexicon)

Inputs:
- Raw data directory with span_predictions
- A selected technique + trimester pair

Outputs:
- outputs/RQ3_DriftLexicon/<run_id>/distinctive_terms.csv
- outputs/RQ3_DriftLexicon/<run_id>/distinctive_terms.pdf
- outputs/RQ3_DriftLexicon/<run_id>/examples.txt
"""

# =========================
# CONFIG
# =========================
DATA_DIR = "data"

OUTPUT_ROOT = "outputs/RQ3_DriftLexicon"

# Select the drift point you want to illustrate:
TECHNIQUE = "appeal to fear, prejudice"
TRIMESTER_T = "2023-S1"
TRIMESTER_T1 = "2023-S2"
TIME_GRAIN = "semester"

FIGSIZE = (3.35, 2.3)
LABEL_FONTSIZE = 8
VALUE_TICK_FONTSIZE = 8
TERM_FONTSIZE = 9
TITLE_FONTSIZE = 8
SHOW_TITLE = False
SHOW_DIRECTIONAL_HINT = False
X_LABEL = f"{TRIMESTER_T} <-----> {TRIMESTER_T1}"
DIRECTIONAL_LEFT = TRIMESTER_T
DIRECTIONAL_RIGHT = TRIMESTER_T1
DIRECTIONAL_ARROW_Y = -0.18
DIRECTIONAL_TEXT_Y = -0.26
DIRECTIONAL_PAD = 0.01
BOTTOM_MARGIN = 0.28
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
KEEP_POS = {"NOUN", "PROPN","ADJ"}      # optionally add "ADJ"

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
from collections import defaultdict

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


def load_span_texts_for(technique, trimester_label, months_per, period_prefix):
    """
    Load span texts for a given technique and trimester.
    Deduplicate by CommentID (first occurrence wins), consistent with your RQ3 logic.
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

            if cid is None or ts is None or preds is None or not isinstance(preds, list):
                skipped += 1
                continue
            if cid in seen_ids:
                duplicates += 1
                continue

            period = period_from_timestamp(ts, months_per, period_prefix)
            if period != trimester_label:
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
            # Model not installed; fallback
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
def compute_contrastive_terms(docs_t, docs_t1):
    """
    Fit TF-IDF on combined docs; compute mean difference between groups.
    Returns dataframe with columns: term, diff (positive => characteristic of t1; negative => of t)
    """
    y = np.array([0] * len(docs_t) + [1] * len(docs_t1))
    docs = docs_t + docs_t1

    # Use built-in English stopwords regardless (extra safety),
    # even if POS filtering is active (POS filtering already removes most function words).
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

    Xt = X[y == 0]
    Xt1 = X[y == 1]

    mean_t = np.asarray(Xt.mean(axis=0)).ravel()
    mean_t1 = np.asarray(Xt1.mean(axis=0)).ravel()

    diff = mean_t1 - mean_t  # >0 => more characteristic of t1; <0 => more characteristic of t

    df = pd.DataFrame({"term": terms, "diff": diff})
    df = df.sort_values("diff")
    return df


def add_directional_hint(ax, left_label, right_label):
    left_text = ax.text(
        0.0,
        DIRECTIONAL_TEXT_Y,
        left_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=LABEL_FONTSIZE,
        color="black",
    )
    right_text = ax.text(
        1.0,
        DIRECTIONAL_TEXT_Y,
        right_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=LABEL_FONTSIZE,
        color="black",
    )
    # Measure label extents to keep the arrow between them.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_bbox = left_text.get_window_extent(renderer=renderer)
    right_bbox = right_text.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    left_edge = inv.transform((left_bbox.x1, left_bbox.y0))[0]
    right_edge = inv.transform((right_bbox.x0, right_bbox.y0))[0]
    start = min(max(left_edge + DIRECTIONAL_PAD, 0.0), 1.0)
    end = max(min(right_edge - DIRECTIONAL_PAD, 1.0), 0.0)
    if end <= start:
        start, end = 0.1, 0.9
    ax.annotate(
        "",
        xy=(end, DIRECTIONAL_ARROW_Y),
        xytext=(start, DIRECTIONAL_ARROW_Y),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 0.8},
        annotation_clip=False,
    )


def plot_diverging(df_terms, out_pdf, title, x_label, left_label, right_label, color=None):
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
    if SHOW_DIRECTIONAL_HINT:
        ax.set_xlabel("")
    else:
        ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=VALUE_TICK_FONTSIZE)
    fig.tight_layout()
    if SHOW_DIRECTIONAL_HINT:
        fig.subplots_adjust(bottom=BOTTOM_MARGIN)
        add_directional_hint(ax, left_label, right_label)
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

    run_id = f"{re.sub(r'[^A-Za-z0-9]+','_',TECHNIQUE)[:40]}_{TRIMESTER_T}_to_{TRIMESTER_T1}"
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Load raw span texts
    texts_t, stats_t = load_span_texts_for(
        TECHNIQUE, TRIMESTER_T, months_per, period_prefix
    )
    texts_t1, stats_t1 = load_span_texts_for(
        TECHNIQUE, TRIMESTER_T1, months_per, period_prefix
    )

    # Dedupe within each side to avoid repetitive "terrorists" dominating examples
    texts_t = dedupe_texts(texts_t)
    texts_t1 = dedupe_texts(texts_t1)

    # Sampling for speed (optional)
    if len(texts_t) > MAX_SPANS_PER_SIDE:
        texts_t = random.sample(texts_t, MAX_SPANS_PER_SIDE)
    if len(texts_t1) > MAX_SPANS_PER_SIDE:
        texts_t1 = random.sample(texts_t1, MAX_SPANS_PER_SIDE)

    if len(texts_t) < MIN_SPANS_PER_SIDE or len(texts_t1) < MIN_SPANS_PER_SIDE:
        print("Insufficient span texts after dedupe/sampling.")
        print("Counts:", len(texts_t), len(texts_t1))
        return

    # Prepare docs for TF-IDF
    nlp = None
    if USE_POS_FILTER:
        nlp = try_load_spacy()
        if nlp is None:
            print("spaCy POS filter requested, but en_core_web_sm not available. Falling back to stopword-based TF-IDF.")

    if nlp is not None:
        docs_t = pos_filter_to_content_words(texts_t, nlp, KEEP_POS)
        docs_t1 = pos_filter_to_content_words(texts_t1, nlp, KEEP_POS)
    else:
        # Fallback: use raw texts; stopword removal happens in the vectorizer
        docs_t = texts_t
        docs_t1 = texts_t1

    # Compute contrastive terms
    df_terms = compute_contrastive_terms(docs_t, docs_t1)

    # Save full table
    csv_path = os.path.join(out_dir, "distinctive_terms.csv")
    df_terms.to_csv(csv_path, index=False)

    # Plot diverging bars and save PDF
    title = f"{TECHNIQUE} | Contrastive lexicon: {TRIMESTER_T} vs {TRIMESTER_T1}"
    pdf_path = os.path.join(out_dir, "distinctive_terms.pdf")
    plot_diverging(
        df_terms,
        pdf_path,
        title,
        X_LABEL,
        DIRECTIONAL_LEFT,
        DIRECTIONAL_RIGHT,
        color=color_for_label(TECHNIQUE),
    )

    if SAVE_PNG_TOO:
        png_path = os.path.join(out_dir, "distinctive_terms.png")
        plot_diverging(
            df_terms,
            png_path,
            title,
            X_LABEL,
            DIRECTIONAL_LEFT,
            DIRECTIONAL_RIGHT,
            color=color_for_label(TECHNIQUE),
        )

    # Write simple examples file (random samples from each side; deduped)
    # This is intentionally light; you already have centroid/shift-aligned examples from earlier script.
    ex_path = os.path.join(out_dir, "examples.txt")
    with open(ex_path, "w", encoding="utf-8") as f:
        f.write(f"Technique: {TECHNIQUE}\n")
        f.write(f"{grain_label} pair: {TRIMESTER_T} -> {TRIMESTER_T1}\n")
        f.write(f"Counts (deduped): {TRIMESTER_T}={len(texts_t)} | {TRIMESTER_T1}={len(texts_t1)}\n\n")

        f.write(f"=== Random span examples from {TRIMESTER_T} (deduped) ===\n")
        for s in random.sample(texts_t, min(N_EXAMPLES_PER_SIDE, len(texts_t))):
            f.write(f"- {s}\n")
        f.write("\n")

        f.write(f"=== Random span examples from {TRIMESTER_T1} (deduped) ===\n")
        for s in random.sample(texts_t1, min(N_EXAMPLES_PER_SIDE, len(texts_t1))):
            f.write(f"- {s}\n")

    # Print summary
    print("Wrote outputs to:", out_dir)
    print("Stats T:", stats_t)
    print("Stats T1:", stats_t1)
    print("Saved:", csv_path)
    print("Saved:", pdf_path)
    print("Saved:", ex_path)

    # Print top terms each side for quick inspection
    neg = df_terms.head(TOP_TERMS_EACH_SIDE)
    pos = df_terms.tail(TOP_TERMS_EACH_SIDE)
    print(f"\nTop terms characteristic of earlier {grain_label.lower()} (negative diffs):")
    for _, r in neg.iterrows():
        print(f"  {r['term']}\t{r['diff']:.6f}")
    print(f"\nTop terms characteristic of later {grain_label.lower()} (positive diffs):")
    for _, r in pos.iterrows():
        print(f"  {r['term']}\t{r['diff']:.6f}")


if __name__ == "__main__":
    main()
