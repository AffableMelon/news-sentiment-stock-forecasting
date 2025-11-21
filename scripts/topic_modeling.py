import re
from collections import Counter
import pandas as pd


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def top_words(df: pd.DataFrame, n=20) -> pd.DataFrame:
    all_tokens = []
    for t in df["headline"].fillna("").apply(tokenize):
        all_tokens.extend(t)

    stopwords = {
        "the",
        "and",
        "a",
        "of",
        "in",
        "on",
        "to",
        "for",
        "with",
        "that",
        "is",
        "was",
        "it",
    }
    filtered = [t for t in all_tokens if t not in stopwords and len(t) > 1]

    counts = Counter(filtered).most_common(n)
    return pd.DataFrame(counts, columns=["word", "count"])
