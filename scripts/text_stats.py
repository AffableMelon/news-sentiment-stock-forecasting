import pandas as pd


def compute_headline_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds headline length columns to the DataFrame.
    """
    df["headline_len_chars"] = df["headline"].fillna("").str.len()
    df["headline_len_words"] = df["headline"].fillna("").apply(lambda t: len(t.split()))
    return df


def basic_headline_stats(df: pd.DataFrame):
    """
    Returns descriptive statistics for headline lengths.
    """
    return df[["headline_len_chars", "headline_len_words"]].describe()
