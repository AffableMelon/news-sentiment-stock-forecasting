from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Union, List
from collections import Counter
import re

import pandas as pd


class EdaStatistics:
    def __init__(
        self,
        stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
    ):
        """
        Parameters:
        - stopwords: base set of stopwords used by top_words (can be extended per-call)
        - min_token_len: minimum token length to keep in top_words
        """
        default_stopwords: Set[str] = {
            "the",
            "and",
            "a",
            "of",
            "in",
            "on",
            "to",
            "for",
            "when",
            "with",
            "that",
            "is",
            "was",
            "it",
        }
        self.stopwords: Set[str] = (
            set(stopwords) if stopwords is not None else default_stopwords
        )
        self.min_token_len = int(min_token_len)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Lowercases, removes non-alphanumeric chars (except spaces), splits on whitespace.
        Returns a list of tokens; non-string inputs yield [].
        """
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if t]

    def top_words(
        self,
        df: pd.DataFrame,
        n: int = 20,
        text_col: str = "headline",
        extra_stopwords: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Finds the top-N most frequent tokens in text_col, excluding stopwords and short tokens.

        - n: number of words to return
        - text_col: column containing text (default 'headline')
        - extra_stopwords: additional stopwords to exclude for this call

        Returns: DataFrame with columns ['word', 'count']
        """
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found. Available columns: {
                    df.columns.tolist()
                }"
            )

        stop = set(self.stopwords)
        if extra_stopwords:
            stop |= set(extra_stopwords)

        all_tokens: List[str] = []
        for tokens in df[text_col].fillna("").map(self.tokenize):
            all_tokens.extend(tokens)

        filtered = [
            t for t in all_tokens if t not in stop and len(t) >= self.min_token_len
        ]
        counts = Counter(filtered).most_common(n)
        return pd.DataFrame(counts, columns=["word", "count"])

    def articles_per_day(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Computes number of articles per day based on a date-like column.

        If date_col is not datetime, attempts to parse it.
        Returns: DataFrame with columns ['date_only', 'count']
        """
        if date_col not in df.columns:
            raise ValueError(
                f"Column '{date_col}' not found. Available columns: {
                    df.columns.tolist()
                }"
            )

        dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
        result = (
            pd.DataFrame({"date_only": dates})
            .groupby("date_only", dropna=False)
            .size()
            .reset_index(name="count")
        )
        # Drop NaT rows if desired; keep for visibility by default.
        result = (
            result[result["date_only"].notna()]
            .sort_values("date_only")
            .reset_index(drop=True)
        )
        return result

    def compute_headline_lengths(
        self,
        df: pd.DataFrame,
        text_col: str = "headline",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Adds two columns to the DataFrame:
        - 'headline_len_chars': number of characters in text_col
        - 'headline_len_words': number of whitespace-separated tokens in text_col

        Set inplace=True to modify the input DataFrame; otherwise returns a copy.
        """
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found. Available columns: {
                    df.columns.tolist()
                }"
            )

        target = df if inplace else df.copy()

        s = target[text_col].fillna("")
        target["headline_len_chars"] = s.str.len()
        target["headline_len_words"] = s.str.split().str.len()

        return target

    def basic_headline_stats(
        self,
        df: pd.DataFrame,
        text_col: str = "headline",
    ) -> pd.DataFrame:
        """
        Returns descriptive statistics for headline lengths.
        Computes length columns if they are missing, without mutating the input DataFrame.
        """
        needs_chars = "headline_len_chars" not in df.columns
        needs_words = "headline_len_words" not in df.columns

        if needs_chars or needs_words:
            temp = self.compute_headline_lengths(df, text_col=text_col, inplace=False)
            return temp[["headline_len_chars", "headline_len_words"]].describe()

        return df[["headline_len_chars", "headline_len_words"]].describe()

    def count_publishers(
        self,
        df: pd.DataFrame,
        publisher_col: str = "publisher",
        top: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Counts how many rows exist for each publisher.

        - publisher_col: column name with publisher values
        - top: if provided, returns only the top-k publishers

        Returns: DataFrame with columns ['publisher', 'count']
        """
        if publisher_col not in df.columns:
            raise ValueError(
                f"Column '{publisher_col}' not found. Available columns: {
                    df.columns.tolist()
                }"
            )

        counts = df[publisher_col].fillna("UNKNOWN").value_counts().reset_index()
        counts.columns = ["publisher", "count"]
        if top is not None:
            counts = counts.head(int(top))
        return counts


if __name__ == "__main__":
    # Example usage:
    # df_news = pd.read_csv("news.csv")
    # eda = EdaStatistics()
    # print(eda.top_words(df_news, n=25))
    # print(eda.articles_per_day(df_news))
    # df_with_lengths = eda.compute_headline_lengths(df_news)
    # print(eda.basic_headline_stats(df_with_lengths))
    # print(eda.count_publishers(df_news, top=10))
    pass
