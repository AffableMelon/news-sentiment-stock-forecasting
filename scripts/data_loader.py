from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.map(lambda x: x.strip() if isinstance(x, str) else x)

    def _standardize_columns(
        self, df: pd.DataFrame, lowercase: bool = True
    ) -> pd.DataFrame:
        cols = [c.strip() for c in df.columns]
        if lowercase:
            cols = [c.lower() for c in cols]
        df.columns = cols
        return df

    def load_news_csv(
        self, path: Union[str, Path], date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Loads the raw news dataset and performs basic cleaning.
        - Strips whitespace from string cells
        - Standardizes column names (trim + lowercase)
        - Parses date column if present
        - Drops fully empty rows
        """
        df = pd.read_csv(path)
        df = self._strip_whitespace(df)
        df = self._standardize_columns(df, lowercase=True)

        if date_col.lower() in df.columns:
            df[date_col.lower()] = pd.to_datetime(df[date_col.lower()], errors="coerce")

        # remove fully empty rows
        df = df.dropna(how="all")

        return df

    def load_stock_price(
        self,
        path: Union[str, Path],
        cols: Optional[Sequence[str]] = ("volume", "close", "high", "low", "open"),
        date_col_candidates: Sequence[str] = ("date", "timestamp", "datetime"),
        preview: bool = True,
    ) -> pd.DataFrame:
        """
        Loads the raw stock prices dataset and performs basic cleaning.
        - Strips whitespace from string cells
        - Standardizes column names (trim only; preserves original casing)
        - Parses a date-like column if present (tries common names, case-insensitive)
        - Drops fully empty rows
        - Validates required columns if provided (case-insensitive)
        """
        df = pd.read_csv(path)
        df = self._strip_whitespace(df)
        df = self._standardize_columns(df, lowercase=False)

        # Build case-insensitive column lookup
        lower_map = {c.lower(): c for c in df.columns}

        # Parse a date column if we can find one (case-insensitive)
        for candidate in date_col_candidates:
            if candidate.lower() in lower_map:
                orig_name = lower_map[candidate.lower()]
                df[orig_name] = pd.to_datetime(df[orig_name], errors="coerce")
                break  # parse only the first matching candidate

        df = df.dropna(how="all")

        # Validate required columns (case-insensitive) if requested
        if cols is not None:
            missing_cols = [col for col in cols if col.lower() not in lower_map]
            if missing_cols:
                available = df.columns.tolist()
                raise ValueError(
                    f"DataFrame is missing the following required columns: {
                        missing_cols
                    }. "
                    f"Available columns are: {available}"
                )

        if preview:
            print(df.head())

        return df


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_stock_price("../src/data/AAPL.csv")
    print(df.columns)
