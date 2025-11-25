from pathlib import Path
from typing import Optional, Sequence, Union
import re
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
        self,
        path: Union[str, Path],
        date_col: str = "date",
        naive_timezone: str = "UTC",
    ) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = self._strip_whitespace(df)
        df = self._standardize_columns(df, lowercase=False)

        lower_map = {c.lower(): c for c in df.columns}
        target = date_col.lower()
        if target in lower_map:
            real_col = lower_map[target]
            raw = df[real_col].astype(str).str.strip()
            offset_pattern = re.compile(r".*[+-]\d{2}:\d{2}$")

            def _parse_one(s: str):
                if s == "" or s.lower() in ("nan", "none"):
                    return pd.NaT
                try:
                    if offset_pattern.match(s):
                        return pd.to_datetime(s, utc=True)
                    else:
                        dt = pd.to_datetime(s, utc=False)
                        if dt.tzinfo is None:
                            return dt.tz_localize(naive_timezone)
                        return dt.tz_convert("UTC")
                except Exception:
                    return pd.NaT

            parsed = raw.apply(_parse_one)

            try:
                parsed = parsed.dt.tz_convert("UTC")
            except Exception:
                pass

            df[real_col] = parsed

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
    from io import StringIO

    data = """date
    2020-06-03 10:45:20-04:00
    2020-05-26 04:30:07-04:00
    2020-05-22 12:45:06-04:00
    2020-05-22 11:38:59-04:00
    2020-05-22 11:23:25-04:00
    2020-05-22 09:36:20-04:00
    2020-05-22 09:07:04-04:00
    2020-05-22 09:37:59-04:00
    2020-05-22 08:06:17-04:00
    2020-05-22 00:00:00
    2020-05-21 00:00:00
    2020-05-15 00:00:00
    """
    df_raw = pd.read_csv(StringIO(data))

    print("\n--- RAW CSV READ ---")
    print(df_raw)

    loader = DataLoader()
    news_df = loader.load_news_csv(
        # for demo; normally pass a filepath
        path="../src/data/raw_analyst_ratings.csv",
        date_col="date",
    )
    print("\n--- LOADER PARSED NEWS DF ---")
    print(news_df.tail())
