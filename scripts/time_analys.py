import pandas as pd


def articles_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes number of articles published per day.
    """
    df["date_only"] = df["date"].dt.date
    return df.groupby("date_only").size().reset_index(name="count")
