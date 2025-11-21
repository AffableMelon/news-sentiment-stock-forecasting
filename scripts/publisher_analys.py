import pandas as pd


def count_publishers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts how many articles each publisher has.
    """
    publisher_counts = df["publisher"].fillna("UNKNOWN").value_counts().reset_index()
    publisher_counts.columns = ["publisher", "count"]
    return publisher_counts
