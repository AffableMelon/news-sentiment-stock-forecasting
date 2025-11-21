import pandas as pd


def load_news_csv(path: str) -> pd.DataFrame:
    """
    Loads the raw news dataset and performs basic cleaning.
    """
    df = pd.read_csv(path)
    # Strip whitespace
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    # Parse date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    #  remove fully empty rows
    df = df.dropna(how="all")

    return df
