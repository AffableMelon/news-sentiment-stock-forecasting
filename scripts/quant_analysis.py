from typing import Optional, Mapping, Sequence, Union, Dict, Any
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np
import pandas as pd
import talib


@dataclass
class MovingAverageConfig:
    sma_period: int = 20
    ema_period: int = 20


@dataclass
class RSIConfig:
    period: int = 14


@dataclass
class MACDConfig:
    fast: int = 12
    slow: int = 26
    signal: int = 9


@dataclass
class ATRConfig:
    period: int = 14


@dataclass
class BollingerBandsConfig:
    period: int = 20
    nbdevup: float = 2.0
    nbdevdn: float = 2.0
    matype: int = 0  # TA-Lib SMA


class QuantAnalysis:
    def __init__(
        self,
        column_map: Optional[Mapping[str, str]] = None,
        periods_per_year: int = 252,
    ):
        """
        Parameters:
        - column_map: mapping of logical keys to actual DataFrame columns.
          Logical keys: 'close','high','low','open','volume','date'
          Defaults assume Yahoo-style caps.
        - periods_per_year: used to annualize ratios
        """
        default_map = {
            "close": "Close",
            "high": "High",
            "low": "Low",
            "open": "Open",
            "volume": "Volume",
            "date": "Date",
        }
        self.col = {**default_map, **(column_map or {})}
        self.periods_per_year = int(periods_per_year)

    def _validate_columns(self, df: pd.DataFrame, required: Sequence[str]) -> None:
        """
        required should be logical keys, e.g. ['close','high','low','volume'].
        """
        missing = [key for key in required if self.col[key] not in df.columns]
        if missing:
            expected = [self.col[k] for k in required]
            raise ValueError(
                f"Missing required columns: {
                    missing
                }. Expected DataFrame columns to include: {expected}"
            )

    @staticmethod
    def _cfg_to_dict(
        cfg: Optional[Union[Dict[str, Any], Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Accept dict or dataclass; return dict. None stays None.
        """
        if cfg is None:
            return None
        if isinstance(cfg, dict):
            return cfg
        if is_dataclass(cfg):
            return asdict(cfg)
        raise TypeError(
            "Configuration must be a dict or a matching dataclass instance."
        )

    def add_indicators(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
        prefix: str = "",
        ma_cfg: Optional[Union[MovingAverageConfig, Dict[str, Any]]] = None,
        rsi_cfg: Optional[Union[RSIConfig, Dict[str, Any]]] = None,
        macd_cfg: Optional[Union[MACDConfig, Dict[str, Any]]] = None,
        atr_cfg: Optional[Union[ATRConfig, Dict[str, Any]]] = None,
        bbands_cfg: Optional[Union[BollingerBandsConfig, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Adds moving averages, RSI, MACD, ATR, and Bollinger Bands indicators using TA-Lib.

        None config => skip that indicator group.
        Dict or dataclass => compute using provided values.
        """
        # Validate with logical keys (lowercase)
        self._validate_columns(df, required=["close", "high", "low", "volume"])
        target = df if inplace else df.copy()

        close = target[self.col["close"]]
        high = target[self.col["high"]]
        low = target[self.col["low"]]

        ma_cfg = self._cfg_to_dict(ma_cfg)
        rsi_cfg = self._cfg_to_dict(rsi_cfg)
        macd_cfg = self._cfg_to_dict(macd_cfg)
        atr_cfg = self._cfg_to_dict(atr_cfg)
        bbands_cfg = self._cfg_to_dict(bbands_cfg)

        # Moving Averages
        if ma_cfg is not None:
            sma_p = int(ma_cfg["sma_period"])
            ema_p = int(ma_cfg["ema_period"])
            target[f"{prefix}SMA_{sma_p}"] = talib.SMA(close, timeperiod=sma_p)
            target[f"{prefix}EMA_{ema_p}"] = talib.EMA(close, timeperiod=ema_p)

        # RSI
        if rsi_cfg is not None:
            rsi_p = int(rsi_cfg["period"])
            target[f"{prefix}RSI_{rsi_p}"] = talib.RSI(close, timeperiod=rsi_p)

        # MACD
        if macd_cfg is not None:
            fast = int(macd_cfg["fast"])
            slow = int(macd_cfg["slow"])
            signal = int(macd_cfg["signal"])
            macd, macd_sig, macd_hist = talib.MACD(
                close, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            target[f"{prefix}MACD"] = macd
            target[f"{prefix}MACD_signal"] = macd_sig
            target[f"{prefix}MACD_hist"] = macd_hist

        # ATR
        if atr_cfg is not None:
            atr_p = int(atr_cfg["period"])
            target[f"{prefix}ATR_{atr_p}"] = talib.ATR(
                high, low, close, timeperiod=atr_p
            )

        # Bollinger Bands
        if bbands_cfg is not None:
            per = int(bbands_cfg["period"])
            up = float(bbands_cfg["nbdevup"])
            dn = float(bbands_cfg["nbdevdn"])
            matype = int(bbands_cfg["matype"])
            upper, mid, lower = talib.BBANDS(
                close, timeperiod=per, nbdevup=up, nbdevdn=dn, matype=matype
            )
            target[f"{prefix}BB_upper"] = upper
            target[f"{prefix}BB_mid"] = mid
            target[f"{prefix}BB_lower"] = lower

        return target

    def compute_returns(
        self,
        df: pd.DataFrame,
        price_col: Optional[str] = None,
        return_col: str = "daily_return",
        cumret_col: str = "cumulative_return",
        method: str = "simple",  # 'simple' or 'log'
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Computes period returns and cumulative return.
        """
        price_col = price_col or self.col["close"]
        if price_col not in df.columns:
            raise ValueError(
                f"Price column '{price_col}' not found. Available: {
                    df.columns.tolist()
                }"
            )

        target = df if inplace else df.copy()
        price = target[price_col].astype(float)

        if method == "simple":
            target[return_col] = price.pct_change()
            target[cumret_col] = (1.0 + target[return_col]).cumprod()
        elif method == "log":
            target[return_col] = np.log(price / price.shift(1))
            target[cumret_col] = np.exp(target[return_col].cumsum())
        else:
            raise ValueError("method must be 'simple' or 'log'")

        return target

    def sharpe_ratio(
        self,
        df: pd.DataFrame,
        return_col: str = "daily_return",
        risk_free_rate_per_period: float = 0.0,
        annualize: bool = True,
    ) -> float:
        """
        Sharpe ratio using mean excess return over standard deviation.
        risk_free_rate_per_period should be on the same period as return_col.
        """
        if return_col not in df.columns:
            raise ValueError(f"Return column '{return_col}' not found.")
        r = df[return_col].dropna().astype(float)
        if r.empty:
            return np.nan

        excess = r - risk_free_rate_per_period
        denom = excess.std(ddof=0)
        if denom == 0 or np.isnan(denom):
            return np.nan

        sr = excess.mean() / denom
        if annualize:
            sr *= np.sqrt(self.periods_per_year)
        return float(sr)

    def sortino_ratio(
        self,
        df: pd.DataFrame,
        return_col: str = "daily_return",
        risk_free_rate_per_period: float = 0.0,
        annualize: bool = True,
    ) -> float:
        """
        Sortino ratio uses downside deviation (std of negative excess returns).
        """
        if return_col not in df.columns:
            raise ValueError(f"Return column '{return_col}' not found.")
        r = df[return_col].dropna().astype(float)
        if r.empty:
            return np.nan

        excess = r - risk_free_rate_per_period
        downside = excess[excess < 0]
        denom = downside.std(ddof=0)
        if denom == 0 or np.isnan(denom):
            return np.nan

        sortino = excess.mean() / denom
        if annualize:
            sortino *= np.sqrt(self.periods_per_year)
        return float(sortino)

    def max_drawdown(
        self,
        df: pd.DataFrame,
        equity_col: str = "cumulative_return",
        return_col: Optional[str] = None,
    ) -> float:
        """
        Max drawdown of an equity curve.
        """
        if equity_col not in df.columns:
            if return_col is None or return_col not in df.columns:
                raise ValueError(
                    f"Equity column '{
                        equity_col
                    }' not found and no valid return_col provided."
                )
            equity = (1.0 + df[return_col].astype(float)).cumprod()
        else:
            equity = df[equity_col].astype(float)

        if equity.isna().all():
            return np.nan

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min())

    def calmar_ratio(
        self,
        df: pd.DataFrame,
        equity_col: str = "cumulative_return",
        return_col: Optional[str] = None,
    ) -> float:
        """
        Calmar ratio approximated as total return divided by |max_drawdown|.
        """
        mdd = self.max_drawdown(df, equity_col=equity_col, return_col=return_col)
        if np.isnan(mdd) or mdd == 0:
            return np.nan

        if equity_col not in df.columns:
            if return_col is None or return_col not in df.columns:
                raise ValueError(
                    f"Equity column '{
                        equity_col
                    }' not found and no valid return_col provided."
                )
            equity = (1.0 + df[return_col].astype(float)).cumprod()
        else:
            equity = df[equity_col].astype(float)

        if equity.empty or np.isnan(equity.iloc[-1]):
            return np.nan

        total_return = equity.iloc[-1] - 1.0
        return float(total_return / abs(mdd))


if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv("AAPL.csv")
    # qa = QuantAnalysis()
    # df = qa.add_indicators(df)
    # df = qa.compute_returns(df, method="simple", inplace=False)
    # print("Sharpe:", qa.sharpe_ratio(df))
    # print("Sortino:", qa.sortino_ratio(df))
    # print("Max Drawdown:", qa.max_drawdown(df))
    # print("Calmar:", qa.calmar_ratio(df))
    pass
