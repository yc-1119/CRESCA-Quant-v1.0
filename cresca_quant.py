"""CRESCA Quant: reusable quantitative finance research functions.

Educational research prototype; not investment advice.
This module intentionally performs no downloads, model training, plotting, or
printing at import time. Call the functions from a notebook or Streamlit app.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)

@dataclass
class Config:
    tickers: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD")
    benchmark: str = "SPY"
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    trading_days: int = 252
    risk_free_rate: float = 0.02
    transaction_cost_bps: float = 5.0
    momentum_lookback: int = 126
    volatility_window: int = 21
    forecast_horizon: int = 21
    max_weight: float = 0.60

CFG = Config()

def download_adjusted_prices(tickers: Tuple[str, ...], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Download adjusted prices and return a clean, date-indexed DataFrame."""
    tickers = tuple(tickers)
    if not tickers:
        raise ValueError("At least one ticker is required.")
    raw = yf.download(tickers=list(tickers), start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("No market data returned. Check tickers, dates, or provider availability.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Expected 'Close' field not found in downloaded data.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise RuntimeError("Expected 'Close' field not found in downloaded data.")
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    prices = prices.dropna(axis=1, how="all")
    if prices.empty or prices.shape[1] == 0:
        raise RuntimeError("No usable price columns were returned.")
    prices = prices.dropna(how="any")
    if len(prices) < 100:
        raise RuntimeError(f"Only {len(prices)} common observations remain; choose a longer date range or fewer tickers.")
    return prices

def max_drawdown(return_series: pd.Series) -> float:
    wealth = (1 + return_series.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())

def annualized_return(return_series: pd.Series, periods_per_year: int = 252) -> float:
    r = return_series.dropna()
    if len(r) == 0:
        return np.nan
    total_growth = (1 + r).prod()
    years = len(r) / periods_per_year
    return float(total_growth ** (1 / years) - 1) if total_growth > 0 and years > 0 else np.nan

def annualized_volatility(return_series: pd.Series, periods_per_year: int = 252) -> float:
    return float(return_series.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe_ratio(return_series: pd.Series, annual_rf: float = 0.02, periods_per_year: int = 252) -> float:
    r = return_series.dropna()
    if len(r) < 2:
        return np.nan
    daily_rf = (1 + annual_rf) ** (1 / periods_per_year) - 1
    excess = r - daily_rf
    vol = excess.std(ddof=1)
    return float(np.sqrt(periods_per_year) * excess.mean() / vol) if vol > 0 else np.nan

def historical_var(return_series: pd.Series, confidence: float = 0.95) -> float:
    return float(-return_series.dropna().quantile(1 - confidence))

def historical_cvar(return_series: pd.Series, confidence: float = 0.95) -> float:
    r = return_series.dropna()
    if r.empty:
        return np.nan
    tail = r[r <= r.quantile(1 - confidence)]
