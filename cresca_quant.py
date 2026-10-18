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
    return float(-tail.mean()) if len(tail) else np.nan

def summarize_returns(return_frame: pd.DataFrame, rf: float = 0.02) -> pd.DataFrame:
    rows = {}
    for col in return_frame.columns:
        r = return_frame[col].dropna()
        rows[col] = {
            "Annualized Return": annualized_return(r),
            "Annualized Volatility": annualized_volatility(r),
            "Sharpe Ratio": sharpe_ratio(r, rf),
            "Maximum Drawdown": max_drawdown(r),
            "Historical VaR 95% (daily)": historical_var(r, 0.95),
            "Historical CVaR 95% (daily)": historical_cvar(r, 0.95),
        }
    return pd.DataFrame(rows).T

def make_volatility_dataset(asset_returns: pd.Series, window: int = 21, horizon: int = 21) -> Tuple[pd.DataFrame, pd.Series]:
    r = asset_returns.dropna().copy()
    df = pd.DataFrame(index=r.index)
    df["ret_1d"] = r
    df["mom_21d"] = (1 + r).rolling(21).apply(np.prod, raw=True).shift(1) - 1
    df["mom_63d"] = (1 + r).rolling(63).apply(np.prod, raw=True).shift(1) - 1
    df["vol_21d"] = r.rolling(window).std().shift(1) * np.sqrt(252)
    df["vol_63d"] = r.rolling(63).std().shift(1) * np.sqrt(252)
    df["abs_return_1d"] = r.abs().shift(1)
    df["rolling_mean_21d"] = r.rolling(21).mean().shift(1)
    future_sq = r.pow(2).shift(-1)
    target = future_sq.rolling(horizon).mean().shift(-(horizon - 1)).pow(0.5) * np.sqrt(252)
    df["target_future_vol"] = target
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df.pop("target_future_vol")
    return df.drop(columns=["ret_1d"]), y

def train_volatility_model(asset_returns: pd.Series, window: int = 21, horizon: int = 21, seed: int = SEED):
    """Fit a time-ordered Random Forest and compare it with a naive baseline."""
    X, y = make_volatility_dataset(asset_returns, window, horizon)
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_train) < 100 or len(X_test) < 30:
        raise RuntimeError("Not enough observations for the volatility model; extend the date range.")
    model = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=10,
                                  random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    results = pd.DataFrame({
        "Actual future volatility": y_test,
        "Random Forest forecast": pd.Series(model.predict(X_test), index=X_test.index),
        "Naive baseline": X_test["vol_21d"].clip(lower=0),
    }).dropna()
    actual = results["Actual future volatility"]
    metrics = pd.DataFrame({
        "Model": ["Random Forest", "Naive baseline"],
        "MAE": [mean_absolute_error(actual, results["Random Forest forecast"]),
                mean_absolute_error(actual, results["Naive baseline"])],
        "RMSE": [mean_squared_error(actual, results["Random Forest forecast"]) ** 0.5,
                 mean_squared_error(actual, results["Naive baseline"]) ** 0.5],
    })
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return results, metrics, importance, model

def optimize_max_sharpe(asset_returns: pd.DataFrame, annual_rf: float = 0.02, max_weight: float = 0.60) -> pd.Series:
    clean = asset_returns.dropna()
    if clean.shape[1] < 2:
        raise ValueError("At least two assets are needed for portfolio optimization.")
    mu, cov = clean.mean().values * 252, clean.cov().values * 252
    n = len(mu)
    if max_weight * n < 1:
        raise ValueError("max_weight is too small to allow weights to sum to one.")
    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0)))
        return -(ret - annual_rf) / vol if vol > 1e-12 else 1e6
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    initial = np.repeat(1 / n, n)
    initial = np.clip(initial, 0, max_weight)
    initial = initial / initial.sum()
    result = minimize(neg_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-10})
    if not result.success:
        return pd.Series(np.repeat(1 / n, n), index=clean.columns, name="Weight")
    return pd.Series(result.x, index=clean.columns, name="Weight")

def momentum_backtest(asset_returns: pd.DataFrame, lookback: int = 126, cost_bps: float = 5.0) -> Tuple[pd.Series, pd.DataFrame]:
    r = asset_returns.dropna(how="any").copy()
    prices_local = (1 + r).cumprod()
    momentum = prices_local.pct_change(lookback).shift(1)
    month_end = momentum.groupby(momentum.index.to_period("M")).tail(1).index
    target_weights = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for date in month_end:
        row = momentum.loc[date].dropna()
        if row.empty:
            continue
        chosen = row.idxmax()
        target_weights.loc[date, :] = 0.0
        target_weights.loc[date, chosen] = 1.0
    daily_weights = target_weights.ffill().fillna(0.0).shift(1).fillna(0.0)
    gross_returns = (daily_weights * r).sum(axis=1)
    turnover = daily_weights.diff().abs().sum(axis=1).fillna(daily_weights.abs().sum(axis=1))
    net_returns = gross_returns - turnover * (cost_bps / 10000.0)
    return net_returns.rename("Momentum strategy"), daily_weights

def portfolio_return_series(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = weights.reindex(asset_returns.columns).fillna(0)
    if not np.isclose(aligned.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    return (asset_returns * aligned).sum(axis=1)
