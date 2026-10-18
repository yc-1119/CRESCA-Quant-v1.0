"""CRESCA Quant: AI-Powered Quantitative Investment & Risk Intelligence System.
Educational research prototype; not investment advice.
Run in Colab/Jupyter for display() support, or adapt plots for a script.
"""


# ========================================================================
# CRESCA Quant
# ========================================================================


# ========================================================================
# 1. Environment setup
# ========================================================================

# In Colab, run: %pip install yfinance pandas numpy scipy scikit-learn matplotlib seaborn


# ========================================================================
# 2. Imports and configuration
# ========================================================================

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

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
    end_date: Optional[str] = None  # None means through the latest available date
    trading_days: int = 252
    risk_free_rate: float = 0.02  # illustrative annual rate; update if needed
    transaction_cost_bps: float = 5.0
    momentum_lookback: int = 126
    volatility_window: int = 21
    forecast_horizon: int = 21
    max_weight: float = 0.60

CFG = Config()
print(CFG)


# ========================================================================
# 3. Data collection pipeline
# ========================================================================

def download_adjusted_prices(
    tickers: Tuple[str, ...],
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download adjusted close prices and return a clean, date-indexed DataFrame."""
    if not tickers:
        raise ValueError("At least one ticker is required.")

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("No market data returned. Check tickers, dates, or data-provider availability.")

    # With auto_adjust=True, Close represents split/dividend-adjusted prices.
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

    # Ensure requested tickers are represented where data exists.
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(how="all")

    # Do not forward-fill long gaps: retain only dates where all selected assets have prices.
    prices = prices.dropna(axis=1, how="all")
    if prices.empty or prices.shape[1] == 0:
        raise RuntimeError("No usable price columns were returned.")

    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        print(f"Warning: no data returned for: {missing}")

    prices = prices.dropna(how="any")
    if len(prices) < 100:
        raise RuntimeError(f"Only {len(prices)} common observations remain; choose a longer date range or fewer tickers.")
    return prices


prices = download_adjusted_prices(CFG.tickers, CFG.start_date, CFG.end_date)
returns = prices.pct_change().dropna()

print(f"Price rows: {len(prices):,} | Assets: {list(prices.columns)}")
display(prices.tail())


# ========================================================================
# 4. Data quality checks
# ========================================================================

quality_report = pd.DataFrame({
    "missing_prices": prices.isna().sum(),
    "first_valid_date": prices.apply(lambda s: s.first_valid_index()),
    "last_valid_date": prices.apply(lambda s: s.last_valid_index()),
    "observations": prices.count(),
})
display(quality_report)

if (prices <= 0).any().any():
    raise ValueError("Prices must be positive for log-return calculations.")

if returns.isna().any().any():
    print("Warning: returns contain missing values.")
else:
    print("Data checks passed: no missing values in the aligned price/return panels.")


# ========================================================================
# 5. Market intelligence: return and risk analytics
# ========================================================================

def max_drawdown(return_series: pd.Series) -> float:
    wealth = (1 + return_series.fillna(0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return float(drawdown.min())

def annualized_return(return_series: pd.Series, periods_per_year: int = 252) -> float:
    r = return_series.dropna()
    if len(r) == 0:
        return np.nan
    total_growth = (1 + r).prod()
    years = len(r) / periods_per_year
    if total_growth <= 0 or years <= 0:
        return np.nan
    return float(total_growth ** (1 / years) - 1)

def annualized_volatility(return_series: pd.Series, periods_per_year: int = 252) -> float:
    return float(return_series.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe_ratio(
    return_series: pd.Series,
    annual_rf: float = 0.02,
    periods_per_year: int = 252,
) -> float:
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
    threshold = r.quantile(1 - confidence)
    tail = r[r <= threshold]
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

asset_summary = summarize_returns(returns, CFG.risk_free_rate)
asset_formatters = {
    "Annualized Return": "{:.2%}",
    "Annualized Volatility": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Maximum Drawdown": "{:.2%}",
    "Historical VaR 95% (daily)": "{:.2%}",
    "Historical CVaR 95% (daily)": "{:.2%}",
}
display(asset_summary.style.format(asset_formatters).set_caption("Historical asset statistics"))


# ========================================================================
# 6. Visualize prices, growth, drawdowns, and correlations
# ========================================================================

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 5))
(prices / prices.iloc[0]).plot(ax=ax)
ax.set_title("CRESCA Quant | Normalized Adjusted Prices")
ax.set_ylabel("Growth of $1 (price-normalized; not dividend-reinvested total return if source differs)")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()

growth = (1 + returns).cumprod()
fig, ax = plt.subplots(figsize=(12, 5))
growth.plot(ax=ax)
ax.set_title("Cumulative Growth of $1 from Daily Returns")
ax.set_ylabel("Growth of $1")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()

drawdowns = growth / growth.cummax() - 1
fig, ax = plt.subplots(figsize=(12, 4))
drawdowns.plot(ax=ax)
ax.set_title("Historical Drawdowns")
ax.set_ylabel("Drawdown")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Daily Return Correlation Matrix")
plt.tight_layout()
plt.show()


# ========================================================================
# 7. Feature engineering for volatility forecasting
# ========================================================================

def make_volatility_dataset(
    asset_returns: pd.Series,
    window: int = 21,
    horizon: int = 21,
) -> Tuple[pd.DataFrame, pd.Series]:
    r = asset_returns.dropna().copy()
    df = pd.DataFrame(index=r.index)
    df["ret_1d"] = r
    df["mom_21d"] = (1 + r).rolling(21).apply(np.prod, raw=True).shift(1) - 1
    df["mom_63d"] = (1 + r).rolling(63).apply(np.prod, raw=True).shift(1) - 1
    df["vol_21d"] = r.rolling(window).std().shift(1) * np.sqrt(252)
    df["vol_63d"] = r.rolling(63).std().shift(1) * np.sqrt(252)
    df["abs_return_1d"] = r.abs().shift(1)
    df["rolling_mean_21d"] = r.rolling(21).mean().shift(1)
    # Forward-looking target: volatility over the next horizon trading days.
    future_sq = r.pow(2).shift(-1)
    target = future_sq.rolling(horizon).mean().shift(-(horizon - 1)).pow(0.5) * np.sqrt(252)
    df["target_future_vol"] = target
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df.pop("target_future_vol")
    X = df.drop(columns=["ret_1d"])
    return X, y

forecast_asset = CFG.benchmark if CFG.benchmark in returns.columns else returns.columns[0]
X, y = make_volatility_dataset(returns[forecast_asset], CFG.volatility_window, CFG.forecast_horizon)

split = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Forecast asset: {forecast_asset}")
print(f"Training observations: {len(X_train):,} | Test observations: {len(X_test):,}")


# ========================================================================
# 8. Train and evaluate a machine-learning volatility model
# ========================================================================

if len(X_train) < 100 or len(X_test) < 30:
    raise RuntimeError("Not enough observations for a meaningful demonstration. Extend the date range.")

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=10,
    random_state=SEED,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
pred_rf = pd.Series(rf_model.predict(X_test), index=X_test.index, name="Random Forest")

# Naive baseline: latest available 21-day volatility feature, annualized.
baseline_col = "vol_21d"
pred_naive = X_test[baseline_col].clip(lower=0).rename("Naive 21-day volatility")

model_results = pd.DataFrame({
    "Actual future volatility": y_test,
    "Random Forest forecast": pred_rf,
    "Naive baseline": pred_naive,
}).dropna()

metrics = pd.DataFrame({
    "Model": ["Random Forest", "Naive baseline"],
    "MAE": [
        mean_absolute_error(model_results["Actual future volatility"], model_results["Random Forest forecast"]),
        mean_absolute_error(model_results["Actual future volatility"], model_results["Naive baseline"]),
    ],
    "RMSE": [
        mean_squared_error(model_results["Actual future volatility"], model_results["Random Forest forecast"]) ** 0.5,
        mean_squared_error(model_results["Actual future volatility"], model_results["Naive baseline"]) ** 0.5,
    ],
})
display(metrics)

fig, ax = plt.subplots(figsize=(12, 5))
model_results.plot(ax=ax)
ax.set_title(f"{forecast_asset}: Future Volatility Forecast vs Realized Volatility")
ax.set_ylabel("Annualized volatility")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()

feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
feature_importance.plot(kind="barh", ax=ax)
ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()


# ========================================================================
# 9. Long-only portfolio optimization
# ========================================================================

def optimize_max_sharpe(
    asset_returns: pd.DataFrame,
    annual_rf: float = 0.02,
    max_weight: float = 0.60,
) -> pd.Series:
    clean = asset_returns.dropna()
    if clean.shape[1] < 2:
        raise ValueError("At least two assets are needed for portfolio optimization.")

    mu = clean.mean().values * 252
    cov = clean.cov().values * 252
    n = len(mu)

    if max_weight * n < 1:
        raise ValueError("max_weight is too small to allow weights to sum to one.")

    def portfolio_stats(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0)))
        return ret, vol

    def neg_sharpe(w):
        ret, vol = portfolio_stats(w)
        return -(ret - annual_rf) / vol if vol > 1e-12 else 1e6

    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    initial = np.repeat(1 / n, n)
    initial = np.clip(initial, 0, max_weight)
    initial = initial / initial.sum()

    result = minimize(
        neg_sharpe,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if not result.success:
        print(f"Optimizer warning: {result.message}. Returning equal weights as a fallback.")
        return pd.Series(np.repeat(1 / n, n), index=clean.columns, name="Weight")
    return pd.Series(result.x, index=clean.columns, name="Weight")

opt_weights = optimize_max_sharpe(returns, CFG.risk_free_rate, CFG.max_weight)
equal_weights = pd.Series(1 / len(returns.columns), index=returns.columns)

weights_table = pd.DataFrame({
    "Optimized weight": opt_weights,
    "Equal weight": equal_weights,
})
display(weights_table.style.format("{:.1%}"))

fig, ax = plt.subplots(figsize=(9, 4))
weights_table.plot(kind="bar", ax=ax)
ax.set_title("Portfolio Allocation Comparison")
ax.set_ylabel("Weight")
ax.set_xlabel("Asset")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ========================================================================
# 10. Strategy backtest: simple momentum with transparent assumptions
# ========================================================================

def momentum_backtest(
    asset_returns: pd.DataFrame,
    lookback: int = 126,
    cost_bps: float = 5.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    r = asset_returns.dropna(how="any").copy()
    prices_local = (1 + r).cumprod()

    # Momentum observed through the prior trading day.
    momentum = prices_local.pct_change(lookback).shift(1)

    # Rebalance on the last available trading day of each month.
    month_end = momentum.groupby(momentum.index.to_period("M")).tail(1).index
    # Store a complete target allocation only on rebalance dates. Non-rebalance
    # rows are NaN so the full allocation vector is carried forward together.
    target_weights = pd.DataFrame(np.nan, index=r.index, columns=r.columns)

    for date in month_end:
        row = momentum.loc[date].dropna()
        if row.empty:
            continue
        chosen = row.idxmax()
        target_weights.loc[date, :] = 0.0
        target_weights.loc[date, chosen] = 1.0

    # The allocation chosen at month-end becomes active on the next trading
    # day, avoiding use of the same day's close-to-close return as a signal.
    daily_weights = target_weights.ffill().fillna(0.0)
    daily_weights = daily_weights.shift(1).fillna(0.0)

    gross_returns = (daily_weights * r).sum(axis=1)
    turnover = daily_weights.diff().abs().sum(axis=1).fillna(daily_weights.abs().sum(axis=1))
    transaction_cost = turnover * (cost_bps / 10000.0)
    net_returns = gross_returns - transaction_cost
    return net_returns.rename("Momentum strategy"), daily_weights

momentum_returns, momentum_weights = momentum_backtest(
    returns,
    CFG.momentum_lookback,
    CFG.transaction_cost_bps,
)

equal_portfolio_returns = returns.mean(axis=1).rename("Equal-weight portfolio")
benchmark_returns = returns[CFG.benchmark].rename(f"{CFG.benchmark} benchmark") if CFG.benchmark in returns.columns else returns.iloc[:, 0].rename("Benchmark")

comparison = pd.concat(
    [momentum_returns, equal_portfolio_returns, benchmark_returns],
    axis=1
).dropna()

comparison_growth = (1 + comparison).cumprod()
fig, ax = plt.subplots(figsize=(12, 5))
comparison_growth.plot(ax=ax)
ax.set_title("Strategy Research: Cumulative Growth Comparison")
ax.set_ylabel("Growth of $1")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()

strategy_summary = summarize_returns(comparison, CFG.risk_free_rate)
strategy_formatters = {
    "Annualized Return": "{:.2%}",
    "Annualized Volatility": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Maximum Drawdown": "{:.2%}",
    "Historical VaR 95% (daily)": "{:.2%}",
    "Historical CVaR 95% (daily)": "{:.2%}",
}
display(strategy_summary.style.format(strategy_formatters).set_caption("Strategy performance summary"))


# ========================================================================
# 11. Portfolio risk engine and stress scenarios
# ========================================================================

def portfolio_return_series(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = weights.reindex(asset_returns.columns).fillna(0)
    if not np.isclose(aligned.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    return (asset_returns * aligned).sum(axis=1)

optimized_portfolio_returns = portfolio_return_series(returns, opt_weights)
risk_report = pd.DataFrame({
    "Optimized portfolio": {
        "Annualized Return": annualized_return(optimized_portfolio_returns),
        "Annualized Volatility": annualized_volatility(optimized_portfolio_returns),
        "Sharpe Ratio": sharpe_ratio(optimized_portfolio_returns, CFG.risk_free_rate),
        "Maximum Drawdown": max_drawdown(optimized_portfolio_returns),
        "Historical VaR 95% (daily)": historical_var(optimized_portfolio_returns, 0.95),
        "Historical CVaR 95% (daily)": historical_cvar(optimized_portfolio_returns, 0.95),
    }
}).T
risk_formatters = {
    "Annualized Return": "{:.2%}",
    "Annualized Volatility": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Maximum Drawdown": "{:.2%}",
    "Historical VaR 95% (daily)": "{:.2%}",
    "Historical CVaR 95% (daily)": "{:.2%}",
}
display(risk_report.style.format(risk_formatters))

# Illustrative one-day price shocks by asset. Edit these assumptions and explain why.
stress_shocks = pd.Series({
    ticker: (-0.10 if ticker in ["SPY", "QQQ", "IWM"] else (-0.05 if ticker == "GLD" else 0.03))
    for ticker in returns.columns
})
stress_impact = float((opt_weights.reindex(stress_shocks.index).fillna(0) * stress_shocks).sum())

stress_table = pd.DataFrame({
    "Portfolio weight": opt_weights.reindex(stress_shocks.index),
    "Hypothetical asset shock": stress_shocks,
    "Weighted contribution": opt_weights.reindex(stress_shocks.index).fillna(0) * stress_shocks,
})
display(stress_table.style.format("{:.2%}"))
print(f"Illustrative combined portfolio shock: {stress_impact:.2%}")


# ========================================================================
# 12. Compare risk and performance visually
# ========================================================================

risk_plot = strategy_summary[["Annualized Return", "Annualized Volatility", "Maximum Drawdown"]].copy()
fig, ax = plt.subplots(figsize=(10, 5))
risk_plot.plot(kind="bar", ax=ax)
ax.set_title("Return, Volatility, and Maximum Drawdown Comparison")
ax.set_ylabel("Ratio")
ax.set_xlabel("Strategy")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
momentum_weights.plot.area(ax=ax, stacked=True, alpha=0.8)
ax.set_title("Momentum Strategy Holdings Through Time")
ax.set_ylabel("Portfolio weight")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()


# ========================================================================
# 13. Reproducibility and limitations checklist
# ========================================================================


# ========================================================================
# 14. Suggested project structure for GitHub
# ========================================================================
