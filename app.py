import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cresca_quant import (
    CFG,
    annualized_return,
    annualized_volatility,
    download_adjusted_prices,
    historical_cvar,
    historical_var,
    max_drawdown,
    momentum_backtest,
    optimize_max_sharpe,
    portfolio_return_series,
    sharpe_ratio,
    summarize_returns,
    train_volatility_model,
)

st.set_page_config(page_title="CRESCA Quant", page_icon="📈", layout="wide")

st.title("CRESCA Quant")
st.subheader("Intelligence That Compounds")
st.caption("AI-powered quantitative investment and risk intelligence • Educational research prototype, not investment advice.")

with st.sidebar:
    st.header("Research settings")
    ticker_text = st.text_input("Tickers (comma-separated)", "SPY, QQQ, IWM, TLT, GLD")
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in ticker_text.split(",") if t.strip()))
    start_date = st.date_input("Start date", value=pd.Timestamp("2018-01-01").date())
    benchmark = st.selectbox("Benchmark", options=tickers or ["SPY"], index=0)
    risk_free_rate = st.slider("Illustrative annual risk-free rate", 0.0, 0.10, 0.02, 0.005)
    max_weight = st.slider("Maximum weight per asset", 0.20, 1.00, 0.60, 0.05)
    lookback = st.slider("Momentum lookback (trading days)", 21, 252, 126, 21)
    cost_bps = st.slider("Transaction cost (basis points)", 0.0, 25.0, 5.0, 1.0)
    run = st.button("Run quantitative analysis", type="primary", use_container_width=True)

if run:
    if len(tickers) < 2:
        st.error("Enter at least two valid tickers.")
        st.stop()
    with st.spinner("Downloading data and running the research models..."):
        try:
            prices = download_adjusted_prices(tickers, start_date.isoformat())
            returns = prices.pct_change().dropna()
            if benchmark not in returns.columns:
                benchmark_used = returns.columns[0]
                st.warning(f"{benchmark} data was unavailable; using {benchmark_used} as the benchmark.")
            else:
                benchmark_used = benchmark
            opt_weights = optimize_max_sharpe(returns, risk_free_rate, max_weight)
            momentum_returns, momentum_weights = momentum_backtest(returns, lookback, cost_bps)
            equal_returns = returns.mean(axis=1).rename("Equal-weight portfolio")
            benchmark_returns = returns[benchmark_used].rename(f"{benchmark_used} benchmark")
            comparison = pd.concat([momentum_returns, equal_returns, benchmark_returns], axis=1).dropna()
            summary = summarize_returns(returns, risk_free_rate)
            strategy_summary = summarize_returns(comparison, risk_free_rate)
            optimized_returns = portfolio_return_series(returns, opt_weights)
            risk_report = summarize_returns(optimized_returns.to_frame("Optimized portfolio"), risk_free_rate)
            forecast_asset = benchmark_used
            forecast_results, forecast_metrics, feature_importance, _ = train_volatility_model(returns[forecast_asset])
            stress_shocks = pd.Series({
                ticker: (-0.10 if ticker in ["SPY", "QQQ", "IWM"] else (-0.05 if ticker == "GLD" else 0.03))
                for ticker in returns.columns
            })
            stress_table = pd.DataFrame({
                "Portfolio weight": opt_weights.reindex(stress_shocks.index),
                "Hypothetical asset shock": stress_shocks,
                "Weighted contribution": opt_weights.reindex(stress_shocks.index).fillna(0) * stress_shocks,
            })
            st.session_state["cresca_results"] = {
                "prices": prices, "returns": returns, "summary": summary,
                "opt_weights": opt_weights, "momentum_returns": momentum_returns,
                "momentum_weights": momentum_weights, "comparison": comparison,
                "comparison_growth": (1 + comparison).cumprod(),
                "strategy_summary": strategy_summary, "risk_report": risk_report,
                "forecast_results": forecast_results, "forecast_metrics": forecast_metrics,
                "feature_importance": feature_importance, "stress_table": stress_table,
                "stress_impact": float(stress_table["Weighted contribution"].sum()),
                "benchmark_used": benchmark_used, "forecast_asset": forecast_asset,
                "start_date": start_date.isoformat(),
            }
        except Exception as exc:
            st.error(f"Analysis could not complete: {exc}")
            st.info("Check ticker symbols, date range, internet/data-provider availability, and dependency installation.")
            st.stop()

if "cresca_results" not in st.session_state:
    st.info("Choose your settings in the sidebar, then click **Run quantitative analysis**.")
    st.markdown("""
    ### What this dashboard explores
    - Historical market performance and risk
    - Random Forest volatility forecasts compared with a naive baseline
    - Long-only portfolio allocation research
    - A simple momentum backtest with illustrative transaction costs
    - Portfolio risk metrics and hypothetical stress scenarios

    Results depend on the selected assets, dates, and data-provider availability. Historical results do not predict future performance.
    """)
    st.stop()

data = st.session_state["cresca_results"]
# Re-run results if settings changed and user clicks the button; otherwise show last successful run.
prices = data["prices"]
returns = data["returns"]

st.caption(f"Data from {data['start_date']} through {prices.index[-1].date()} • {len(prices):,} aligned trading days • Assets: {', '.join(prices.columns)}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market overview", "Volatility ML", "Portfolio optimizer", "Strategy backtest", "Risk & stress test"
])

with tab1:
    st.header("Market overview")
    total_growth = (1 + returns).cumprod()
    cols = st.columns(3)
    cols[0].metric("Assets analyzed", len(prices.columns))
    cols[1].metric("Trading observations", f"{len(prices):,}")
    cols[2].metric("Latest data date", str(prices.index[-1].date()))
    normalized = prices / prices.iloc[0]
    st.plotly_chart(px.line(normalized, title="Normalized adjusted prices", labels={"value": "Growth relative to start", "index": "Date"}), use_container_width=True)
    st.plotly_chart(px.line(total_growth, title="Cumulative growth from daily returns", labels={"value": "Growth of $1", "index": "Date"}), use_container_width=True)
    drawdowns = total_growth / total_growth.cummax() - 1
    st.plotly_chart(px.line(drawdowns, title="Historical drawdowns", labels={"value": "Drawdown", "index": "Date"}), use_container_width=True)
    st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", aspect="auto", title="Daily return correlation matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    st.subheader("Asset risk and return summary")
    st.dataframe(data["summary"].style.format({
        "Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}", "Maximum Drawdown": "{:.2%}",
        "Historical VaR 95% (daily)": "{:.2%}", "Historical CVaR 95% (daily)": "{:.2%}",
    }), use_container_width=True)

with tab2:
    st.header("Machine-learning volatility forecast")
    st.write("A time-ordered Random Forest is compared with a simple recent-volatility baseline. Forecasts are research estimates, not predictions of investment returns.")
    st.dataframe(data["forecast_metrics"].style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}), use_container_width=True)
    st.plotly_chart(px.line(data["forecast_results"], title=f"{data['forecast_asset']} future volatility: forecast vs realized", labels={"value": "Annualized volatility", "index": "Date"}), use_container_width=True)
    fi = data["feature_importance"].sort_values()
    st.plotly_chart(px.bar(x=fi.values, y=fi.index, orientation="h", title="Random Forest feature importance", labels={"x": "Importance", "y": "Feature"}), use_container_width=True)

with tab3:
    st.header("Long-only portfolio optimizer")
    st.write("Weights maximize estimated historical Sharpe ratio subject to long-only positions and the selected per-asset cap.")
    weight_df = pd.DataFrame({"Optimized weight": data["opt_weights"], "Equal weight": 1 / len(data["opt_weights"])})
    st.plotly_chart(px.bar(weight_df, barmode="group", title="Portfolio allocation comparison", labels={"index": "Asset", "value": "Weight"}), use_container_width=True)
    st.dataframe(weight_df.style.format("{:.2%}"), use_container_width=True)
    st.subheader("Optimized portfolio risk summary")
    st.dataframe(data["risk_report"].style.format({
        "Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}", "Maximum Drawdown": "{:.2%}",
        "Historical VaR 95% (daily)": "{:.2%}", "Historical CVaR 95% (daily)": "{:.2%}",
    }), use_container_width=True)

with tab4:
    st.header("Momentum strategy backtest")
    st.write(f"Simple monthly momentum rotation using a {lookback}-trading-day lookback and {cost_bps:.1f} bps illustrative transaction costs. Backtest results may be sensitive to assumptions and are not a live strategy.")
    st.plotly_chart(px.line(data["comparison_growth"], title="Cumulative growth comparison", labels={"value": "Growth of $1", "index": "Date"}), use_container_width=True)
    st.dataframe(data["strategy_summary"].style.format({
        "Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}", "Maximum Drawdown": "{:.2%}",
        "Historical VaR 95% (daily)": "{:.2%}", "Historical CVaR 95% (daily)": "{:.2%}",
    }), use_container_width=True)
    st.plotly_chart(px.area(data["momentum_weights"], title="Momentum strategy holdings through time", labels={"value": "Weight", "index": "Date"}), use_container_width=True)

with tab5:
    st.header("Portfolio risk engine & stress test")
    st.write("Historical VaR/CVaR are estimated from daily returns. Stress assumptions below are hypothetical shocks, not forecasts.")
    st.metric("Illustrative combined portfolio shock", f"{data['stress_impact']:.2%}")
    st.dataframe(data["stress_table"].style.format("{:.2%}"), use_container_width=True)
    st.caption("Stress scenario assumptions: SPY/QQQ/IWM −10%; GLD −5%; other included assets +3%. These are deliberately simplified and editable in the code.")

with st.expander("Download research data", expanded=False):
    st.write("Export the data behind the charts for further inspection. CSV downloads contain historical research outputs, not live trading instructions.")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download adjusted prices (CSV)",
        data=prices.to_csv(index_label="Date").encode("utf-8"),
        file_name="cresca_adjusted_prices.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "Download asset risk summary (CSV)",
        data=data["summary"].to_csv(index_label="Asset").encode("utf-8"),
        file_name="cresca_asset_risk_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c3.download_button(
        "Download strategy comparison (CSV)",
        data=data["comparison"].to_csv(index_label="Date").encode("utf-8"),
        file_name="cresca_strategy_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c4, c5, c6 = st.columns(3)
    c4.download_button(
        "Download optimized weights (CSV)",
        data=data["opt_weights"].rename("Optimized weight").to_csv().encode("utf-8"),
        file_name="cresca_optimized_weights.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c5.download_button(
        "Download volatility forecasts (CSV)",
        data=data["forecast_results"].to_csv(index_label="Date").encode("utf-8"),
        file_name="cresca_volatility_forecasts.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c6.download_button(
        "Download stress test (CSV)",
        data=data["stress_table"].to_csv(index_label="Asset").encode("utf-8"),
        file_name="cresca_stress_test.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption("CRESCA Quant is an educational research prototype. It does not provide personalized financial advice, guarantee results, or account for all real-world trading constraints.")
