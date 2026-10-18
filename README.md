
# CRESCA Quant v1.0
### The AI-Powered Quantitative Investment & Risk Intelligence System

**Intelligence That Compounds.**

CRESCA Quant is a Python-based quantitative finance research project exploring how data science, machine learning, and portfolio theory can support investment analysis and risk management.

## Project Objectives

- **Market Data Analytics** — Collect and analyze historical market data.
- **Risk Intelligence** — Evaluate volatility, Value at Risk (VaR), Conditional Value at Risk (CVaR), and maximum drawdown.
- **Machine Learning** — Explore Random Forest models for volatility forecasting and compare predictions against a baseline.
- **Portfolio Optimization** — Investigate long-only portfolio allocation using risk-adjusted return metrics.
- **Strategy Backtesting** — Evaluate a momentum-based strategy while accounting for transaction costs.
- **Stress Testing** — Examine how hypothetical market shocks may affect portfolio performance.
- **Data Visualization** — Present findings through charts and quantitative metrics.

## Technology Stack

- Python
- pandas and NumPy
- yfinance
- scikit-learn
- Matplotlib
- SciPy
- Google Colab / Jupyter Notebook

## Getting Started

### Option 1: Google Colab

1. Open Google Colab: https://colab.research.google.com/
2. Select **File → Open notebook → GitHub**.
3. Enter this repository's URL: https://github.com/yc-1119/CRESCA-Quant-v1.0
4. Open `CRESCA_Quant_Colab.ipynb`.
5. Follow the notebook instructions and run the cells in order.

### Option 2: Run Locally

Clone the repository and install the required packages:

```bash
git clone https://github.com/yc-1119/CRESCA-Quant-v1.0.git
cd CRESCA-Quant-v1.0
pip install yfinance pandas numpy matplotlib scikit-learn scipy
```

Open `CRESCA_Quant_Colab.ipynb` in Jupyter Notebook, or use the Python script as appropriate.

## Research Approach

The project combines quantitative analysis, statistical evaluation, and machine learning experimentation. Model predictions and historical backtests should be assessed against appropriate benchmarks, with attention to transaction costs, data limitations, and overfitting.

## Current Status

**Version 1.0 — Initial prototype**

The initial notebook and Python script have been uploaded. End-to-end execution, validation of results, and further refinements remain part of the development process.

## Disclaimer

CRESCA Quant is an educational and research project, not financial advice. Historical performance and model predictions do not guarantee future results. Hypothetical backtests may not reflect real-world trading conditions.
