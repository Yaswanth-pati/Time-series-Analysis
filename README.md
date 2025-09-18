# Time Series Analysis of U.S. Airline Passenger Traffic

## Overview

This project analyzes and forecasts U.S. airline passenger traffic using advanced time series techniques. The analysis leverages both **annual** and **monthly** datasets to capture long-term trends and short-term seasonality. Forecasting models such as **ARIMA**, **SARIMA**, **Holt-Winters Exponential Smoothing**, and **ensemble averaging** are applied to provide robust predictions for airline passenger volumes.

## Motivation

Accurate forecasts of airline passenger traffic are crucial for airline companies, policymakers, and infrastructure planners. The U.S. air transport sector is highly seasonal, with strong fluctuations during holidays and summer months, making it an ideal candidate for time series modeling.

## Data Sources

- **Annual Data (Nonseasonal):**
  - Source: [World Bank](https://data.worldbank.org/)
  - Description: Annual air passengers carried within the USA (1970–2022)
- **Monthly Data (Seasonal):**
  - Source: [U.S. Bureau of Transportation Statistics (BTS)](https://www.bts.gov/)
  - Description: Domestic monthly air traffic passenger counts (Jan 2021–Dec 2024)

## Methodology

1. **Data Cleaning & Preprocessing**
   - Removal of irrelevant columns
   - Handling missing values
   - Formatting time indices

2. **Stationarity Testing**
   - Augmented Dickey-Fuller (ADF) Test
   - KPSS Test
   - Application of differencing as needed

3. **Model Building**
   - **Nonseasonal Data:** ARIMA and Holt’s Linear Trend methods
   - **Seasonal Data:** SARIMA and Holt-Winters Multiplicative Seasonal models

4. **Model Selection & Evaluation**
   - ACF and PACF plots for order selection
   - Model diagnostics (AIC, BIC, residual analysis)
   - RMSE for forecast accuracy

5. **Ensemble Forecasting**
   - Averaging forecasts from ARIMA/SARIMA and Holt/Holt-Winters models

6. **Econometric Analysis**
   - Linear regression on time as a predictor for passenger volumes

## Results

- **Annual Data:** Holt’s Linear Trend model and ARIMA(0,1,1) provided strong forecasts, with Holt’s model yielding the lowest RMSE.
- **Monthly Data:** Holt-Winters Multiplicative model captured seasonality best, outperforming SARIMA in RMSE.
- **Ensemble Models:** Averaging forecasts provided stable and robust predictions for both datasets.

## Visualizations

- Time series plots for annual and monthly passenger data
- ACF and PACF plots for model diagnostics
- Forecast plots with confidence intervals
- Residual analysis and RMSE comparison charts

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Yaswanth-pati/Time-series-Analysis.git
   cd Time-series-Analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the notebook:
   - Open `Time series analysis.ipynb` in Jupyter Notebook or VSCode.
   - Run cells sequentially for full analysis and visualizations.

## Dependencies

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- pmdarima (for auto_arima)
- openpyxl (for Excel file support)

*(Install dependencies via `pip install -r requirements.txt`)*

## File Structure

- `Time series analysis.ipynb` — Main analysis notebook
- `non-seasonal.csv` — Annual passenger data (World Bank)
- `December 2024 Air Traffic.xlsx` — Monthly passenger data (BTS)
- `README.md` — Project overview and instructions

## Executive Summary

This project demonstrates the practical application of statistical time series models for transportation prediction. Through thorough model comparison, residual inspection, and ensemble techniques, it provides actionable insights and reliable forecasts for U.S. airline passenger volumes.

---

**Author:** [Yaswanth-pati](https://github.com/Yaswanth-pati)  
**License:** MIT
