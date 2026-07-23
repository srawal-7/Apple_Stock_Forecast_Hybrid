# Hybrid Stock Forecasting: Merging Market Signals with News Sentiment

Comparing four forecasting architectures for next-day Apple (AAPL) stock price prediction, combining historical price data with FinBERT-scored news sentiment.

## Problem

Traditional stock forecasting models rely on historical price data alone and often miss sentiment-driven signals from financial news, particularly during periods of heightened volatility. This project tests whether adding news sentiment as a feature measurably improves next-day price prediction compared to price data alone.

*Originally developed as a team project for a graduate course at Drexel University.*

## Approach

**Data:** 9 years of AAPL historical price data (Open, High, Low, Close, Volume) via yfinance, merged with a Kaggle dataset of Apple-related financial news headlines and article text.

**Preprocessing:** Cleaned and merged both datasets on date. News text was tokenized, lemmatized, and stripped of stopwords. Sentiment scores were generated using FinBERT (`yiyanghkust/finbert-tone`), a transformer model fine-tuned specifically for financial sentiment, rather than relying on the news dataset's own pre-computed sentiment scores.

**Feature engineering:** Technical indicators including Simple and Exponential Moving Averages (SMA, EMA), Relative Strength Index (RSI), MACD, and a lagged sentiment score (previous day's news sentiment), plus 1 and 2-step lag features for tree-based models.

**Four models compared**, all predicting next-day closing price as a continuous value (regression, not classification):
1. Linear Regression (baseline)
2. Random Forest
3. XGBoost, tuned via grid search across 2,187 parameter combinations with 5-fold cross-validation
4. A 2-layer LSTM (60-day sequences, 50 units per layer, dropout 0.2)

## Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression (historical only) | 0.997 | $0.83 | $1.16 |
| Linear Regression (+ sentiment) | 0.997 | $0.80 | $1.11 |
| Random Forest (no lag features) | 0.18 | $12.91 | $20.88 |
| XGBoost (tuned, no lag features) | 0.20 | $12.58 | $20.58 |
| Random Forest (+ lag features) | 0.30 | $11.19 | $19.22 |
| XGBoost (+ lag features) | 0.26 | $11.62 | $19.80 |
| **LSTM** | **0.984** | **$1.04** | **$1.85** |

Sentiment integration produced a small, consistent improvement in prediction error for the linear model across every metric. Tree-based models substantially underperformed the linear baseline until lag features were added, since Random Forest and XGBoost don't natively account for sequence order in time-series data. The LSTM, which is explicitly designed for sequential data, was the strongest non-linear approach.

**Notable finding:** the very high R² across models (0.997 for linear, 0.984 for LSTM) is itself a caution flag rather than pure evidence of skill. Apple's daily closing price changes very little day to day, so predicting "close to today's price" is a strong baseline on its own, and the model comparison is more informative than any single R² value in isolation.

## Limitations

- Dataset covers a single stock (AAPL) over a relatively short window, so findings may not generalize to more volatile stocks or different market regimes
- No classification or up/down directional prediction was built in this version; all models predict continuous next-day closing price
- High R² values likely partly reflect strong day-to-day price autocorrelation rather than purely model skill

## Tech Stack

Python, yfinance, FinBERT (Transformers), XGBoost, scikit-learn, TensorFlow/Keras (LSTM), NLTK

## Repository Structure

```
hybrid-stock-forecasting/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Proposal_and_EDA.ipynb
│   ├── Preprocessing_and_Sentiment_Analysis.ipynb
│   └── Models.ipynb
├── data/
│   └── (Kaggle news dataset not included; price data pulled live via yfinance)
└── outputs/
    └── (merged_data.csv, model comparison results)
```

## Data Sources

- [Apple Stock (AAPL) Historical Financial News Data](https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data/data), Kaggle
- Historical price data via [yfinance](https://pypi.org/project/yfinance/)

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Proposal_and_EDA.ipynb
```

Run in order: EDA and proposal notebook first, then preprocessing/sentiment analysis, then the models notebook, which depends on `merged_data.csv` produced by the previous step.
