# news-sentiment-stock-forecasting: Predicting Price Moves with News Sentimen


## Project Overview
This project is an intensive challenge focused on analyzing a large corpus of financial news data (FNSPID) to discover and quantify the correlation between news sentiment and stock market movements. 
The goal is to enhance Nova Financial Solutions' predictive analytics capabilities by leveraging insights from natural language processing (NLP) and technical analysis.
This repository will host all code, notebooks, and documentation generated during the challenge, focusing on Data Engineering, Financial Analytics, and Machine Learning Engineering skills.

## Tasks and Deliverables

### Task 1: Git and GitHub & Exploratory Data Analysis (EDA)
Focus: Setting up a reproducible Python environment, implementing Git version control, and performing initial data understanding.
EDA Activities: Descriptive statistics (headline length), publisher activity counts, time series analysis of publication frequency, and basic text analysis (keywords/topic modeling).

### Task 2: Quantitative Analysis using PyNance and TA-Lib
Focus: Integrating external stock price data and calculating essential technical indicators.
Activities: Loading OHLCV data, calculating indicators like Moving Averages (MA), Relative Strength Index (RSI), and MACD using TA-Lib, and visualizing the data.

### Task 3: Correlation between News and Stock Movement
Focus: The core correlation analysis required by the business objective.
Activities: Date alignment and normalization, conducting sentiment analysis on headlines using libraries like nltk and TextBlob, calculating daily stock returns, aggregating daily sentiment scores, and determining the Pearson correlation coefficient between sentiment and returns.


# Recreate on your machine
```
git clone
```

```
cd 

```
# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r req.txt
