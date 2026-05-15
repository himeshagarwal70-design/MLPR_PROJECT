# MLPR_PROJECT
semester 4 endsemester project

Rahil Shah,Himesh Agarwal,Sahil Patel
# Volatility Forecasting: Leveling Up Risk Prediction

### A Quantitative ML Pipeline for Financial Risk Management

This repository contains a high-integrity Machine Learning pipeline designed to predict the **21-day forward volatility** (risk) of 19 large-cap Indian equities.

Most trading desks rely on a 30-year-old method called **EWMA** (Exponentially Weighted Moving Average). This project "levels up" that standard by using an ensemble of Machine Learning models to catch the complex, non-linear patterns that EWMA misses.

---

## 1. Project Objective

The goal is to predict how much a stock's price will swing over the next trading month (21 days).

* **The Target:** We predict the **Log-Variance** of the next 21 days. Using "Log" ensures that extreme market crashes don't break our mathematical models.
* **The "Twist":** Instead of predicting volatility from scratch, our models predict the **Residual** (the error) of the industry-standard EWMA. We take the baseline and "correct" it.

---

## 2. The Data & Feature Engineering

We use a "Hybrid-5" framework to turn raw price and sentiment data into 15 powerful, non-redundant features.

### The Hybrid-5 Framework

To prevent the model from getting confused by too much similar data (Multicollinearity), we summarize every "Family" of data into exactly 5 features:

1. **Lag 1:** The value yesterday.
2. **Lag 2:** The value 2 days ago.
3. **Lag 3:** The value 3 days ago.
4. **14-Day Rolling Mean:** The recent trend (is it generally going up or down?).
5. **14-Day Rolling Std:** The recent stability (is it swinging wildly?).

### The Signal Families

We apply this "Hybrid-5" logic to three areas:

* **Volatility Family:** History of past risk.
* **Sentiment Family:** News flow and social media signals.
* **Returns Family:** Recent price performance.

---

## 3. Integrity Measures (No-Loophole Pipeline)

Financial data is famous for "Data Leakage" where a model accidentally "cheats" by looking at the future. This pipeline uses three strict protocols to ensure the results are honest and tradable:

### I. Cross-Fund Deduplication

The raw data contained the same stock listed under different mutual funds. If not handled, the same stock-day could appear in both the training and testing sets. We deduplicate the data to unique `(stock, date)` pairs to prevent this "memorization" bug.

### II. The 21-Day Embargo

Because our target variable summarizes the *next* 21 days, the end of our training data actually contains "future info" about the start of our testing data.

* **The Fix:** We forcefully drop (embargo) the last 21 days of the training set. This creates a "clean gap" so the model cannot see into the test period.

### III. No Look-Ahead Bias

We never use "Backward Fill" (`bfill`) for missing data. Using future knowledge to fill past gaps is cheating. We use **Forward Fill** (only carrying known past data forward) and a **60-day warm-up trim** to ensure every calculation is historically accurate.

---

## 4. Model Architecture

We use a **Residual Ensemble** consisting of three models:

1. **Ridge Regression:** A linear model that uses L2 Regularization to ignore noisy, redundant features.
2. **Random Forest:** A non-linear "bagging" model that looks for complex interactions between features.
3. **XGBoost:** a gradient boosting algorithm specifically tuned with high penalties (L1/L2) to prevent overfitting in the noisy financial environment.

---

## 5. Evaluation (How we measure success)

Standard accuracy (like R-squared) is often misleading in finance. We use industry-standard "stress tests":

* **QLIKE Loss:** The gold standard for volatility. It penalizes **under-predicting** risk more heavily than over-predicting (because under-predicting leads to portfolio blowups).
* **Mincer-Zarnowitz Regression:** A test to see if our predictions are "efficient" and "unbiased" compared to the actual market outcome.
* **Diebold-Mariano (DM) Test:** A statistical proof that our ML model is *genuinely* better than the EWMA baseline, and not just lucky. Our XGBoost model achieved a DM-Stat of **-13.17**, proving significant outperformance.

---

## 6. How to Run

1. **Install Dependencies:** `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`
2. **Prepare Data:** Ensure `clean_mf_dataset.csv` is in the root directory.
3. **Execute Pipeline:** Run `MLPR_Final_Pipeline.ipynb`.
* The notebook will automatically perform the Walk-Forward Cross-Validation.




---

## References

* **Patton (2011):** Foundations for using QLIKE as a robust loss function.
* **RiskMetrics (1996):** The J.P. Morgan standard for the $\lambda=0.94$ EWMA baseline.
* **Diebold & Mariano (1995):** The mathematical test for comparing forecast accuracy.
