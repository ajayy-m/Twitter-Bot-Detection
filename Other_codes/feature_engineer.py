import pandas as pd
import numpy as np

# ===============================
# LOAD DATA
# ===============================
INPUT_FILE = "cleaned_project_dataset.xlsx"
df = pd.read_excel(INPUT_FILE)

# ===============================
# FEATURE ENGINEERING
# ===============================

df["account_age_days"] = df["account_age_days"].replace(0, np.nan)

df["tweets_per_day"] = np.where(
    df["account_age_days"] > 0,
    df["tweets_count"] / df["account_age_days"],
    0
)

df["followers_per_day"] = np.where(
    df["account_age_days"] > 0,
    df["followers_count"] / df["account_age_days"],
    0
)

df["log_tweets_per_day"] = np.log1p(df["tweets_per_day"])
df["log_followers_per_day"] = np.log1p(df["followers_per_day"])

df["followers_spike"] = np.where(
    df["account_age_days"] > 0,
    df["followers_count"] / np.sqrt(df["account_age_days"]),
    0
)

df["tweet_spike"] = np.where(
    df["account_age_days"] > 0,
    df["tweets_count"] / np.sqrt(df["account_age_days"]),
    0
)

df["extreme_activity_score"] = (
    df["log_followers_per_day"] + df["log_tweets_per_day"]
)

df["young_account_flag"] = (
    df["account_age_days"] < 90
).astype(int)

df["huge_followers_flag"] = (
    df["followers_count"] > df["followers_count"].quantile(0.95)
).astype(int)

df["young_and_popular"] = (
    df["young_account_flag"] * df["huge_followers_flag"]
)

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)
df["user_id"] = df["user_id"].astype(str)
df.to_excel("feature_engineered_dataset.xlsx", index=False)
print("Feature Engineered File saved as feature_engineered_dataset.xlsx")