import pandas as pd
import numpy as np
import re

# ==============================================================
# 1. LOAD RAW (UNCLEANED) DATASET — FORCE user_id AS STRING
# ==============================================================
INPUT_FILE = "project_dataset.xlsx"

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Sheet1",
    dtype={"user_id": str}   # 🔥 CRITICAL FIX (PREVENT ROUNDING)
)

print("Original shape:", df.shape)
print("Original label distribution:")
print(df["Label"].value_counts(dropna=False))


# ==============================================================
# 2. CLEAN user_id (KEEP AS STRING — NEVER INT)
# ==============================================================
def clean_user_id(uid):
    if pd.isna(uid):
        return np.nan

    uid = str(uid).strip()
    uid = re.sub(r'^u', '', uid)     # remove leading 'u'
    uid = re.sub(r'\.0$', '', uid)   # remove trailing .0 if any
    uid = re.sub(r'[^0-9]', '', uid) # keep digits only

    return uid if uid else np.nan

df["user_id"] = df["user_id"].apply(clean_user_id).astype("string")

print("\nUser ID length check:")
print(df["user_id"].dropna().apply(len).describe())


# ==============================================================
# 3. NORMALIZE LABELS
# ==============================================================
label_map = {
    "bot": 1, "Bot": 1, "BOT": 1,
    "human": 0, "Human": 0, "HUMAN": 0
}

df["Label"] = df["Label"].map(label_map).fillna(-1).astype(int)

print("\nUnknown labels:", (df["Label"] == -1).sum())


# ==============================================================
# 4. SELECT IMPORTANT FEATURES
# ==============================================================
important_cols = [
    'user_id', 'user_name', 'verified',
    'friends_count', 'followers_count', 'listed_count',
    'Label', 'tweets_count', 'hashtag_count',
    'mentions_count', 'retweet_count', 'reply_count',
    'url_count', 'ff_ratio', 'avg_hashtag',
    'avg_mentions', 'avg_retweet', 'avg_reply',
    'avg_url', 'avg_user_engagement',
    'has_description', 'has_prof_url',
    'has_location', 'has_prof_img',
    'profile_completeness', 'avg_polarity',
    'avg_subjectivity', 'unique_word_count',
    'unique_word_use', 'punctuation_count',
    'avg_sentence_length', 'punctuation_density','account_age_days',
    'dataset'
]

df = df[[c for c in important_cols if c in df.columns]]

# ==============================================================
# 5. BOOLEAN → INT
# ==============================================================
bool_cols = [
    "has_description", "has_prof_url",
    "has_location", "has_prof_img", "verified"
]

for col in bool_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .replace({'true':1, 'false':0, True:1, False:0, '1':1, '0':0})
            .fillna(0)
            .astype(int)
        )


# ==============================================================
# 6. RECOMPUTE ff_ratio SAFELY
# ==============================================================
if all(c in df.columns for c in ["followers_count", "friends_count"]):
    df["ff_ratio"] = np.where(
        df["friends_count"] > 0,
        df["followers_count"] / df["friends_count"],
        np.where(df["followers_count"] > 0, 999999, 0)
    )


# ==============================================================
# 7. HANDLE MISSING VALUES + OUTLIERS
# ==============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ==============================================================
# 8. REMOVE DUPLICATES (NO SORTING — PRESERVE ORDER)
#    Keep row with highest tweets_count per user_id
# ==============================================================
df = (
    df.sort_values("tweets_count", ascending=False)
      .drop_duplicates(subset="user_id", keep="first")
      .reset_index(drop=True)
)


print("\nFinal shape:", df.shape)
print("\nFinal label distribution:")
print(df["Label"].value_counts())
# ================================= Account Age days Cleaning ================================
if "account_age_days" in df.columns:

    # Remove negative ages
    df.loc[df["account_age_days"] < 0, "account_age_days"] = np.nan

    # Optional: Remove unrealistic ages (> 20 years)
    max_reasonable_days = 365 * 20
    df.loc[df["account_age_days"] > max_reasonable_days, "account_age_days"] = np.nan

# ==============================================================
# 9. SAVE CLEANED DATASET AS XLSX (NO ROUNDING, NO EXTRA LIBS)
# ==============================================================
OUTPUT_FILE = "cleaned_project_dataset.xlsx"

df["user_id"] = df["user_id"].astype(str)

df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

print(f"\n✅ Cleaned dataset saved as XLSX: {OUTPUT_FILE}")
print("Sample user_id:", df['user_id'].iloc[0])
print("Length:", len(df['user_id'].iloc[0]))