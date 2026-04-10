"""
Bot Detection API — Flask backend
Install: pip install flask flask-cors
Run:     python app.py
Then open index.html in your browser.
"""

import os, time, requests
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
import lime, lime.lime_tabular
import random

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY   = "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"
LLM_MODEL            = "stepfun/step-3.5-flash:free"
LLM_MAX_TOKENS       = 800
LLM_TEMPERATURE      = 0.3
LLM_RETRY_BASE_DELAY = 10
LLM_RETRIES          = 5

# ── CLASSIFICATION THRESHOLD ─────────────────────────────────────────────────
# Optimal threshold derived from ROC/precision-recall analysis.
#   P(Human) >= 0.4237 → HUMAN
#   P(Human) <  0.4237 → BOT
THRESHOLD = 0.39

def classify(prob: float) -> str:
    return "HUMAN" if prob >= THRESHOLD else "BOT"

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
print("Loading model artifacts...")
model          = joblib.load("Models/random_forest_bot_classifier.pkl")
feature_list   = joblib.load("Models/random_forest_feature_list.pkl")
ml_cols        = joblib.load("Models/random_forest_ml_feature_cols.pkl")
robust_scaler  = joblib.load("Models/robust_scaler.pkl")
age_scaler     = joblib.load("Models/age_scaler.pkl")
emb_normalizer = joblib.load("Models/embedding_normalizer.pkl")

ref_emb       = pd.read_csv("Dataset/training_embeddings_reference.csv")
train_tabular = pd.read_csv("Dataset/training_tabular_reference.csv")
api_df        = pd.read_excel("api_file.xlsx", dtype={"user_id": str})

rename_map = {f"emb_{i}": str(i) for i in range(64)}
ref_emb    = ref_emb.rename(columns=rename_map)

emb_cols        = [c for c in ref_emb.columns if c.isdigit()]
DROP_SIM_COLS   = ["user_id", "user_name", "Label", "dataset"]
similarity_cols = [c for c in train_tabular.columns if c not in DROP_SIM_COLS]

nn = NearestNeighbors(n_neighbors=10, metric="cosine")
nn.fit(train_tabular[similarity_cols])
followers_95_quantile = train_tabular["followers_count"].quantile(0.95)

RAW_INPUT_COLS = [c for c in api_df.columns if c not in
                  ["user_id", "user_name", "Label", "label", "dataset"]]
binary_cols    = ["verified", "has_description", "has_prof_url", "has_location",
                  "has_prof_img", "young_account_flag", "huge_followers_flag", "young_and_popular"]

user_id_to_idx = {uid: i for i, uid in enumerate(ref_emb["user_id"].values)}
train_user_ids = train_tabular["user_id"].values

# Map each training row → its embedding row index (-1 if missing)
train_to_emb_idx = np.array([
    user_id_to_idx.get(uid, -1) for uid in train_user_ids
])
emb_matrix = ref_emb[emb_cols].values.astype(np.float32) 

# ── LIME SETUP ────────────────────────────────────────────────────────────────
def _build_background():
    bg = pd.DataFrame(index=train_tabular.index)
    for c in RAW_INPUT_COLS:
        bg[c] = train_tabular[c] if c in train_tabular.columns else 0
    return bg.replace([np.inf, -np.inf], 0).fillna(0).values

background_data = _build_background()
bg_medians      = pd.DataFrame(background_data, columns=RAW_INPUT_COLS).median()
cat_indices     = [i for i, n in enumerate(RAW_INPUT_COLS) if n in binary_cols]

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=background_data, feature_names=RAW_INPUT_COLS,
    class_names=["BOT", "HUMAN"], categorical_features=cat_indices,
    mode="classification", discretize_continuous=True, random_state=42,
)
print("Ready — http://localhost:5000")

def preprocess(df: pd.DataFrame, fill_nulls: bool = False) -> pd.DataFrame:
    df = df.copy()

    if fill_nulls:
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        df = df.fillna(bg_medians)

    # ── 1. Clip raw counts ────────────────────────────────────────────────────
    df["account_age_days"] = df["account_age_days"].clip(lower=1.0)
    df["tweets_count"]     = df["tweets_count"].clip(lower=0.0)
    df["followers_count"]  = df["followers_count"].clip(lower=0.0)

    # ── 2. Engineered features ────────────────────────────────────────────────
    df["tweets_per_day"]         = df["tweets_count"]    / df["account_age_days"]
    df["followers_per_day"]      = df["followers_count"] / df["account_age_days"]
    df["log_tweets_per_day"]     = np.log1p(df["tweets_per_day"])
    df["log_followers_per_day"]  = np.log1p(df["followers_per_day"])
    df["followers_spike"]        = df["followers_count"] / np.sqrt(df["account_age_days"])
    df["tweet_spike"]            = df["tweets_count"]    / np.sqrt(df["account_age_days"])
    df["extreme_activity_score"] = df["log_followers_per_day"] + df["log_tweets_per_day"]
    df["young_account_flag"]     = (df["account_age_days"] < 90).astype(int)
    df["huge_followers_flag"]    = (df["followers_count"] > followers_95_quantile).astype(int)
    df["young_and_popular"]      = df["young_account_flag"] * df["huge_followers_flag"]

    # ── 3. KNN neighbour lookup + embedding (fully vectorised) ────────────────
    for c in similarity_cols:
        if c not in df.columns:
            df[c] = 0
    sim_data = df[similarity_cols].replace([np.inf, -np.inf], 0).fillna(0)

    _, idx_matrix = nn.kneighbors(sim_data)          # shape: (N_rows, 10)

    # Map neighbour train-indices → embedding-indices, then average — pure numpy
    emb_idx_matrix = train_to_emb_idx[idx_matrix]   # shape: (N_rows, 10)

    # For missing embeddings (idx == -1), use zeros
    valid_mask = emb_idx_matrix >= 0                 # shape: (N_rows, 10)
    safe_idx   = np.where(valid_mask, emb_idx_matrix, 0)

    fetched    = emb_matrix[safe_idx]                # shape: (N_rows, 10, 64)
    fetched    = np.where(valid_mask[:, :, None], fetched, 0.0)

    count      = valid_mask.sum(axis=1, keepdims=True).clip(min=1)
    emb_means  = fetched.sum(axis=1) / count         # shape: (N_rows, 64)

    emb_df = pd.DataFrame(emb_means, columns=emb_cols, index=df.index)
    df     = pd.concat([df, emb_df], axis=1)

    # ── 4. Scale embeddings ───────────────────────────────────────────────────
    df[emb_cols] = df[emb_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[emb_cols] = emb_normalizer.transform(df[emb_cols])

    # ── 5. Robust scaler ──────────────────────────────────────────────────────
    robust_cols = robust_scaler.feature_names_in_
    for c in robust_cols:
        if c not in df.columns:
            df[c] = 0
    df[robust_cols] = robust_scaler.transform(df[robust_cols])

    # ── 6. Age scaler ─────────────────────────────────────────────────────────
    df[["account_age_days"]] = age_scaler.transform(df[["account_age_days"]])

    # ── 7. Ensure all model features exist ───────────────────────────────────
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0

    return df

# ── CORE FUNCTIONS ────────────────────────────────────────────────────
def run_pipeline(row: pd.DataFrame) -> float:
    processed = preprocess(row,fill_nulls=False)
    return model.predict_proba(processed[feature_list])[:, 1][0]

def _lime_predict(raw_array: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(raw_array, columns=RAW_INPUT_COLS)
    processed = preprocess(df,fill_nulls=True)
    probs     = model.predict_proba(processed[feature_list])[:, 1]
    return np.column_stack([1 - probs, probs])

def get_lime_features(row):
    lr = row[RAW_INPUT_COLS].copy()
    for col in binary_cols:
        if col in lr.columns: lr[col] = lr[col].fillna(0)
    lr  = lr.fillna(bg_medians)
    random.seed(42)
    np.random.seed(42)
    exp = explainer.explain_instance(
        data_row=lr.values[0].astype(float), predict_fn=_lime_predict,
        num_features=15, num_samples=500, labels=(1,)
    )
    return [{"rank": i+1, "feature": f, "weight": round(w, 6),
             "direction": "HUMAN" if w > 0 else "BOT"}
            for i, (f, w) in enumerate(exp.as_list(label=1))]

FEATURE_DESC = {
    "followers_count":"number of followers","friends_count":"number of accounts followed",
    "tweets_count":"total tweets posted","listed_count":"times added to Twitter lists",
    "hashtag_count":"total hashtags used","mentions_count":"total mentions of other users",
    "retweet_count":"total retweets received","reply_count":"total replies received",
    "url_count":"total URLs shared","ff_ratio":"follower-to-following ratio",
    "avg_hashtag":"avg hashtags per tweet","avg_mentions":"avg mentions per tweet",
    "avg_retweet":"avg retweets per tweet","avg_reply":"avg replies per tweet",
    "avg_url":"avg URLs per tweet","avg_user_engagement":"avg user engagement score",
    "has_description":"account has a bio","has_prof_url":"account has a profile URL",
    "has_location":"account has a location set","has_prof_img":"account has a profile image",
    "profile_completeness":"profile completeness","avg_polarity":"avg tweet sentiment polarity",
    "avg_subjectivity":"avg tweet subjectivity","unique_word_count":"total unique words used",
    "unique_word_use":"ratio of unique words","punctuation_count":"total punctuation used",
    "avg_sentence_length":"avg sentence length","punctuation_density":"punctuation density",
    "account_age_days":"account age in days","verified":"account is verified",
    "tweets_per_day":"avg tweets per day","followers_per_day":"avg follower gain per day",
    "followers_spike":"follower spike vs account age","tweet_spike":"tweet spike vs account age",
    "extreme_activity_score":"combined activity spike score",
    "young_account_flag":"account under 90 days old",
    "huge_followers_flag":"follower count in top 5%","young_and_popular":"new and high-follower account",
}

def _h(raw):
    for key, desc in FEATURE_DESC.items():
        if raw.startswith(key) or f" {key} " in raw or raw.endswith(key):
            cond = raw.replace(key,"").strip().lstrip("=").strip()
            return f"{desc} ({cond})" if cond else desc
    return raw

def get_llm_explanation(user_id, prediction, p_human, lime_features):
    bots   = [(f["feature"], f["weight"]) for f in lime_features if f["weight"] < 0]
    humans = [(f["feature"], f["weight"]) for f in lime_features if f["weight"] > 0]
    bot_lines   = "\n".join(f"  - {_h(f)} (influence: {abs(w):.4f})" for f,w in sorted(bots,   key=lambda x:x[1])[:5])
    human_lines = "\n".join(f"  - {_h(f)} (influence: {abs(w):.4f})" for f,w in sorted(humans, key=lambda x:-x[1])[:5])

    if prediction == "HUMAN":
        verdict    = "likely human"
        confidence = "narrow" if p_human <= 0.46 else "moderate"
    else:  # BOT
        verdict    = "likely automated (bot)"
        confidence = "narrow" if p_human >= 0.38 else "moderate"

    prompt = f"""You are an expert analyst writing a clear, structured report on a Twitter account's authenticity.

A machine learning model analysed account {user_id} and concluded it is {verdict}, with a {confidence} margin of confidence.

Evidence suggesting automated (bot) behaviour:
{bot_lines or "  - None identified"}

Evidence suggesting genuine human behaviour:
{human_lines or "  - None identified"}

Write a structured analysis as exactly 4 bullet points that a non-technical reader would understand. Format each bullet point starting with "• " on its own line.

Rules:
- Use **double asterisks** around key terms that deserve emphasis: the verdict (e.g. **likely a bot** or **likely human**), the confidence level (e.g. **moderate confidence** or **narrow margin**), and the 1-2 most important behavioural signals.
- Do NOT mention feature names, numbers, or model scores.
- Bullet 1: Clear verdict — state plainly whether this account appears to be **human**, a **bot**, or is **uncertain**, and at what confidence level.
- Bullet 2: Main behavioural patterns that led to this conclusion (plain English).
- Bullet 3: Any contradictory or mixed signals, if present. If none, note one additional supporting pattern.
- Bullet 4: A brief caveat about confidence and what this analysis can and cannot tell us.
Professional, journalistic tone. Complete every sentence. Each bullet must be 1-2 sentences."""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}",
               "Content-Type": "application/json", "HTTP-Referer": "https://bot-detection"}
    payload = {"model": LLM_MODEL, "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE,
               "messages": [{"role": "user", "content": prompt}]}

    for attempt in range(1, LLM_RETRIES + 1):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                choices = r.json().get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content")
                    text = content.strip() if isinstance(content, str) else ""
                    if text: return text
                if attempt < LLM_RETRIES: time.sleep(LLM_RETRY_BASE_DELAY); continue
            elif r.status_code == 429:
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            else:
                return f"LLM error {r.status_code}"
        except Exception as e:
            return f"Request failed: {e}"
    return "LLM explanation unavailable."

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/api/users")
def list_users():
    users = (api_df[["user_id","user_name"]]
             .dropna(subset=["user_id"])
             .assign(user_id=lambda d: d["user_id"].astype(str))
             .drop_duplicates("user_id"))
    return jsonify(users.to_dict(orient="records"))

@app.route("/api/analyse/<user_id>")
def analyse(user_id):
    row = api_df[api_df["user_id"] == str(user_id)]
    if row.empty:
        return jsonify({"error": f"User {user_id} not found"}), 404
    row       = row.iloc[[0]].copy()
    user_name = str(row["user_name"].values[0]) if "user_name" in row.columns else user_id
    prob          = run_pipeline(row)
    label         = classify(prob)
    lime_features = get_lime_features(row)
    llm_text      = get_llm_explanation(user_id, label, prob, lime_features)
    # Raw input features for frontend display (exclude ID/label cols)
    raw_display_cols = [c for c in RAW_INPUT_COLS if c not in
                        ["user_id","user_name","Label","label","dataset"]]
    raw_features = {}
    for c in raw_display_cols:
        v = row[c].values[0] if c in row.columns else None
        # convert numpy types to native Python for JSON serialisation
        if hasattr(v, 'item'):
            v = v.item()
        # replace NaN / Inf with None so Flask produces valid JSON
        if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
            v = None
        raw_features[c] = v

    return jsonify({
        "user_id": user_id, "user_name": user_name,
        "prediction": label, "p_human": round(prob, 4), "p_bot": round(1-prob, 4),
        "lime_features": lime_features, "llm_explanation": llm_text,
        "raw_features": raw_features,
    })

from flask import send_from_directory

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(debug=False, port=5000)