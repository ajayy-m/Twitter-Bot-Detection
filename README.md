# Twitter Bot Detection

A machine learning system for detecting automated accounts (bots) on Twitter/X using hybrid approaches combining tabular features, GraphSAGE embeddings, and explainable AI.

## Overview

This project implements a bot detection pipeline that leverages:
- **Tabular Features**: User profile statistics, activity patterns, account age
- **Graph Embeddings**: GraphSAGE neural network for learning user representations from similarity graphs
- **Ensemble Classification**: XGBoost + Random Forest ensemble with feature selection
- **Explainable AI**: LIME explanations for model predictions
- **LLM Integration**: Optional LLM-powered natural language explanations

## Project Structure

```
Twitter_Bot_detection/
├── Dataset/                      # Training and test datasets
│   ├── training_embeddings_reference.csv
│   ├── training_tabular_reference.csv
│   └── project_dataset.xlsx
├── Other_codes/                  # Core ML pipeline
│   ├── DataPreProcess.py         # Data cleaning and preprocessing
│   ├── feature_engineer.py       # Feature engineering
│   ├── graphsage_embedding.py    # GraphSAGE embedding generation
│   ├── Normalize.py              # Feature normalization
│   ├── split.py                  # Train/test splitting
│   ├── classifier.py             # Model training and evaluation
│   └── similairty graph.py       # Graph construction
├── Project_inference_code/       # Inference API
│   ├── app.py.py                 # Flask API server
│   ├── index.html                # Web UI
│   └── api_file.xlsx             # API data
├── Trained_Models/               # Saved model artifacts
├── Documents/                   # Project documentation
├── literature survey papers/    # Reference papers
├── Demo video/                  # Demo video of this project
└── Requirement.txt              # Python dependencies
```

## Installation

```bash
pip install -r Requirement.txt.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 3.0.1 | Data manipulation |
| scikit-learn | 1.8.0 | ML algorithms |
| xgboost | 3.2.0 | Gradient boosting |
| torch | 2.11.0 | Deep learning |
| torch-geometric | 2.7.0 | Graph neural networks |
| Flask | 3.1.3 | Web API |
| lime | 0.2.0.1 | Model explainability |

## Pipeline

### 1. Data Preprocessing

```bash
python Other_codes/DataPreProcess.py
```

Cleans raw Twitter data, handles missing values, and standardizes formats.

### 2. Feature Engineering

```bash
python Other_codes/feature_engineer.py
```

Generates derived features:
- `tweets_per_day` — Tweet frequency normalized by account age
- `followers_per_day` — Follower acquisition rate
- `log_tweets_per_day`, `log_followers_per_day` — Log-transformed rates
- `followers_spike`, `tweet_spike` — Activity spike indicators
- `extreme_activity_score` — Combined activity metric
- `young_account_flag` — Binary flag for accounts < 90 days

### 3. Graph Construction & Embedding

```bash
python Other_codes/similairty graph.py
python Other_codes/graphsage_embedding.py
```

Builds user similarity graphs and generates GraphSAGE embeddings (64-dim).

### 4. Normalization

```bash
python Other_codes/Normalize.py
```

Standardizes features using StandardScaler.

### 5. Model Training

```bash
python Other_codes/classifier.py
```

Trains ensemble classifier with:
- Feature selection using mutual information + tree importance
- XGBoost + Random Forest voting
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### 6. Inference API

```bash
cd Project_inference_code
python app.py
```

Starts Flask server on `http://localhost:5000`. Open `index.html` in browser for web interface.

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict bot/human classification |
| `/explain` | POST | Get LIME explanation |
| `/llm-explain` | POST | LLM-powered explanation (optional) |

#### Environment Variables

Create `.env` file in `Project_inference_code/`:

```env
OPENROUTER_API_KEY=your_api_key_here
USE_LOCAL_MODEL=false
```

Set `USE_LOCAL_MODEL=true` to use local Ollama instance.

## Model Performance

The classifier uses an optimal threshold of **0.39** 

- `P(Bot) >= Threshold` → **BOT**
- `P(Bot) < Threshold` → **HUMAN**

## Dataset

- `Dataset/training_tabular_reference.csv` — Tabular features with labels
- `Dataset/training_embeddings_reference.csv` — Graph embedding features
- `Dataset/project_dataset.xlsx` — Raw project data

## References

See `literature survey papers/` for related academic work on bot detection.
