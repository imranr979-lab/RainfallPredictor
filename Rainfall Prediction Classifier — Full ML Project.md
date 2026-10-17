# 🌧 Rainfall Prediction Classifier

Building, tuning, and evaluating ML classifiers on Australian Bureau of Meteorology weather data — Random Forest vs Logistic Regression

Python 3.11 Scikit-learn 1.8 BOM / Kaggle Dataset Sydney, Australia 2008–2017

[Overview](#overview) [Data Exploration](#eda) [Feature Engineering](#features) [Pipeline](#pipeline) [Random Forest](#rf) [Logistic Regression](#lr) [Comparison](#compare) [Pilot → Production](#prod) [CISO Governance](#ciso)

1 Project Overview & Dataset

Source: Australian Government Bureau of Meteorology · Kaggle weather-dataset-rattle-package

36,530

Raw observations (all locations)

1,935

Sydney rows (after filtering & dropna)

18

Features used after engineering

2008–2017

Observation period

### Step 1 — Install Libraries & Load Data

```
# Install required libraries
!pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df_raw = pd.read_csv('weatherAUS.csv')
print(df_raw.shape)     # (36530, 23)
print(df_raw.dtypes)
print(df_raw.isnull().sum())
```

### Step 2 — Drop Rows with Missing Values

```
df = df_raw.dropna()
print(f"Before: {df_raw.shape[0]:,} rows")   # 36,530
print(f"After:  {df.shape[0]:,} rows")      # 19,459
print(f"Dropped: {df_raw.shape[0]-df.shape[0]:,} rows")
```

⚠ 46.7% of rows removed by dropna()

This aggressive drop is used here for simplicity. In production you would use `SimpleImputer(strategy='median')` inside the pipeline to retain data — which is exactly what the preprocessing transformer below does for the model's training data.

---

2 Data Exploration & Class Balance

Understanding the distribution of observations and target variable before modelling

Class balance and season distribution

Figure 1 — Class distribution (67.6% No Rain / 32.4% Rain) and the engineered Season feature derived from Date

💡 What can you conclude from the class counts?

The dataset is **moderately imbalanced**: 67.6% of days show no rain the next day vs 32.4% rainy days. This means:

- A naive classifier that always predicts "No Rain" would achieve 67.6% accuracy — making raw accuracy a misleading metric
- We must use **Stratified K-Fold cross-validation** to preserve the 68:32 ratio in every fold
- We report **Recall and F1-score** for the 'Yes' class, not just accuracy
- Classifiers are configured with `class_weight='balanced'` to penalise missed rain predictions more heavily

---

3 Data Leakage & Feature Engineering

Identifying leaky features, selecting location, and extracting seasonality

🔴 Points to Note 1 — Features inefficient for predicting tomorrow's rainfall

| Feature | Problem | Decision |
| --- | --- | --- |
| `Rainfall` | Direct proxy for `RainToday`; encoding today's exact amount can leak tomorrow's label | Dropped ❌ |
| `RainToday` | Binary threshold of `Rainfall`; kept but with caution — retained as categorical signal | Kept (cat) ✅ |
| `Temp9am / Temp3pm` | Intra-day redundancy with `MinTemp / MaxTemp`; adds noise without new information | Dropped ❌ |
| `Date` | Too granular — year/day overfit; replaced with meteorological season | → Season ✅ |
| `Location` | Multiple climate zones violate IID assumption across the dataset | Sydney only ✅ |

### Seasonality Feature & Location Filter

```
# Map YYYY-MM-DD dates to Australian meteorological seasons
def date_to_season(date_str):
    month = pd.to_datetime(date_str).month
    if   month in [12, 1, 2]: return 'Summer'
    elif month in [3, 4, 5]:  return 'Autumn'
    elif month in [6, 7, 8]:  return 'Winter'
    else:                       return 'Spring'

# Filter to Sydney; map dates; drop Date and leaky columns
df = df[df['Location'] == 'Sydney'].copy()
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns=['Date', 'Rainfall', 'Temp9am', 'Temp3pm', 'Location'])

# Define features and target
X = df.drop(columns=['RainTomorrow'])
y = df['RainTomorrow']   # Binary: 'Yes' / 'No'
```

---

4 Preprocessing Pipeline Architecture

Automated column detection, separate transformers, ColumnTransformer, and GridSearchCV

Pipeline architecture diagram

Figure 6 — Scikit-learn Pipeline: raw features → numerical transformer (impute+scale) → categorical transformer (impute+OHE) → ColumnTransformer → Classifier

### Step 3 — Preprocessing & Pipeline Construction

```
# Automatically detect numeric and categorical columns
num_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()
# num_features: [MinTemp, MaxTemp, Evaporation, Sunshine, WindGustSpeed,
#                WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm,
#                Pressure9am, Pressure3pm, Cloud9am, Cloud3pm]  → 13 cols
# cat_features: [WindGustDir, WindDir9am, WindDir3pm, RainToday, Season] → 5 cols

# Numerical transformer: fill median gaps → standardise
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# Categorical transformer: fill mode gaps → one-hot encode
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine into a single ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Stratified 80/20 split preserving 68:32 class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Stratified 5-fold cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

5 Random Forest Classifier

Pipeline + GridSearchCV parameter optimisation → evaluation

### Step 4 — RF Pipeline + Grid Search

```
# Build pipeline: preprocessor → Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(
        random_state=42, class_weight='balanced', n_jobs=-1))
])

# Parameter grid — searching 24 combinations
rf_param_grid = {
    'classifier__n_estimators':      [100, 200],
    'classifier__max_depth':         [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__max_features':      ['sqrt', 'log2'],
}

rf_grid_search = GridSearchCV(
    estimator  = rf_pipeline,
    param_grid = rf_param_grid,
    cv         = cv_strategy,        # StratifiedKFold(5)
    scoring    = 'f1_weighted',     # handles imbalance
    n_jobs     = -1,
    refit      = True               # auto-fit best model
)
rf_grid_search.fit(X_train, y_train)

# Best parameters found:
print(rf_grid_search.best_params_)
# {'classifier__max_depth': 10, 'classifier__max_features': 'log2',
#   'classifier__min_samples_split': 5, 'classifier__n_estimators': 100}
print(f"Best CV score: {rf_grid_search.best_score_:.4f}")  # 0.6844
print(f"Test accuracy: {rf_grid_search.score(X_test, y_test):.4f}")  # 0.6493

# Predictions on unseen test data
y_pred_rf = rf_grid_search.predict(X_test)
print(classification_report(y_test, y_pred_rf))
```

0.649

RF Test Accuracy

0.466

RF Precision (Rain)

0.384

RF Recall / TPR (Rain)

0.421

RF F1-Score (Rain)

Feature importances

Figure 4 — Top 15 features by mean decrease in Gini impurity (Random Forest). Season and RainToday dominate, followed by temperature and atmospheric pressure.

🌧 Points to Note 2 — True Positive Rate (Recall)

**TPR = 0.3840 (38.4%)** — the Random Forest correctly identifies 38.4% of actual rainy days. In an operational weather forecast context, **maximising recall is more important than maximising precision**: missing a rainfall warning (False Negative) is more costly than issuing a false alarm (False Positive). The Logistic Regression achieves higher recall (56.8%) at the cost of slightly lower precision.

---

6 Logistic Regression — Alternative Classifier

Updating the pipeline with a new classifier and different hyperparameter grid

### Step 5 — Update Pipeline to Logistic Regression

```
# Update pipeline — only change the classifier step
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),          # same preprocessing
    ('classifier',   LogisticRegression(
        random_state=42, class_weight='balanced',
        max_iter=1000, solver='lbfgs'))
])

# New parameter grid appropriate for Logistic Regression
lr_param_grid = {
    'classifier__C':       [0.01, 0.1, 1.0, 10.0],  # inverse regularisation
    'classifier__penalty': ['l2'],
    'classifier__solver':  ['lbfgs', 'liblinear'],
}

lr_grid_search = GridSearchCV(
    estimator=lr_pipeline, param_grid=lr_param_grid,
    cv=cv_strategy, scoring='f1_weighted',
    n_jobs=-1, refit=True
)
lr_grid_search.fit(X_train, y_train)

print(lr_grid_search.best_params_)
# {'classifier__C': 0.1, 'classifier__penalty': 'l2', 'classifier__solver': 'liblinear'}
y_pred_lr = lr_grid_search.predict(X_test)
print(classification_report(y_test, y_pred_lr))
```

0.640

LR Test Accuracy

0.444

LR Precision (Rain)

0.568

LR Recall / TPR (Rain)

0.498

LR F1-Score (Rain)

---

7 Model Comparison & Visualisations

Confusion matrices, ROC curves, and head-to-head metric comparison

Confusion matrices

Figure 2 — Confusion matrices for both classifiers on the held-out test set. Rows = actual, columns = predicted.

ROC curves

Figure 3 — ROC curves with AUC scores. A higher AUC indicates better discrimination between rainy and dry days across all thresholds.

Model comparison

Figure 5 — Direct metric comparison: accuracy, precision, and recall for the 'Rain' (Yes) class.

### Head-to-Head Summary

| Metric | Random Forest | Logistic Regression | Winner |
| --- | --- | --- | --- |
| Test Accuracy | 0.6493 | 0.6400 | RF ✅ |
| CV Score (f1_weighted) | 0.6844 | 0.6686 | RF ✅ |
| Precision (Rain) | 0.4660 | 0.4437 | RF ✅ |
| **Recall / TPR (Rain) ⭐** | 0.3840 | 0.5680 | LR ✅ |
| F1-Score (Rain) | 0.4211 | 0.4982 | LR ✅ |

📊 Conclusion

Random Forest wins on **accuracy, CV score, and F1**. Logistic Regression wins on **recall (TPR)** — it misses fewer rainy days. For a rainfall early-warning system where missing rain is costlier than false alarms, **Logistic Regression's higher recall makes it the more appropriate production model**, even at the cost of lower overall accuracy. In practice, you would tune the decision threshold on the RF model to trade precision for recall.

---

8 Pilot → Production Deployment Roadmap

Four-stage transition from notebook experiment to operational ML service

Stage 1 — Pilot Validation (Weeks 1–4)

#### 📋 Acceptance Criteria

- Recall (TPR) ≥ 55% on held-out test set
- F1-weighted ≥ 0.60 on 5-fold CV
- Latency \< 200ms per prediction (99th percentile)
- Zero PII leakage in model artefacts
- Reproducible: same seed → same results

#### 🔬 Pilot Activities

- Shadow mode: run alongside existing forecast — do not serve predictions yet
- Compare model predictions vs actual BOM outcomes daily
- Monitor for data drift (KS test on feature distributions)
- A/B test: model vs climatological baseline
- Document all findings in an ML Experiment Log

Stage 2 — Staging Deployment (Weeks 5–8)

#### 🏗 Infrastructure

- Containerise with Docker (Python 3.11 slim base)
- Deploy to AWS EKS staging cluster (1 replica)
- FastAPI serving endpoint: `POST /predict`
- Model artefact stored in S3 + MLflow tracking server
- Feature store synced from BOM API via Airflow DAG

#### 🧪 Staging Checklist

- Integration tests: schema validation on input payloads
- Load test: 500 RPS for 10 minutes (k6)
- Model card review (Fairness, Transparency, Limitations)
- Rollback plan: blue/green deployment with 1-click revert
- CISO sign-off on staging environment controls

Stage 3 — Production (Weeks 9–12)

#### 🚀 Production Architecture

- 3-replica EKS deployment with HPA (CPU ≥ 70%)
- AWS ALB + WAF in front of prediction API
- Secrets in AWS Secrets Manager (no env files)
- DynamoDB for prediction logging (audit trail)
- CloudWatch alarms: latency, error rate, model drift

#### ✅ Go/No-Go Gates

- Two required approvers: ML Lead + CISO representative
- All Staging tests green
- Penetration test completed (no Critical / High open)
- Model card signed by product owner
- Incident response runbook reviewed and rehearsed

Stage 4 — Steady-State Operations (Ongoing)

#### 📈 Structural ROI Signals

- Forecast accuracy vs BOM official: monthly delta
- False negative rate trend (missed rain alerts)
- Feature drift score (PSI \< 0.2 = stable)
- Retrain cadence: quarterly or on drift trigger

#### 🔄 Re-training Triggers

- Recall drops below 50% on 7-day rolling window
- PSI (Population Stability Index) > 0.25 on any feature
- BOM updates measurement methodology or sensors
- Scheduled quarterly refresh regardless of drift

---

9 CISO Governance — Data Security Framework

Working with the Security team to safeguard company and meteorological data throughout the ML lifecycle

⚠ Why ML Pipelines Need Special Security Attention

Unlike traditional software, ML systems carry three unique risks: (1) **training data poisoning** — adversarial inputs corrupt the model, (2) **model inversion attacks** — attackers reconstruct training data from model outputs, (3) **supply chain risk** — third-party libraries in the ML stack (NumPy, Scikit-learn, Pandas) are high-value attack targets.

2.1 — Data Classification & Handling

| Data Element | Classification | Handling Requirement |
| --- | --- | --- |
| Raw BOM weather observations | Internal — Public Source | Stored encrypted at rest (AES-256 via AWS KMS); no PII present |
| Feature-engineered training set | Internal — Confidential | Version-controlled in S3 with object-level encryption; access via IAM role only |
| Trained model artefact (.pkl) | Internal — Confidential | Stored in MLflow registry on S3; signing key rotated every 90 days |
| Prediction logs (inputs + outputs) | Internal — Confidential | DynamoDB TTL = 7 years (regulatory minimum); append-only IAM policy |
| Grid search CV results | Internal — Use Only | MLflow experiment tracking; no external sharing without review |
| API inference inputs (real-time) | Internal — Transient | Never logged in plaintext; schema-validated before processing |

2.2 — Access Control (Principle of Least Privilege)

| Role | Data Access | Model Access | API Access |
| --- | --- | --- | --- |
| ML Engineer | Read training data; write feature store | Read/write MLflow registry (non-prod) | Dev/staging only |
| ML Ops | Read only | Promote models to production registry | All environments |
| Data Scientist | Read training data; read features | Read all experiments; write non-prod | Dev only |
| Prediction Service (SA) | None — receives features at inference time | Read production model artefact only | Production API endpoint |
| CISO Auditor | Read prediction logs (audit) | Read model card; no artefact access | None |
| API Consumer | None | None | POST /predict via API key |

2.3 — ML Supply Chain & Dependency Security

#### 📦 Dependency Controls

- Pin all library versions in `requirements.txt`
- OWASP Dependency-Check in CI pipeline
- Snyk vulnerability scanning on every PR
- Trivy container image scanning (block on HIGH/CRITICAL CVE)
- Private PyPI mirror for approved packages only
- No `pip install` in production containers post-build

#### 🔐 Model Artefact Integrity

- SHA-256 hash of model artefact recorded at training time
- Hash verified at load time before serving predictions
- Model signing with AWS KMS asymmetric key
- Immutable S3 versioning on model bucket (Object Lock)
- Only ML Ops role can promote to production registry
- Promotion requires 2-person review (4-eyes principle)

2.4 — Encryption & Network Security

#### 🔒 Encryption Standards

- Data at rest: AES-256 via AWS KMS (customer-managed key)
- Data in transit: TLS 1.3 enforced; TLS 1.0/1.1 disabled
- Model artefact: AES-256 SSE-KMS on S3
- Prediction logs: DynamoDB encryption with KMS
- Secrets: AWS Secrets Manager (90-day rotation enforced)
- API keys: PBKDF2-SHA256 hashed at rest in Redis

#### 🌐 Network Controls

- Prediction API in private VPC subnet (no public IP)
- AWS WAF: SQL injection, rate limiting (100 RPM/key)
- ALB access logs to S3 (7-year retention)
- BOM data ingestion via NAT Gateway (fixed egress IP)
- No direct database access from inference service
- Kubernetes NetworkPolicy: deny-all + explicit allowlist

2.5 — Responsible AI & Model Governance

#### 📄 Model Card Requirements

1

#### Intended Use

Next-day rainfall prediction for Sydney; not for aviation or emergency services without additional validation

2

#### Limitations

Trained on Sydney only; does not generalise to other climate zones without retraining

3

#### Fairness

No protected characteristics in training data; seasonal bias must be monitored quarterly

4

#### Human Override

All BLOCK-level alerts must have a human meteorologist review path within 2 hours

#### ✅ CISO Sign-Off Checklist

- ☐ Data classification matrix reviewed and approved
- ☐ IAM role matrix reviewed (least privilege confirmed)
- ☐ Model artefact signing configured and tested
- ☐ Penetration test completed (Critical = 0, High = 0)
- ☐ Dependency vulnerability scan: no Critical CVEs
- ☐ Encryption at rest/transit verified by security engineer
- ☐ Prediction audit log append-only policy verified
- ☐ Incident response runbook rehearsed (tabletop exercise)
- ☐ Model card signed by product owner and legal
- ☐ Quarterly drift/bias review cadence documented

2.6 — Incident Response Plan (ML-Specific)

| Incident Type | Trigger | Response Steps | SLA |
| --- | --- | --- | --- |
| Model accuracy degradation | Recall \< 50% on 7-day window | 1. Disable live predictions 2. Root-cause (data drift / concept drift) 3. Retrain 4. Staged re-rollout | Detect \< 1hr; Retrain \< 24hr |
| Data pipeline poisoning | PSI > 0.5 on >3 features simultaneously | 1. Halt ingestion 2. Revert to last clean snapshot 3. Notify CISO 4. Forensic analysis | Halt \< 15min; Report \< 4hr |
| Model artefact tampering | SHA-256 hash mismatch at load | 1. Immediately reject load 2. Page ML Ops 3. Roll back to prior signed artefact 4. CISO incident ticket | Block \< 1sec; Restore \< 30min |
| API abuse / DDoS on inference | WAF rate-limit fired > 10x in 5min | 1. WAF challenge escalation 2. Block originating IP range 3. Notify SOC 4. Review access logs | Block \< 5min; Report \< 2hr |

✅ Governance Summary

The Rainfall Prediction Classifier is cleared for production deployment under the following non-negotiable conditions: (1) all data encrypted at rest and in transit with customer-managed KMS keys; (2) model artefact SHA-256 verified at every load; (3) prediction audit logs written to append-only DynamoDB with 7-year retention; (4) CISO sign-off checklist fully completed before go-live; (5) quarterly model drift and bias review scheduled and owned by ML Ops. This governance framework ensures the system is not only accurate but also auditable, secure, and responsible.

Rainfall Prediction Classifier · Built with Scikit-learn 1.8 · Python 3.11 · Australian BOM Dataset 2008–2017 · Sydney Location \
All company names fictitious · For educational and demonstration purposes