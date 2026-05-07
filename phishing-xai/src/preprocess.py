"""
Preprocessing pipeline for the PhiUSIIL Phishing URL Dataset.

Raw dataset: 235,795 rows, 56 columns
Dropped (raw strings / identifiers): FILENAME, URL, Domain, TLD, Title
Retained: 50 numerical features + label
No missing values in this dataset; imputer kept for pipeline robustness.

Split: stratified 70 / 15 / 15  (train / val / test)
Scaling: Min-Max to [0, 1]
SMOTE: applied only if minority class < 40% of training set
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── constants ────────────────────────────────────────────────────────────────
DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
MODELS_DIR = "models"
TARGET_COL = "label"
IMBALANCE_THRESHOLD = 0.40   # minority ratio below this → apply SMOTE

# Raw string columns that carry no learnable structure after dropping URL
DROP_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]


# ── helpers ──────────────────────────────────────────────────────────────────
def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV with BOM-safe encoding."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[load]  {len(df):,} rows × {df.shape[1]} columns")
    return df


# ── main pipeline ────────────────────────────────────────────────────────────
def preprocess(
    path: str = DATA_PATH,
    smote: bool = True,
    save_artifacts: bool = True,
    models_dir: str = MODELS_DIR,
) -> tuple:
    """
    Run the full preprocessing pipeline.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  (scaled)
    y_train, y_val, y_test : np.ndarray
    feature_names          : list[str]
    """
    df = load_raw(path)

    # Drop non-feature columns (keep only what the model can use)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    feature_names: list[str] = X.columns.tolist()

    dist = y.value_counts().to_dict()
    print(f"[data]  {len(feature_names)} features | label dist: {dist}")

    # ── stratified 70 / 15 / 15 split (split first to prevent leakage) ──────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # ── mean imputation (fit on train only) ──────────────────────────────────
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    # ── Min-Max scaling to [0, 1] (fit on train only) ────────────────────────
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── SMOTE on training set if imbalance detected ──────────────────────────
    minority_ratio = np.bincount(y_train).min() / len(y_train)
    if smote and minority_ratio < IMBALANCE_THRESHOLD:
        from imblearn.over_sampling import SMOTE  # training-only dep
        print(f"[smote] minority={minority_ratio:.2%} < {IMBALANCE_THRESHOLD:.0%} → applying SMOTE")
        sm = SMOTE(random_state=42, n_jobs=-1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[smote] after resampling: {np.bincount(y_train.astype(int))}")
    else:
        print(f"[smote] minority={minority_ratio:.2%} ≥ threshold → skipped")

    print(
        f"[split] train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
    )

    # ── persist preprocessing artifacts ─────────────────────────────────────
    if save_artifacts:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(imputer,       f"{models_dir}/imputer.pkl")
        joblib.dump(scaler,        f"{models_dir}/scaler.pkl")
        joblib.dump(feature_names, f"{models_dir}/feature_names.pkl")
        print(f"[save]  preprocessing artifacts → {models_dir}/")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def transform_single(raw_features: dict, models_dir: str = MODELS_DIR) -> np.ndarray:
    """
    Transform a single URL's feature dict into a model-ready array.
    Used at inference time by the API / Streamlit app.
    """
    feature_names: list[str] = joblib.load(f"{models_dir}/feature_names.pkl")
    imputer: SimpleImputer    = joblib.load(f"{models_dir}/imputer.pkl")
    scaler: MinMaxScaler      = joblib.load(f"{models_dir}/scaler.pkl")

    row = pd.DataFrame([{f: raw_features.get(f, np.nan) for f in feature_names}])
    row = imputer.transform(row)
    row = scaler.transform(row)
    return row


if __name__ == "__main__":
    preprocess()