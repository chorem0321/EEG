import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.stats import entropy, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# -----------------------------
# 1. 특징 추출 함수
# -----------------------------
def compute_features(signal, sfreq):
    freqs, psd = welch(signal, sfreq, nperseg=sfreq*2)
    def band_power(low, high):
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.sum(psd[idx])
    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta  = band_power(13, 30)

    theta_alpha_ratio = theta / (alpha + 1e-6)
    beta_alpha_ratio  = beta / (alpha + 1e-6)

    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)

    return [delta, theta, alpha, beta, theta_alpha_ratio, beta_alpha_ratio, spec_entropy]

def label_consciousness(features):
    delta, theta, alpha, beta, _, _, spec_entropy = features
    if (delta+theta) >= (alpha+beta) and spec_entropy < 4.0:
        return 0
    else:
        return 1

# -----------------------------
# 2. 데이터 로더 (CSV/GDF/EDF)
# -----------------------------
def load_csv_data(base_dir="archive (1)/Dataset"):
    X, y = [], []
    for user in ["user_a.csv", "user_b.csv", "user_c.csv", "user_d.csv"]:
        df = pd.read_csv(os.path.join(base_dir, user))
        signals = df.values
        sfreq = 128
        win_size = 256
        for i in range(0, len(signals)-win_size, win_size):
            seg = signals[i:i+win_size, :].T
            features = compute_features(seg[0], sfreq)
            lab = label_consciousness(features)
            X.append(features)
            y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y)

def load_gdf_data(base_dir="archive (3)/BCICIV_2a_gdf"):
    X, y = [], []
    for subj in range(1, 10):
        for sess in ["E", "T"]:
            fname = f"A{subj:02d}{sess}.gdf"
            file_path = os.path.join(base_dir, fname)
            if not os.path.exists(file_path):
                continue
            raw = mne.io.read_raw_gdf(file_path, preload=True)
            raw.resample(128)
            sfreq = int(raw.info['sfreq'])
            data = raw.get_data()
            events, _ = mne.events_from_annotations(raw)
            win_size = int(sfreq * 2)
            for e in events:
                start = e[0]
                for offset in range(0, int(sfreq*7.5)-win_size, win_size):
                    seg = data[:, start+offset:start+offset+win_size]
                    if seg.shape[1] < win_size:
                        continue
                    features = compute_features(seg[0], sfreq)
                    lab = label_consciousness(features)
                    X.append(features)
                    y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y)

def load_edf_data(base_dir="archive (5)/files", max_subjects=20):
    X, y = [], []
    for subj in range(1, max_subjects+1):
        subj_id = f"S{subj:03d}"
        subj_path = os.path.join(base_dir, subj_id)
        if not os.path.exists(subj_path):
            continue
        for fname in os.listdir(subj_path):
            if fname.endswith(".edf"):
                file_path = os.path.join(subj_path, fname)
                raw = mne.io.read_raw_edf(file_path, preload=True)
                raw.resample(128)
                sfreq = int(raw.info['sfreq'])
                win_size = int(sfreq * 2)
                data = raw.get_data()
                for start in range(0, data.shape[1]-win_size, win_size):
                    seg = data[:, start:start+win_size]
                    features = compute_features(seg[0], sfreq)
                    lab = label_consciousness(features)
                    X.append(features)
                    y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y)

# -----------------------------
# 3. 실행 (Random Forest vs XGBoost)
# -----------------------------
if __name__ == "__main__":
    # 데이터 불러오기
    X_csv, y_csv = load_csv_data()
    X_gdf, y_gdf = load_gdf_data()
    X_edf, y_edf = load_edf_data()

    # 통합
    X_all = np.concatenate([X_csv, X_gdf, X_edf], axis=0)
    y_all = np.concatenate([y_csv, y_gdf, y_edf], axis=0)
    print("전체 데이터셋 크기:", X_all.shape, y_all.shape)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, shuffle=True
    )

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n=== Random Forest 결과 ===")
    print(classification_report(y_test, y_pred_rf, digits=4, zero_division=0))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("Confusion Matrix:\n", cm_rf)
    try:
        chi2, p, dof, expected = chi2_contingency(cm_rf)
        print("Chi-square Test: χ² =", chi2, ", p-value =", p)
    except ValueError:
        print("Chi-square Test 불가능 (expected frequency에 0 존재)")

    # -----------------------------
    # XGBoost
    # -----------------------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train)/np.sum(y_train==1),
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    print("\n=== XGBoost 결과 ===")
    print(classification_report(y_test, y_pred_xgb, digits=4, zero_division=0))
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print("Confusion Matrix:\n", cm_xgb)
    try:
        chi2, p, dof, expected = chi2_contingency(cm_xgb)
        print("Chi-square Test: χ² =", chi2, ", p-value =", p)
    except ValueError:
        print("Chi-square Test 불가능 (expected frequency에 0 존재)")