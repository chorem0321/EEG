import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import chi2_contingency
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

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

    # band ratio (예: theta/alpha, beta/alpha)
    theta_alpha_ratio = theta / (alpha + 1e-6)
    beta_alpha_ratio  = beta / (alpha + 1e-6)

    # spectral entropy
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)

    return [delta, theta, alpha, beta, theta_alpha_ratio, beta_alpha_ratio, spec_entropy]

def label_consciousness(features):
    delta, theta, alpha, beta, _, _, spec_entropy = features
    # 라벨링 기준 강화: δ+θ vs α+β + entropy 조건
    if (delta+theta) >= (alpha+beta) and spec_entropy < 4.0:
        return 0  # 저의식
    else:
        return 1  # 의식

# -----------------------------
# 2. CSV/GDF/EDF 로더
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
# 3. 모델 정의 (MLP)
# -----------------------------
def build_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.pbar = tqdm(total=self.epochs, desc="Training Progress", unit="epoch")
    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        self.pbar.set_postfix({"acc": f"{acc:.2f}", "val_acc": f"{val_acc:.2f}"})
    def on_train_end(self, logs=None):
        self.pbar.close()

# -----------------------------
# 4. 실행
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

    # Train/Validation/Test Split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.4, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )

    # 클래스 불균형 보정
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = dict(zip(np.unique(y_train), class_weights))

    # 모델 학습
    model = build_mlp(X_train.shape[1])
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=cw,
        callbacks=[TQDMProgressBar()]
    )

    # 평가
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # F1-score
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # 카이제곱 검정 (안전 처리)
    cm = confusion_matrix(y_test, y_pred)
    try:
        chi2, p, dof, expected = chi2_contingency(cm)
        print("\nChi-square Test for Independence")
        print("Chi2:", chi2, "p-value:", p)
        if p < 0.05:
            print("→ 모델 예측과 실제 라벨은 통계적으로 유의하게 관련 있음")
        else:
            print("→ 모델 예측과 실제 라벨은 통계적으로 유의하지 않음")
    except ValueError:
        print("\nChi-square Test 불가능 (expected frequency에 0 존재)")