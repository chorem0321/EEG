import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import chi2_contingency
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten
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

    # band ratio
    theta_alpha_ratio = theta / (alpha + 1e-6)
    beta_alpha_ratio  = beta / (alpha + 1e-6)

    # spectral entropy
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
# (이전과 동일, 생략 없이 그대로 사용)

# -----------------------------
# 3. 모델 정의 (CNN + BiLSTM)
# -----------------------------
def build_cnn_bilstm(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
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

    # CNN+BiLSTM 입력 형태 (samples, timesteps=1, features)
    X_train = np.expand_dims(X_train, axis=1)
    X_val   = np.expand_dims(X_val, axis=1)
    X_test  = np.expand_dims(X_test, axis=1)

    # 클래스 불균형 보정
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = dict(zip(np.unique(y_train), class_weights))

    # 모델 학습
    model = build_cnn_bilstm((X_train.shape[1], X_train.shape[2]))
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

    # -----------------------------
    # Recall 우선 threshold 탐색
    # -----------------------------
    y_prob = model.predict(X_test).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # recall을 최우선으로 → recall이 가장 높은 지점 선택
    best_idx = np.argmax(recalls)
    best_threshold = thresholds[best_idx]

    print(f"\n최적 threshold (Recall 기준): {best_threshold:.3f}")
    print(f"최대 Recall: {recalls[best_idx]:.4f}, Precision: {precisions[best_idx]:.4f}")

    # 최적 threshold 적용
    y_pred_opt = (y_prob > best_threshold).astype("int32")
    print("\nClassification Report (Optimal Threshold - Recall 기준):")
    print(classification_report(y_test, y_pred_opt, digits=4, zero_division=0))

    # Confusion Matrix + Chi-square
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    print("\nConfusion Matrix (Optimal Threshold - Recall 기준):\n", cm_opt)

    try:
        chi2, p, dof, expected = chi2_contingency(cm_opt)
        print("\nChi-square Test for Independence")
        print("Chi2:", chi2, "p-value:", p)
        if p < 0.05:
            print("→ 모델 예측과 실제 라벨은 통계적으로 유의하게 관련 있음")
        else:
            print("→ 모델 예측과 실제 라벨은 통계적으로 유의하지 않음")
    except ValueError:
        print("\nChi-square Test 불가능 (expected frequency에 0 존재)")