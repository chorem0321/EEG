import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# -----------------------------
# 1. CSV 데이터 로더
# -----------------------------
def repeat_to_length(segment, target_len):
    if segment.shape[1] == target_len:
        return segment
    reps = target_len // segment.shape[1]
    remainder = target_len % segment.shape[1]
    seg_repeated = np.tile(segment, (1, reps))
    if remainder > 0:
        seg_repeated = np.concatenate([seg_repeated, segment[:, :remainder]], axis=1)
    return seg_repeated

def load_csv_data(base_dir="archive (1)/Dataset", target_len=768):
    X, y = [], []
    for user in ["user_a.csv", "user_b.csv", "user_c.csv", "user_d.csv"]:
        df = pd.read_csv(os.path.join(base_dir, user))
        low_waves = ["delta", "theta"]
        high_waves = ["alpha", "beta"]
        labels = []
        for idx, row in df.iterrows():
            low_sum = sum(row[col] for col in df.columns if any(w in col for w in low_waves))
            high_sum = sum(row[col] for col in df.columns if any(w in col for w in high_waves))
            labels.append(0 if low_sum >= high_sum else 1)
        df["label"] = labels

        win_size = 256
        signals = df.drop(columns=["label"]).values
        labels = df["label"].values
        for i in range(0, len(signals)-win_size, win_size):
            seg = signals[i:i+win_size, :].T
            seg_rep = repeat_to_length(seg, target_len)
            X.append(seg_rep.T)
            y.append(int(np.round(labels[i:i+win_size].mean())))
    return np.array(X, dtype=np.float32), np.array(y)

# -----------------------------
# 2. 모델 정의
# -----------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# -----------------------------
# 3. 학습 퍼센티지 표시용 Callback
# -----------------------------
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
    print("CSV 데이터셋 크기:", X_csv.shape, y_csv.shape)

    # Train/Validation/Test Split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_csv, y_csv, test_size=0.4, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )

    # 모델 학습
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm(input_shape)

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[TQDMProgressBar()]
    )

    # 평가
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # -----------------------------
    # 5. F1-score 계산
    # -----------------------------
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, digits=4))

    # -----------------------------
    # 6. 적합성 검정 (카이제곱 테스트)
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    chi2, p, dof, expected = chi2_contingency(cm)
    print("\nChi-square Test for Independence")
    print("Chi2:", chi2, "p-value:", p)
    if p < 0.05:
        print("→ 모델 예측과 실제 라벨은 통계적으로 유의하게 관련 있음")
    else:
        print("→ 모델 예측과 실제 라벨은 통계적으로 유의하지 않음")