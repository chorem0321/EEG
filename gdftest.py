import os
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# -----------------------------
# 1. 데이터 로더 (GDF)
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

def load_gdf_data(base_dir="archive (3)/BCICIV_2a_gdf", target_len=768):
    X, y = [], []
    for subj in range(1, 10):  # subject 1~9
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

            trial_len = int(sfreq * 7.5)
            for e in events:
                start = e[0]
                trial = data[:, start:start+trial_len]
                if trial.shape[1] < trial_len:
                    continue

                # high segment (0~6s → 의식)
                s = int(0*sfreq); e = int(6*sfreq)
                seg_high = trial[:, s:e]
                if seg_high.shape[1] == target_len:
                    X.append(seg_high.T); y.append(1)

                # low segment (6~7.5s → 저의식)
                s = int(6*sfreq); e = int(7.5*sfreq)
                seg_low = trial[:, s:e]
                seg_low_rep = repeat_to_length(seg_low, target_len)
                X.append(seg_low_rep.T); y.append(0)
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
    X_gdf, y_gdf = load_gdf_data()
    print("GDF 데이터셋 크기:", X_gdf.shape, y_gdf.shape)

    # Train/Validation/Test Split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_gdf, y_gdf, test_size=0.4, random_state=42, shuffle=True
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
    from sklearn.metrics import classification_report
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