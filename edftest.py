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
# 1. 데이터 로더 (EDF)
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

def load_edf_data(base_dir="archive (5)/files", target_len=768, max_subjects=20):
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
                events, event_id = mne.events_from_annotations(raw)
                win_size = int(raw.info['sfreq'] * 2)  # 2초 윈도우
                for e in events:
                    start = e[0]
                    label = e[2]
                    if label == event_id.get('T0', -1):  # rest → 저의식
                        lab = 0
                    elif label in [event_id.get('T1', -1), event_id.get('T2', -1)]:  # motor imagery → 의식
                        lab = 1
                    else:
                        continue
                    segment = raw.get_data(start=start, stop=start+win_size).astype(np.float32)
                    if segment.shape[1] == win_size:
                        seg_rep = repeat_to_length(segment, target_len)
                        X.append(seg_rep.T)
                        y.append(lab)
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
    X_edf, y_edf = load_edf_data()
    print("EDF 데이터셋 크기:", X_edf.shape, y_edf.shape)

    # Train/Validation/Test Split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_edf, y_edf, test_size=0.4, random_state=42, shuffle=True
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