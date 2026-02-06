import os
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# -----------------------------
# 공통 함수: 길이 맞추기
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

# -----------------------------
# PCA 적용 함수
# -----------------------------
def apply_pca(X, target_dim=14):
    n_trials, timesteps, n_channels = X.shape
    # timesteps마다 채널 차원만 줄이기
    X_pca = []
    pca = PCA(n_components=target_dim)
    for trial in X:
        # trial shape: (timesteps, channels)
        trial_pca = pca.fit_transform(trial)
        X_pca.append(trial_pca)
    return np.array(X_pca, dtype=np.float32)

# -----------------------------
# 1. CSV 데이터
# -----------------------------
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
# 2. GDF 데이터
# -----------------------------
def load_gdf_data(base_dir="archive (3)/BCICIV_2a_gdf", target_len=768):
    X, y = [], []
    for subj in range(1, 10):
        for sess in ["E", "T"]:
            fname = f"A{subj:02d}{sess}.gdf"
            file_path = os.path.join(base_dir, fname)
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

                # high segment
                s = int(0*sfreq); e = int(6*sfreq)
                seg_high = trial[:, s:e]
                if seg_high.shape[1] == target_len:
                    X.append(seg_high.T); y.append(1)

                # low segment
                s = int(6*sfreq); e = int(7.5*sfreq)
                seg_low = trial[:, s:e]
                seg_low_rep = repeat_to_length(seg_low, target_len)
                X.append(seg_low_rep.T); y.append(0)
    return np.array(X, dtype=np.float32), np.array(y)

# -----------------------------
# 3. EDF 데이터 (일부 샘플링)
# -----------------------------
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
                win_size = int(raw.info['sfreq'] * 2)
                for e in events:
                    start = e[0]
                    label = e[2]
                    if label == event_id.get('T0', -1):
                        lab = 0
                    elif label in [event_id.get('T1', -1), event_id.get('T2', -1)]:
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
# 4. 데이터 통합 (PCA 적용)
# -----------------------------
X_csv, y_csv = load_csv_data()
X_gdf, y_gdf = load_gdf_data()
X_edf, y_edf = load_edf_data()

X_csv_pca = apply_pca(X_csv, target_dim=14)
X_gdf_pca = apply_pca(X_gdf, target_dim=14)
X_edf_pca = apply_pca(X_edf, target_dim=14)

X_all = np.concatenate([X_csv_pca, X_gdf_pca, X_edf_pca], axis=0)
y_all = np.concatenate([y_csv, y_gdf, y_edf], axis=0)

print("통합 데이터셋 크기:", X_all.shape, y_all.shape)

# -----------------------------
# 5. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# -----------------------------
# 6. 모델 정의 함수들
# -----------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_mlp(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# 7. 학습 퍼센티지 표시용 Callback
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
# 8. 모델 학습 및 비교 (퍼센티지 표시)
# -----------------------------
input_shape = (X_train.shape[1], X_train.shape[2])

models = {
    "LSTM": build_lstm(input_shape),
    "GRU": build_gru(input_shape),
    "CNN": build_cnn(input_shape),
    "MLP": build_mlp(input_shape)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(
        X_train.astype(np.float32), y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test.astype(np.float32), y_test),
        callbacks=[TQDMProgressBar()]  # 퍼센티지 표시
    )
    loss, acc = model.evaluate(X_test.astype(np.float32), y_test)
    results[name] = acc

print("\n모델별 정확도 비교:", results)