"""
Implementasi metode paper:
    You, S.D. "Classification of Relaxation and Concentration Mental States with EEG."
    Information 2021, 12, 187.  (Document from Ezra.pdf)

Semua langkah mengikuti paper, kecuali yang diberi tanda:
  [no-jurnal | tambahan]  -> ditambahkan di luar jurnal (permintaan proyek)
  [no-jurnal | terpaksa]  -> jurnal tidak menjelaskan / alatnya Muse, sehingga harus diisi sendiri

Ringkasan metode paper
----------------------
* Data per rekaman = spektrogram energi: resolusi waktu 0.125 s (8 baris/detik),
  resolusi frekuensi 0.25 Hz (4 kolom/Hz), 0-64 Hz  (Sec. 3.2).
* Satu rekaman 10 s: 5 s pertama baseline rileks, 5 s berikutnya tugas
  (hitung mundur 5 digit dalam hati = konsentrasi, atau tetap rileks).
  Yang dipakai untuk klasifikasi = detik ke-6 s/d ke-8 (3 s)  (Sec. 3.3).
* Persamaan (1):  y[m,l] = 1/8 * sum_{n=8m}^{8m+7} x[n, 4l]
  m = detik (0,1,2 dari segmen 3 s), l = frekuensi (Hz, bilangan bulat).
* Jenis fitur (Sec. 3.3):
    (i)   All band energy : theta, alpha, beta_1 (13-20), beta_2 (21-28), gamma -> 3x5 = 15
    (ii)  beta only       : 13-28 Hz, bandwidth 1/2/4 Hz
    (iii) alpha+beta      : 9-28 Hz,  bandwidth 1/2/4 Hz
    (iv)  alpha+beta+gamma: 9-43 Hz,  bandwidth 1/2/4 Hz
* Classifier (Sec. 3.4): SVM RBF gamma=8, C=32 (LIBSVM); BPNN 3 layer, 10 hidden, MSE.
* Evaluasi (Sec. 3.3): per subjek 20 con + 20 relax, 70% latih / 30% uji acak,
  diulang 40 trial, split yang sama dipakai semua metode dalam satu trial.
* Factor Analysis / PCFA (Sec. 4.3, Pers. 3-6) pada fitur beta 1 Hz, p = 8/6/4.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Konstanta dari paper
# ---------------------------------------------------------------------------
ROWS_PER_SEC = 8            # resolusi waktu 0.125 s
BINS_PER_HZ = 4             # resolusi frekuensi 0.25 Hz
N_FREQ_BINS = 256           # 0 .. 63.75 Hz
SEG_START_SEC = 5           # detik ke-6 dimulai pada t = 5 s
SEG_LEN_SEC = 3             # detik ke-6, 7, 8
SVM_C = 32
SVM_GAMMA = 8
BPNN_HIDDEN = 10
N_PER_STATE = 20            # 20 rekaman per kondisi per subjek
TRAIN_RATIO = 0.7           # 70 % latih
N_TRIALS = 40               # diulang 40 kali

# Rentang band untuk fitur (i) "All band energy".
# beta_1 dan beta_2 tertulis eksplisit di paper (l = 13..20 dan 21..28).
# [no-jurnal | terpaksa] theta/alpha/gamma mengikuti definisi band di Sec. 1 paper:
#   theta 4-7 Hz, alpha 8-12 Hz, gamma > 30 Hz (batas atas 43 Hz = batas fitur iv).
ALL_BANDS = {
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta_1": (13, 20),
    "beta_2": (21, 28),
    "gamma": (30, 43),
}
# Rentang frekuensi (inklusif, Hz) fitur (ii)-(iv)
BAND_RANGES = {
    "beta": (13, 28),
    "alpha+beta": (9, 28),
    "alpha+beta+gamma": (9, 43),
}

FEATURE_SETS = (
    ["all_band"]
    + [f"{name}|{bw}Hz" for name in BAND_RANGES for bw in (1, 2, 4)]
)
RECOMMENDED_FEATURE = "alpha+beta+gamma|4Hz"   # rekomendasi kesimpulan paper (Sec. 5)


# ---------------------------------------------------------------------------
# 1. Membaca dataset GitHub NTUT-LabASPL/EEG (Run_1 / Run_2)
# ---------------------------------------------------------------------------
def read_powerspec_csv(path: str) -> np.ndarray:
    """Baca satu file *.powerspec.csv -> array (n_baris_waktu, 256).

    Run_1: tanpa header, 256 kolom energi.
    Run_2: header 'Time, 0, 0.25, ...' + kolom jam di depan -> kolom Time dibuang.
    """
    with open(path, "r", errors="replace") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    has_header = lines[0].lstrip().startswith("Time")
    if has_header:
        lines = lines[1:]
    rows = []
    for ln in lines:
        parts = ln.split(",")
        if has_header or len(parts) == N_FREQ_BINS + 1:
            parts = parts[1:]
        if all(not p.strip() for p in parts):      # baris kosong ",,,,," di akhir file (Run_1/yo_con)
            continue
        rows.append([float(p) if p.strip() else np.nan for p in parts[:N_FREQ_BINS]])
    return np.asarray(rows, dtype=np.float64)


def _sort_key(path: str):
    stem = os.path.basename(path).split(".")[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def load_run(run_dir: str, n_per_state: int = N_PER_STATE) -> dict:
    """Muat semua subjek dalam satu Run.

    Return: {subjek: {"con": [spektrogram...], "relax": [...], "files_con": [...], ...}}
    Paper memakai 20 rekaman per kondisi. Bila folder berisi lebih dari 20 file
    (mis. yizhe = 40, beberapa subjek Run_2 = 21-22), diambil 20 pertama sesuai urutan
    nama file (urutan waktu rekam).  [no-jurnal | terpaksa]
    """
    data: dict = {}
    for state_dir in sorted(glob.glob(os.path.join(run_dir, "*_*"))):
        if not os.path.isdir(state_dir):
            continue
        name = os.path.basename(state_dir)
        m = re.match(r"(.+)_(con|relax)$", name)
        if not m:
            continue
        subj, state = m.group(1), m.group(2)
        files = glob.glob(os.path.join(state_dir, "**", "*.csv"), recursive=True)
        files = sorted(files, key=_sort_key)[:n_per_state]
        d = data.setdefault(subj, {})
        d[state] = [read_powerspec_csv(f) for f in files]
        d["files_" + state] = files
    return data


# ---------------------------------------------------------------------------
# 2. Ekstraksi fitur (Pers. 1 + jenis fitur i-iv)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1b. [no-jurnal | tambahan] Penyamaan skala dataset <-> Muse (relative power)
# ---------------------------------------------------------------------------
# NeuroSky: satuan energi internal (tidak terdokumentasi). Muse: µV^2/Hz.
# Supaya kedua alat berada pada skala yang SAMA, energi tiap baris waktu dibagi total
# energi 1-45 Hz pada baris itu (relative power, tanpa satuan). Rentang 1-45 Hz =
# bandpass Muse di pengambilan_eeg.py, jadi data NeuroSky "diproses seperti Muse".
# Setelah ini, Pers. (1), fitur, SVM, dan evaluasi tetap persis seperti paper.
# [no-jurnal | tambahan] - jurnal memakai energi asli alat.
NORMALIZATIONS = ("none", "relative")
REL_BAND_HZ = (1.0, 45.0)


def normalize_spectrogram(spec: np.ndarray, norm: str = "none") -> np.ndarray:
    """none     : energi asli (sesuai jurnal)
    relative : x[n,k] / sum_{1<=f<45Hz} x[n,k]   [no-jurnal | tambahan] skala Muse untuk dataset & Muse
    """
    x = np.nan_to_num(np.asarray(spec, dtype=np.float64))
    if norm == "none":
        return x
    if norm == "relative":
        lo, hi = (int(f * BINS_PER_HZ) for f in REL_BAND_HZ)
        tot = x[:, lo:hi].sum(axis=1, keepdims=True)
        return x / np.where(tot > 0, tot, 1.0)
    raise ValueError(f"norm harus salah satu {NORMALIZATIONS}")


def segment_y(spec: np.ndarray, start_sec: int = SEG_START_SEC) -> np.ndarray:
    """Pers. (1): y[m, l] = 1/8 * sum_{n=8m}^{8m+7} x[n, 4l],  m = 0..2, l = 0..63.

    Indeks n dihitung dari awal segmen detik ke-6 (baris 40 s/d 63 rekaman).
    """
    start = start_sec * ROWS_PER_SEC
    stop = start + SEG_LEN_SEC * ROWS_PER_SEC
    if spec.shape[0] < stop:
        raise ValueError(f"Rekaman terlalu pendek: {spec.shape[0]} baris, perlu >= {stop}")
    seg = spec[start:stop]                                   # (24, 256)
    x_int_hz = seg[:, ::BINS_PER_HZ]                         # kolom 4l -> l Hz (64 kolom)
    return x_int_hz.reshape(SEG_LEN_SEC, ROWS_PER_SEC, -1).mean(axis=1)   # (3, 64)


def _group_average(v: np.ndarray, bw: int) -> np.ndarray:
    """Rata-rata energi per bandwidth bw Hz (kelompok terakhir boleh tidak penuh)."""
    return np.array([v[i:i + bw].mean() for i in range(0, len(v), bw)])


def features_from_y(y: np.ndarray, feature_set: str) -> np.ndarray:
    """y (3, 64) -> vektor fitur 1 rekaman (detik-detik digabung berurutan)."""
    out = []
    for m in range(y.shape[0]):
        row = y[m]
        if feature_set == "all_band":
            out.append([row[lo:hi + 1].sum() for lo, hi in ALL_BANDS.values()])
        else:
            name, bw = feature_set.split("|")
            lo, hi = BAND_RANGES[name]
            out.append(_group_average(row[lo:hi + 1], int(bw.replace("Hz", ""))))
    return np.concatenate([np.asarray(o, dtype=np.float64) for o in out])


def extract_features(spec: np.ndarray, feature_set: str = RECOMMENDED_FEATURE,
                     norm: str = "none", start_sec: int = SEG_START_SEC) -> np.ndarray:
    return features_from_y(segment_y(normalize_spectrogram(spec, norm), start_sec), feature_set)


# ---------------------------------------------------------------------------
# 3. Classifier (Sec. 3.4)
# ---------------------------------------------------------------------------
def make_svm() -> Pipeline:
    """SVM RBF, gamma = 8, C = 32 (paper, LIBSVM).

    [no-jurnal | terpaksa] Paper memakai LIBSVM tetapi tidak menyebut penskalaan. Tanpa penskalaan,
    gamma = 8 pada energi mentah membuat kernel RBF ~0 (akurasi ~55 %), jadi fitur
    diskalakan min-max ke [0, 1] (svm-scale -l 0 -u 1, panduan LIBSVM). Dari uji
    [-1,1], [0,1], standar, dan mentah, [0,1] paling mendekati angka paper.
    Skala di-fit hanya pada data latih.
    """
    return Pipeline([
        ("scale", MinMaxScaler(feature_range=(0, 1))),
        ("svm", SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)),
    ])


class BPNNClassifier(BaseEstimator, ClassifierMixin):
    """Feedforward 3 layer (input-10 hidden-1 output), backprop, kriteria MSE (paper).

    Target 0 (relax) / 1 (con), keluaran >= 0.5 -> con.
    """

    def __init__(self, hidden=BPNN_HIDDEN, max_iter=2000, random_state=0):
        self.hidden = hidden
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.net_ = MLPRegressor(hidden_layer_sizes=(self.hidden,), activation="logistic",
                                 solver="lbfgs", max_iter=self.max_iter,
                                 random_state=self.random_state)
        self.net_.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return (self.net_.predict(X) >= 0.5).astype(int)


def make_bpnn(random_state=0) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler(feature_range=(0, 1))),
        ("bpnn", BPNNClassifier(random_state=random_state)),
    ])


# ---------------------------------------------------------------------------
# 4. Factor Analysis / PCFA (Sec. 4.3, Pers. 3-6)
# ---------------------------------------------------------------------------
class PCFAReducer(BaseEstimator):
    """Reduksi dimensi PCFA sesuai Pers. (3)-(6).

    Input fitur 'beta|1Hz' (3 detik x 16 nilai). Setiap detik = satu vektor b_i (16).
    C = 1/N * sum b_i b_i^T ; ambil p eigen terbesar ; L = [sqrt(l1) e1 ... sqrt(lp) ep]
    f = (L^T L)^-1 L^T t ; tiga f (3 detik) digabung -> 3p dimensi.

    Paper: "FA dilakukan per subjek pada satu kondisi (konsentrasi atau relaksasi)"
    dengan 14 rekaman latih x 3 = 42 vektor. -> fit_state menentukan kondisi yang dipakai.
    [no-jurnal | terpaksa] default kondisi konsentrasi (label 1).
    """

    def __init__(self, p=8, per_sec_dim=16, fit_state=1):
        self.p = p
        self.per_sec_dim = per_sec_dim
        self.fit_state = fit_state

    def fit(self, X, y):
        X = np.asarray(X)
        if y is not None and self.fit_state is not None:
            X = X[np.asarray(y) == self.fit_state]
        B = X.reshape(-1, self.per_sec_dim)                     # (14*3=42, 16)
        C = (B.T @ B) / B.shape[0]                              # Pers. (4)
        lam, E = np.linalg.eigh(C)
        idx = np.argsort(lam)[::-1][: self.p]
        L = E[:, idx] * np.sqrt(np.clip(lam[idx], 0, None))     # Pers. (5)
        self.proj_ = np.linalg.pinv(L.T @ L) @ L.T              # Pers. (6)
        return self

    def transform(self, X):
        X = np.asarray(X)
        T = X.reshape(X.shape[0], -1, self.per_sec_dim)         # (N, 3, 16)
        return (T @ self.proj_.T).reshape(X.shape[0], -1)       # (N, 3p)


class FASVM(BaseEstimator, ClassifierMixin):
    """PCFA (pakai label latih untuk memilih kondisi) -> SVM paper."""

    def __init__(self, p=8, fit_state=1):
        self.p = p
        self.fit_state = fit_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fa_ = PCFAReducer(p=self.p, fit_state=self.fit_state).fit(X, y)
        self.svm_ = make_svm().fit(self.fa_.transform(X), y)
        return self

    def predict(self, X):
        return self.svm_.predict(self.fa_.transform(X))


# ---------------------------------------------------------------------------
# 5. Protokol evaluasi (70/30, 40 trial, split sama untuk semua metode)
# ---------------------------------------------------------------------------
def trial_splits(n_per_state: int = N_PER_STATE, n_trials: int = N_TRIALS, seed: int = 0):
    """Hasilkan indeks (train, test) untuk 40 trial.

    Label: index 0..n-1 = con (1), n..2n-1 = relax (0). 70 % tiap kondisi = 14 latih,
    6 uji (paper: 14 rekaman latih per kondisi -> 28 latih per subjek).
    """
    rng = np.random.default_rng(seed)
    n_tr = int(round(TRAIN_RATIO * n_per_state))
    out = []
    for _ in range(n_trials):
        pc = rng.permutation(n_per_state)
        pr = rng.permutation(n_per_state) + n_per_state
        out.append((np.r_[pc[:n_tr], pr[:n_tr]], np.r_[pc[n_tr:], pr[n_tr:]]))
    return out


def confusion(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def metrics_from_counts(tp, tn, fp, fn):
    """Sensitivity = TP/P (con positif), Specificity = TN/N, Accuracy, F1 = 2TP/(2TP+FP+FN)."""
    p, n = tp + fn, tn + fp
    return {
        "sensitivity": 100 * tp / p if p else np.nan,
        "specificity": 100 * tn / n if n else np.nan,
        "accuracy": 100 * (tp + tn) / (p + n),
        "f1": 100 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
    }


def subject_matrix(subj_data: dict, feature_set: str, norm: str = "none",
                   start_sec: int = SEG_START_SEC):
    """X (40, d), y (40,) : 20 con (1) lalu 20 relax (0)."""
    X = [extract_features(s, feature_set, norm, start_sec) for s in subj_data["con"]]
    X += [extract_features(s, feature_set, norm, start_sec) for s in subj_data["relax"]]
    y = np.r_[np.ones(len(subj_data["con"]), int), np.zeros(len(subj_data["relax"]), int)]
    return np.vstack(X), y


def evaluate_individual(X, y, estimator, splits):
    """Model individual: latih & uji per subjek, akumulasi TP/TN/FP/FN atas 40 trial."""
    acc_trials, counts = [], np.zeros(4, int)
    for tr, te in splits:
        est = clone(estimator).fit(X[tr], y[tr])
        c = confusion(y[te], est.predict(X[te]))
        counts += c
        acc_trials.append(metrics_from_counts(*c)["accuracy"])
    m = metrics_from_counts(*counts)
    m["accuracy_trial_std"] = float(np.std(acc_trials))
    return m


# ---------------------------------------------------------------------------
# 6. Objek model yang disimpan ke .pkl
# ---------------------------------------------------------------------------
@dataclass
class PaperEEGModel:
    """Model siap pakai: spektrogram (baris 0.125 s x 256 bin 0.25 Hz) -> con/relax.

    predict_spectrogram() menerima output dataset NeuroSky ATAU output
    muse_raw_to_spectrogram() (format sama).
    """
    estimator: object
    feature_set: str = RECOMMENDED_FEATURE
    device: str = "muse"
    trained_on: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    labels: tuple = ("relax", "con")
    norm: str = "relative"      # [no-jurnal | tambahan] skala Muse (relative power)

    def features(self, spec):
        return extract_features(np.asarray(spec, dtype=float), self.feature_set, self.norm)

    def predict_spectrogram(self, spec) -> str:
        return self.labels[int(self.estimator.predict(self.features(spec)[None, :])[0])]

    def predict_many(self, specs) -> list:
        X = np.vstack([self.features(s) for s in specs])
        return [self.labels[int(p)] for p in self.estimator.predict(X)]


# ---------------------------------------------------------------------------
# 7. [no-jurnal | terpaksa] Konversi sinyal mentah Muse (mikrovolt) -> spektrogram format paper
# ---------------------------------------------------------------------------
MUSE_FS = 256
MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
# [no-jurnal | terpaksa] NeuroSky pada paper: sensor di dahi FP1/FP2, 1 kanal keluaran.
# Kanal Muse yang paling dekat posisi tersebut = AF7 & AF8 (dahi). Energi kedua kanal
# dirata-rata menjadi satu kanal, sama seperti paper yang hanya memakai 1 kanal.
MUSE_FRONTAL = ["AF7", "AF8"]


def muse_preprocess(raw_uv: np.ndarray, fs: int = MUSE_FS) -> np.ndarray:
    """Filter sama dengan pengambilan_eeg.py: bandpass 1-45 Hz (Butterworth orde 2)
    + bandstop 48-52 Hz (notch listrik 50 Hz, orde 2).

    [no-jurnal | terpaksa] Paper memakai software NeuroSky yang tidak diketahui filternya. Filter ini
    tidak mengubah rentang fitur paper (maks 43 Hz). Offline memakai zero-phase
    (sosfiltfilt) agar tidak ada pergeseran waktu terhadap segmen detik ke-6 s/d 8.
    """
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(raw_uv, dtype=np.float64)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x)
    sos_bp = butter(2, [1.0, 45.0], btype="bandpass", fs=fs, output="sos")
    sos_bs = butter(2, [48.0, 52.0], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos_bs, sosfiltfilt(sos_bp, x, axis=0), axis=0)


def muse_raw_to_spectrogram(raw_uv: np.ndarray, fs: int = MUSE_FS,
                            channel_names=None, use_channels=MUSE_FRONTAL,
                            win_sec: float = 1.0) -> np.ndarray:
    """Sinyal mentah Muse (n_sampel x n_kanal, µV) -> spektrogram (n_baris, 256).

    Format keluaran IDENTIK dengan file dataset paper:
      * satu baris tiap 0.125 s  (hop = fs/8 = 32 sampel)
      * 256 kolom = 0, 0.25, ..., 63.75 Hz  (nfft = 4*fs = 1024 -> bin 0.25 Hz)
      * nilai = energi |FFT|^2 (µV^2/Hz, PSD Hann)
    [no-jurnal | terpaksa] Software NeuroSky tidak mendokumentasikan panjang jendelanya. Dipakai jendela
    Hann 1 s (fitur paper per detik) mulai pada t = n*0.125 s, zero-padded ke 0.25 Hz.
    """
    from scipy.signal import stft
    raw_uv = np.asarray(raw_uv, dtype=np.float64)
    if raw_uv.ndim == 1:
        raw_uv = raw_uv[:, None]
        channel_names = channel_names or ["CH"]
        use_channels = None
    channel_names = list(channel_names or MUSE_CHANNELS[: raw_uv.shape[1]])
    if use_channels:
        idx = [channel_names.index(c) for c in use_channels]
        raw_uv = raw_uv[:, idx]
    x = muse_preprocess(raw_uv, fs)
    nperseg = int(win_sec * fs)
    hop = fs // ROWS_PER_SEC
    nfft = fs * BINS_PER_HZ
    _, _, Z = stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg - hop,
                   nfft=nfft, boundary=None, padded=False, axis=0, scaling="psd")
    # Z: (n_freq, n_kanal, n_frame) -> energi, rata-rata kanal
    P = (np.abs(Z) ** 2).mean(axis=1)                        # (n_freq, n_frame)
    return P[:N_FREQ_BINS].T                                  # (n_frame, 256)


def read_muse_csv(path: str):
    """Baca CSV hasil rekam_muse_paper.py / pengambilan_eeg.py -> (raw µV, nama kanal)."""
    import pandas as pd
    df = pd.read_csv(path)
    chans = [c for c in df.columns if c in MUSE_CHANNELS]
    return df[chans].to_numpy(dtype=float), chans


def load_muse_subject(folder: str, n_per_state: int = N_PER_STATE) -> dict:
    """folder/con/*.csv dan folder/relax/*.csv (rekaman 10 s format rekam_muse_paper.py)."""
    out = {}
    for state in ("con", "relax"):
        files = sorted(glob.glob(os.path.join(folder, state, "*.csv")))[:n_per_state]
        specs = []
        for f in files:
            raw, ch = read_muse_csv(f)
            specs.append(muse_raw_to_spectrogram(raw, channel_names=ch))
        out[state] = specs
        out["files_" + state] = files
    return out
