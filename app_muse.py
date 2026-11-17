"""
Aplikasi Streamlit: Muse -> grafik EEG mikrovolt (µV) -> klasifikasi Relaksasi vs Konsentrasi
dengan 2 algoritma paper (SVM dan BPNN).

Metode klasifikasi = paper You (2021), Information 12:187 (Document from Ezra.pdf):
  * 1 rekaman = 10 s: 5 s baseline rileks (mata tertutup) + 5 s tugas
      konsentrasi -> tester membacakan 5 digit, subjek menyebutkannya TERBALIK dalam hati
      relaksasi   -> tetap rileks, pikiran kosong
  * fitur dari detik ke-6 s/d ke-8, Pers. (1), alpha+beta+gamma 4 Hz
  * model: models/svm_muse.pkl (SVM RBF gamma=8, C=32) & models/bpnn_muse.pkl (BPNN 10 hidden, MSE)
Koneksi Muse mengikuti referensi/pengambilan_eeg.py (BrainFlow).

Bagian di luar jurnal ditandai [no-jurnal | tambahan] / [no-jurnal | terpaksa].

Install (sekali):  pip install -r requirements.txt
Jalankan:          streamlit run app_muse.py
"""
import os
import random
import subprocess
import sys
import threading
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # path relatif (models/, eeg/) tetap benar dari folder mana pun
import paper_eeg as P                                   # metode paper + konversi µV Muse

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
BOARDS = {                                              # pilihan alat
    "Muse 2": BoardIds.MUSE_2_BOARD,
    "Muse S": BoardIds.MUSE_S_BOARD,
    "Simulasi (tanpa Muse)": BoardIds.SYNTHETIC_BOARD,  # [no-jurnal | tambahan] untuk uji aplikasi tanpa alat
}
# Model terbaik (diuji pada data yang sama, 40 trial data uji):
#   svm_muse.pkl  F1 64.3  (metode jurnal, C=32 gamma=8)  > svm_jahzeel  akurasi 52-68 % (format input ditebak)
# Untuk memakai model Jahzeel lagi, ganti "models/svm_muse.pkl" -> "models/svm_jahzeel".
MODEL_FILES = {"SVM": "models/svm_muse.pkl", "BPNN": "models/bpnn_muse.pkl"}

# [no-jurnal | terpaksa] svm_jahzeel = SVC scikit-learn polos (RBF, C=0.5, gamma=0.5; jurnal: C=32, gamma=8),
# 12 fitur dalam rentang 0-1, TANPA info fitur & scaler. Format input di bawah adalah TEBAKAN terbaik
# dari uji pada dataset jurnal (akurasi 61.8 %). Ganti bila Jahzeel memberi info yang benar.
JAHZEEL_CONFIG = {
    "feature_set": "beta|4Hz",      # 12 fitur = beta only 4 Hz x 3 detik (fitur jurnal)
    "norm": "none",                 # energi asli (bukan relative power)
    "log10": True,                  # fitur di-log10 sebelum min-max
    "scaler_data": ["Run_1", "Run_2"],   # min/max diambil dari dataset jurnal
    "kelas_1": "con",               # kelas 1 = konsentrasi
}
RECORD_SEC = 11                 # 10 s protokol paper + 1 s cadangan  [no-jurnal | terpaksa]
BASELINE_SEC = 5                # 5 s baseline rileks (paper Sec. 3.3)
LIVE_SEC = 5                    # panjang grafik real-time (detik)
DATA_DIR = "data_muse"          # tempat menyimpan rekaman (format sama dgn rekam_muse_paper.py)

BoardShim.disable_board_logger()                        # log BrainFlow tidak memenuhi terminal
st.set_page_config(page_title="Muse EEG - SVM & BPNN", layout="wide")


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
class SklearnSVMWrapper:
    """Bungkus SVC polos (svm_jahzeel) agar bisa dipakai seperti PaperEEGModel:
    spektrogram -> fitur jurnal -> (log10) -> min-max -> SVC."""

    def __init__(self, svc, cfg):
        from sklearn.preprocessing import MinMaxScaler
        self.estimator, self.cfg = svc, cfg
        self.labels = ("relax", "con") if cfg["kelas_1"] == "con" else ("con", "relax")
        X, y = [], []
        for r in cfg["scaler_data"]:                    # min/max dari dataset jurnal
            for d in P.load_run(f"eeg/extracted/{r}").values():
                a, b = P.subject_matrix(d, cfg["feature_set"], cfg["norm"]); X.append(a); y.append(b)
        X, y = np.vstack(X), np.concatenate(y)
        self.scaler = MinMaxScaler().fit(self._pre(X))
        pred = svc.predict(self.scaler.transform(self._pre(X)))
        yy = y if cfg["kelas_1"] == "con" else 1 - y
        tp, tn, fp, fn = P.confusion(yy, pred)
        mt = P.metrics_from_counts(tp, tn, fp, fn)
        self.info = {"sumber_latih": "tidak diketahui (svm_jahzeel)",
                     "f1_40trial": f"{mt['f1']:.1f} (uji dataset, format input ditebak)",
                     "akurasi_dataset": round(mt["accuracy"], 1)}

    def _pre(self, X):
        return np.log10(np.asarray(X) + 1e-12) if self.cfg["log10"] else np.asarray(X)

    def features(self, spec):
        f = P.extract_features(spec, self.cfg["feature_set"], self.cfg["norm"])
        return np.clip(self.scaler.transform(self._pre(f[None, :]))[0], 0, 1)


@st.cache_resource
def load_models():
    """Muat 2 model (.pkl) sekali saja."""
    out = {}
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            out[name] = None
        elif os.path.basename(path) == "svm_jahzeel":
            out[name] = SklearnSVMWrapper(joblib.load(path), JAHZEEL_CONFIG)   # SVC polos -> dibungkus
        else:
            out[name] = joblib.load(path)
    return out


BRIDGE_KEY = {"Muse 2": "muse_2", "Muse S": "muse_s", "Simulasi (tanpa Muse)": "synthetic"}


class MuseManager:
    """Koneksi Muse OTOMATIS (dipegang server, seperti get_board() di pengambilan_eeg.py).

    [no-jurnal | terpaksa] Bluetooth dikerjakan proses terpisah muse_bridge.py, karena SimpleBLE
    (BrainFlow) bisa deadlock bila dipanggil dari thread Streamlit. Data diterima lewat
    BoardIds.STREAMING_BOARD (fitur streaming bawaan BrainFlow).
    * Terus mencoba sampai Muse ditemukan; tidak perlu tekan tombol.
    * Bila Muse mati / data berhenti, bridge keluar lalu otomatis dijalankan ulang.
    """

    def __init__(self, board_name="Muse 2"):
        self.board_name = board_name        # alat target
        self.serial = ""                    # nama perangkat (opsional)
        self.board = None                   # BoardShim STREAMING_BOARD saat terhubung
        self.auto = True                    # mode sambung otomatis
        self.status = "Memulai pencarian ..."
        self.attempt = 0
        self.proc = None                    # proses muse_bridge.py
        self._dead = []                     # objek BoardShim lama: disimpan agar tidak di-GC
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def release(self):
        """Lepas koneksi: tutup streaming board & hentikan bridge."""
        with self.lock:
            if self.board is not None:
                try:
                    self.board.stop_stream(); self.board.release_session()
                except Exception:
                    pass
                # BrainFlow: saat objek lama di-GC, __del__-nya bisa menutup koneksi BARU yang
                # memakai parameter sama -> BOARD_NOT_CREATED. Simpan referensinya.
                self._dead.append(self.board)
            self.board = None
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(5)
                except Exception:
                    self.proc.kill()
            self.proc = None

    def _open_stream(self):
        params = BrainFlowInputParams()
        params.ip_address, params.ip_port = "225.1.1.1", 6677          # alamat streamer bridge
        params.master_board = BOARDS[self.board_name]                   # deskripsi kanal = Muse
        b = BoardShim(BoardIds.STREAMING_BOARD, params)
        try:
            b.prepare_session(); b.start_stream()
        except Exception:
            self._dead.append(b)                         # cegah GC menutup koneksi berikutnya
            raise
        return b

    def _loop(self):
        bridge = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muse_bridge.py")
        while True:
            if not self.auto or self.board is not None:
                if self.board is not None and self.proc is not None and self.proc.poll() is not None:
                    self.status = "Koneksi Muse terputus, mencari ulang ..."   # bridge keluar
                    print(time.strftime("%H:%M:%S"), "[MUSE]", self.status, flush=True)
                    self.release()
                time.sleep(1)
                continue
            self.attempt += 1
            self.status = f"Mencari {self.board_name} ... (percobaan {self.attempt})"
            cmd = [sys.executable, bridge, "--board", BRIDGE_KEY[self.board_name]]
            if self.serial.strip():
                cmd += ["--serial", self.serial.strip()]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.proc = proc
            line = proc.stdout.readline().strip()          # tunggu "CONNECTED" / "NOT_FOUND ..."
            if line == "CONNECTED" and self.auto:
                try:
                    b = self._open_stream()
                    with self.lock:
                        self.board = b
                    self.status = f"Terhubung: {self.board_name}"
                    print(time.strftime("%H:%M:%S"), "[MUSE]", self.status, flush=True)
                    continue
                except Exception as e:
                    line = f"STREAM_ERROR {e}"
            self.release()
            self.status = (f"{self.board_name} belum ditemukan (percobaan {self.attempt}). "
                           "Pastikan Muse menyala & tidak tersambung ke HP.")
            print(time.strftime("%H:%M:%S"), "[MUSE] gagal percobaan", self.attempt, line[:80], flush=True)
            time.sleep(2)


@st.cache_resource
def get_manager():
    """Satu pengelola koneksi untuk seluruh server (otomatis jalan saat aplikasi dibuka)."""
    return MuseManager("Muse 2")


def board_info(board: BoardShim):
    """Sampling rate, baris kanal EEG, nama kanal, baris timestamp."""
    bid = board.get_board_id()
    fs = BoardShim.get_sampling_rate(bid)
    rows = BoardShim.get_eeg_channels(bid)[:4]          # 4 kanal (Muse: TP9, AF7, AF8, TP10)
    names = BoardShim.get_eeg_names(bid)[:4]
    if bid == BoardIds.SYNTHETIC_BOARD:                 # simulasi: beri nama kanal Muse
        names = list(P.MUSE_CHANNELS)
    return fs, rows, names, BoardShim.get_timestamp_channel(bid)


def get_uv(board: BoardShim, seconds: float):
    """Ambil `seconds` detik terakhir sinyal mentah (µV): array (n_sampel, 4) + nama kanal."""
    fs, rows, names, ts_row = board_info(board)
    data = board.get_current_board_data(int(seconds * fs))   # TIDAK mengosongkan buffer
    return data[rows].T.astype(float), names, fs, data[ts_row]


def classify(raw_uv, names, fs, models):
    """Rekaman µV -> spektrogram format paper -> fitur paper -> prediksi SVM & BPNN."""
    S = P.muse_raw_to_spectrogram(raw_uv, fs=fs, channel_names=names)   # [no-jurnal | terpaksa] konversi Muse
    res = {}
    for name, m in models.items():
        if m is None:
            continue
        f = m.features(S)[None, :]                       # relative power + Pers.(1) + alpha+beta+gamma 4 Hz
        label = m.labels[int(m.estimator.predict(f)[0])] # 'con' atau 'relax'
        if name == "SVM":                                # skor SVM: jarak ke batas keputusan (>0 = con)
            score = float(m.estimator.decision_function(f)[0])
        else:                                            # skor BPNN: keluaran jaringan (target 0=relax, 1=con; >=0.5 = con)
            Xs = m.estimator.named_steps["scale"].transform(f)
            score = float(m.estimator.named_steps["bpnn"].net_.predict(Xs)[0])
        res[name] = {"label": label, "score": score}
    return S, res


def label_id(label: str) -> str:
    return {"con": "KONSENTRASI", "relax": "RELAKSASI"}[label]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
st.session_state.setdefault("history", [])              # riwayat hasil klasifikasi
st.session_state.setdefault("last", None)               # hasil rekaman terakhir
st.session_state.setdefault("was_connected", False)     # untuk refresh halaman saat koneksi berubah
models = load_models()
mgr = get_manager()                                     # koneksi Muse otomatis (background)

# ---------------------------------------------------------------------------
# Sidebar: koneksi Muse (otomatis)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Koneksi Muse (otomatis)")
    pilih = st.selectbox("Alat", list(BOARDS), index=list(BOARDS).index(mgr.board_name))
    serial = st.text_input("Nama perangkat (opsional)", value=mgr.serial, placeholder="mis. Muse-1A2B",
                           help="Kosongkan agar BrainFlow mencari Muse terdekat.")
    if pilih != mgr.board_name or serial != mgr.serial:  # ganti alat -> lepas & cari ulang otomatis
        mgr.board_name, mgr.serial, mgr.attempt = pilih, serial, 0
        mgr.release()
    c1, c2 = st.columns(2)
    if c1.button("Sambung ulang", use_container_width=True):
        mgr.auto, mgr.attempt = True, 0
        mgr.release()
    if c2.button("Putuskan", use_container_width=True, disabled=not mgr.auto):
        mgr.auto = False
        mgr.release(); mgr.status = "Diputus manual. Tekan 'Sambung ulang' untuk menyambung lagi."


    @st.fragment(run_every=1.0)                         # status koneksi diperbarui tiap 1 s
    def status_box():
        if mgr.board is not None:
            fs_, _, names_, _ = board_info(mgr.board)
            st.success(f"✅ Terhubung: {mgr.board_name}\n\n{fs_} Hz · kanal {', '.join(names_)}")
        else:
            st.warning(f"🔎 {mgr.status}")
            st.caption("* Nyalakan Muse (lampu berkedip).\n* Muse **tidak** boleh tersambung ke HP.\n"
                       "* macOS: izinkan Bluetooth untuk Terminal (Privacy → Bluetooth).")
        if (mgr.board is not None) != st.session_state.was_connected:   # koneksi berubah -> refresh halaman
            st.session_state.was_connected = mgr.board is not None
            st.rerun(scope="app")


    status_box()

    st.header("2. Model (paper)")
    for name, path in MODEL_FILES.items():
        m = models[name]
        if m is None:
            st.error(f"{name}: {path} tidak ditemukan. Jalankan notebook dulu.")
        else:
            st.write(f"**{name}**: `{path}`  \nlatih: {m.info.get('sumber_latih')} · "
                     f"F1 {m.info.get('f1_40trial')}")
            if os.path.basename(path) == "svm_jahzeel":
                st.warning("svm_jahzeel: format fitur & scaler tidak diketahui; dipakai tebakan terbaik "
                           f"(β 4 Hz, log10, min-max dataset), akurasi dataset {m.info['akurasi_dataset']}%. "
                           "C=0.5, γ=0.5 [no-jurnal]. Hasil SVM belum dapat dipercaya penuh.")
    if st.button("Muat ulang model"):
        load_models.clear(); st.rerun()

# ---------------------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------------------
st.title("Muse EEG: Relaksasi vs Konsentrasi (SVM & BPNN)")
st.caption("Metode klasifikasi: You (2021), *Classification of Relaxation and Concentration Mental States "
           "with EEG*. Grafik dalam mikrovolt (µV), filter 1–45 Hz + notch 50 Hz seperti pengambilan_eeg.py.")

board = mgr.board

# ---- Grafik µV real-time --------------------------------------------------
st.subheader("Sinyal EEG real-time (µV)")
live_on = st.toggle("Tampilkan grafik real-time", value=True)


@st.fragment(run_every=0.5)                             # perbarui grafik tiap 0.5 s tanpa reload halaman
def live_plot():
    b = mgr.board
    if b is None or not live_on:
        st.info("Menunggu Muse tersambung otomatis ... (lihat status di sidebar)")
        return
    try:
        raw, names, fs, _ = get_uv(b, LIVE_SEC + 1)      # +1 s untuk dibuang (tepi filter)
    except Exception as e:
        st.write(f"Koneksi sedang disambung ulang ... ({type(e).__name__}: {e})")
        print(time.strftime("%H:%M:%S"), "[LIVE] error get_uv:", repr(e), flush=True)
        return
    if raw.shape[0] < 2 * fs:
        st.write("Menunggu data ...")
        return
    filt = P.muse_preprocess(raw, fs)[fs:]               # filter 1-45 Hz + notch 50 Hz, buang 1 s tepi
    t = np.arange(filt.shape[0]) / fs - filt.shape[0] / fs   # sumbu waktu: detik ke belakang (0 = sekarang)
    cols = st.columns(2)
    for i, ch in enumerate(names):                       # 1 grafik per kanal, satuan µV
        df = pd.DataFrame({"waktu (s)": t, f"{ch} (µV)": filt[:, i]}).set_index("waktu (s)")
        letak = {"TP9": "belakang telinga kiri", "AF7": "dahi kiri", "AF8": "dahi kanan",
                 "TP10": "belakang telinga kanan"}.get(ch, "")
        with cols[i % 2]:
            st.markdown(f"**{ch}** · {letak}" + (" · *dipakai klasifikasi*" if ch in P.MUSE_FRONTAL else ""))
            st.line_chart(df, height=170, y_label="µV", x_label="waktu (s)")
    rms = np.sqrt((filt ** 2).mean(axis=0))              # amplitudo RMS µV per kanal
    st.caption("Amplitudo RMS (µV): " + " · ".join(f"{n} {v:.1f}" for n, v in zip(names, rms)) +
               ".  Nilai sangat besar (> 100 µV) biasanya berarti kontak elektroda kurang baik.")


live_plot()


# ---- Klasifikasi per rekaman (protokol paper) -----------------------------
st.divider()
st.subheader("Klasifikasi: rekaman 10 detik (protokol paper)")
st.markdown(
    "1. Subjek **menutup mata**.\n"
    "2. Detik **0–5**: baseline, subjek rileks.\n"
    "3. Detik **5–10**: **konsentrasi** berarti tester membacakan 5 digit dan subjek menyebutkannya *terbalik* dalam hati; "
    "**relaksasi** berarti subjek tetap rileks.\n"
    "4. Model SVM & BPNN memakai detik ke-6 s/d ke-8 dari rekaman.")

c1, c2, c3 = st.columns([1.2, 1, 1])
kondisi = c1.radio("Kondisi yang diminta ke subjek",
                   ["konsentrasi (con)", "relaksasi (relax)", "tidak ditentukan (prediksi saja)"])
simpan = c2.checkbox("Simpan rekaman (CSV µV) untuk melatih model Muse",
                     help=f"Disimpan ke {DATA_DIR}/<nama>/<con|relax>/. Dipakai notebook "
                          "(SUMBER_LATIH = 'muse:<nama>').")
nama = c2.text_input("Nama subjek", value="subjek1", disabled=not simpan)
mulai = c3.button("▶ Mulai rekaman 10 detik", type="primary", disabled=board is None,
                  use_container_width=True)

if mulai and board is not None:
    state = {"konsentrasi (con)": "con", "relaksasi (relax)": "relax"}.get(kondisi)
    digits = "".join(random.choice("0123456789") for _ in range(5))   # angka 5 digit [no-jurnal | terpaksa] acak
    box = st.empty()
    bar = st.progress(0.0)
    t0 = time.time()
    while (el := time.time() - t0) < RECORD_SEC:
        if el < BASELINE_SEC:
            box.info(f"⏱ {el:4.1f} s, **BASELINE**: rileks, mata tertutup.")
        elif state == "con":
            box.warning(f"⏱ {el:4.1f} s, **KONSENTRASI**: tester bacakan **{digits}**. "
                        f"Subjek sebutkan terbalik dalam hati ({digits[::-1]}).")
        else:
            box.success(f"⏱ {el:4.1f} s, **TETAP RILEKS**.")
        bar.progress(min(el / RECORD_SEC, 1.0))
        time.sleep(0.1)
    bar.progress(1.0)
    raw, names, fs, ts = get_uv(board, RECORD_SEC)       # 11 s terakhir = rekaman ini
    if raw.shape[0] < int(10 * fs):
        box.error(f"Data kurang ({raw.shape[0] / fs:.1f} s). Cek koneksi Muse lalu ulangi.")
    else:
        S, res = classify(raw, names, fs, models)
        box.success("Rekaman selesai dan sudah diklasifikasi.")
        saved = None
        if simpan and state is not None:                  # simpan CSV µV (format rekam_muse_paper.py)
            folder = os.path.join(DATA_DIR, nama.strip() or "subjek1", state)
            os.makedirs(folder, exist_ok=True)
            saved = os.path.join(folder, time.strftime("%y%m%d%H%M%S") + ".csv")
            df = pd.DataFrame(raw, columns=names)
            df.insert(0, "timestamp", ts[-raw.shape[0]:])
            df.to_csv(saved, index=False)
        st.session_state.last = {"raw": raw, "names": names, "fs": fs, "S": S, "res": res,
                                 "state": state, "saved": saved}
        st.session_state.history.insert(0, {
            "waktu": time.strftime("%H:%M:%S"), "diminta": state or "-",
            "SVM": res.get("SVM", {}).get("label", "-"), "BPNN": res.get("BPNN", {}).get("label", "-"),
            "skor SVM": round(res.get("SVM", {}).get("score", np.nan), 3),
            "skor BPNN": round(res.get("BPNN", {}).get("score", np.nan), 3),
            "file": saved or "-"})

# ---- Tampilkan hasil terakhir ---------------------------------------------
last = st.session_state.last
if last:
    st.markdown("#### Hasil klasifikasi")
    cols = st.columns(len(last["res"]) + 1)
    for col, (name, r) in zip(cols, last["res"].items()):
        benar = "" if last["state"] is None else (" ✅" if r["label"] == last["state"] else " ❌")
        col.metric(f"{name}", label_id(r["label"]) + benar,
                   help=("SVM: skor > 0 berarti konsentrasi" if name == "SVM" else "BPNN: skor ≥ 0.5 berarti konsentrasi (target latih: 0 = relaksasi, 1 = konsentrasi)"))
        col.caption(f"skor = {r['score']:.3f}")
    labels = {r["label"] for r in last["res"].values()}
    cols[-1].metric("Kesepakatan 2 model", "SAMA" if len(labels) == 1 else "BERBEDA")
    if last["saved"]:
        st.caption(f"Rekaman disimpan: `{last['saved']}`")

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    fs = last["fs"]
    filt = P.muse_preprocess(last["raw"], fs)            # sinyal µV terfilter seluruh rekaman
    t = np.arange(filt.shape[0]) / fs
    for i, ch in enumerate(last["names"]):
        ax[0].plot(t, filt[:, i] + i * 100, lw=0.6, label=ch)   # geser 100 µV per kanal agar terbaca
    ax[0].axvspan(5, 8, color="orange", alpha=0.15, label="detik 6–8 (fitur)")
    ax[0].axvline(5, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("Waktu (s)"); ax[0].set_ylabel("µV (tiap kanal digeser 100 µV)")
    ax[0].set_title("Sinyal rekaman (µV)"); ax[0].legend(fontsize=7, ncol=5, loc="upper right")
    S = last["S"]
    im = ax[1].imshow(10 * np.log10(S[:, :180].T + 1e-12), aspect="auto", origin="lower",
                      extent=[0, S.shape[0] / 8, 0, 45], cmap="jet")
    ax[1].axvspan(5, 8, color="w", alpha=0.2); ax[1].axvline(5, color="w", ls="--", lw=0.8)
    ax[1].set_xlabel("Waktu (s)"); ax[1].set_ylabel("Hz"); ax[1].set_title("Spektrogram AF7+AF8 (format paper)")
    plt.colorbar(im, ax=ax[1], label="dB (µV²/Hz)")
    plt.tight_layout()
    st.pyplot(fig)

if st.session_state.history:
    st.markdown("#### Riwayat klasifikasi")
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist, use_container_width=True, hide_index=True)
    known = hist[hist["diminta"].isin(["con", "relax"])]
    if len(known):
        acc = {m: (known[m] == known["diminta"]).mean() * 100 for m in ("SVM", "BPNN")}
        st.caption(f"Kecocokan dengan kondisi yang diminta ({len(known)} rekaman): "
                   f"SVM {acc['SVM']:.0f}% · BPNN {acc['BPNN']:.0f}%")
    if st.button("Hapus riwayat"):
        st.session_state.history = []; st.session_state.last = None; st.rerun()

st.divider()
st.caption("Catatan: model saat ini dilatih dari dataset paper yang dikonversi ke skala Muse "
           "[no-jurnal | tambahan]. Paper menganjurkan model individual. Simpan 20 rekaman con + 20 relax, "
           "lalu set SUMBER_LATIH = 'muse:<nama>' di notebook untuk melatih ulang svm_muse.pkl & bpnn_muse.pkl.")
