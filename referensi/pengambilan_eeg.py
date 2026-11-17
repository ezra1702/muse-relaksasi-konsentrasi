"""
Dashboard EEG Muse 2 versi simpel (Streamlit).

Install:
    python -m pip install streamlit brainflow pandas numpy

Jalankan:
    streamlit run muse_streamlit.py
"""
import time

import numpy as np
import pandas as pd
import streamlit as st
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations

BOARD_ID = BoardIds.MUSE_2_BOARD  # ganti ke MUSE_S_BOARD kalau Muse S
WINDOW_SEC = 5
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
BAND_RANGES = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

st.set_page_config(page_title="Muse EEG", layout="wide")
st.title("Muse EEG Real-time")


@st.cache_resource
def get_board():
    """Konek ke Muse sekali saja, tetap hidup walau halaman refresh."""
    params = BrainFlowInputParams()
    params.timeout = 30
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream()
    return board


fs = BoardShim.get_sampling_rate(BOARD_ID)
eeg_rows = BoardShim.get_eeg_channels(BOARD_ID)
names = BoardShim.get_eeg_names(BOARD_ID)

run = st.toggle("Mulai tampilkan data")

if st.button("Simpan CSV"):
    board = get_board()
    data = board.get_current_board_data(fs * 3600)
    df = pd.DataFrame(data[eeg_rows].T, columns=names)
    df.insert(0, "timestamp", data[BoardShim.get_timestamp_channel(BOARD_ID)])
    fname = time.strftime("muse_eeg_%Y%m%d_%H%M%S.csv")
    df.to_csv(fname, index=False)
    st.success(f"Tersimpan: {fname} ({len(df)} sampel)")

col1, col2 = st.columns([2, 1])
col1.caption("Sumbu Y = mikrovolt (µV). Sinyal sudah difilter 1-45 Hz + notch 50 Hz.")
signal_box = col1.empty()
band_box = col2.empty()

if run:
    board = get_board()
    while run:
        data = board.get_current_board_data(WINDOW_SEC * fs)
        if data.shape[1] >= fs * 2:
            sig = {}
            for ch, name in zip(eeg_rows, names):
                y = data[ch].astype(np.float64).copy()
                DataFilter.perform_bandpass(y, fs, 1.0, 45.0, 2, FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(y, fs, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
                sig[name] = y
            signal_box.line_chart(pd.DataFrame(sig).iloc[fs // 2:])  # buang tepi filter

            # Band power dihitung sendiri dari PSD (Welch) supaya tidak kena error buffer BrainFlow
            try:
                psds = []
                for y in sig.values():
                    y = np.ascontiguousarray(y[fs // 2:], dtype=np.float64)  # buang tepi filter
                    psd, freqs = DataFilter.get_psd_welch(
                        y, 256, 128, fs, WindowOperations.HANNING.value)
                    psds.append(psd)
                psd = np.mean(psds, axis=0)
                bands = [psd[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in BAND_RANGES]
                rel = 100 * np.array(bands) / max(sum(bands), 1e-12)
                band_box.bar_chart(pd.DataFrame({"Band power (%)": rel}, index=BANDS))
            except Exception as e:
                band_box.warning(f"Band power belum bisa dihitung: {e}")
        time.sleep(0.5)
else:
    st.info("Nyalakan toggle di atas untuk mulai. Pastikan Muse menyala dan tidak terhubung ke HP.")
