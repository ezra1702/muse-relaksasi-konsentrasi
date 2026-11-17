"""
Perekaman data Muse dengan PROTOKOL PAPER (You, 2021 - Document from Ezra.pdf, Sec. 3.1 & 3.3).
Koneksi Muse mengikuti referensi/pengambilan_eeg.py (BrainFlow).  [no-jurnal | terpaksa] (alat Muse)

Protokol per rekaman (10 s, direkam 11 s supaya ada cadangan):
  * Subjek MENUTUP MATA selama rekaman (Sec. 3.2, menghindari artefak kedip).
  * Detik 0-5  : baseline, subjek rileks tanpa berpikir.
  * Detik 5-10 :
        con   -> tester membacakan 5 digit angka (dicetak di layar), subjek
                 menyebutkan angka itu TERBALIK di dalam hati (13542 -> 24531).
        relax -> subjek tetap rileks, pikiran kosong.
  * 20 rekaman con + 20 rekaman relax per subjek.
  * Urutan blok mengikuti dataset paper: 20 relax dulu, lalu 20 con.

Hasil: data_muse/<nama>/relax/*.csv dan data_muse/<nama>/con/*.csv
Kolom: timestamp, TP9, AF7, AF8, TP10 (mikrovolt, mentah tanpa filter;
filter & spektrogram dikerjakan di notebook / paper_eeg.py).

Install:  pip install -r requirements.txt
Jalankan: python rekam_muse_paper.py --nama budi
"""
import argparse
import os
import random
import sys
import time

import pandas as pd
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

RECORD_SEC = 11          # 10 s protokol paper + 1 s cadangan  [no-jurnal | terpaksa]
BASELINE_SEC = 5
N_PER_STATE = 20


def beep(n=1):  # [no-jurnal | tambahan] penanda waktu untuk tester (mata subjek tertutup)
    sys.stdout.write("\a" * n)
    sys.stdout.flush()


def connect(board_name: str) -> BoardShim:
    board_id = BoardIds.MUSE_S_BOARD if board_name == "muse_s" else BoardIds.MUSE_2_BOARD
    params = BrainFlowInputParams()
    params.timeout = 30
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    return board


def record_trial(board: BoardShim, state: str) -> pd.DataFrame:
    board_id = board.get_board_id()
    fs = BoardShim.get_sampling_rate(board_id)
    eeg_rows = BoardShim.get_eeg_channels(board_id)
    names = BoardShim.get_eeg_names(board_id)
    ts_row = BoardShim.get_timestamp_channel(board_id)

    digits = "".join(random.choice("0123456789") for _ in range(5))   # angka 5 digit (jurnal); diacak oleh script [no-jurnal | terpaksa]
    if state == "con":
        print(f"   Angka untuk dibacakan di detik ke-5: >>> {digits} <<<  (jawaban: {digits[::-1]})")
    else:
        print("   Kondisi RELAX: subjek tetap rileks sampai selesai.")
    input("   Subjek sudah menutup mata? Tekan Enter untuk mulai...")

    board.get_board_data()                       # kosongkan buffer
    t0 = time.time()
    beep(1)
    print("   [0 s] baseline rileks...")
    while time.time() - t0 < BASELINE_SEC:
        time.sleep(0.01)
    beep(2)
    print("   [5 s] " + (f"BACAKAN SEKARANG: {digits}" if state == "con" else "tetap rileks"))
    while time.time() - t0 < RECORD_SEC:
        time.sleep(0.01)
    beep(1)
    data = board.get_board_data()

    df = pd.DataFrame(data[eeg_rows].T, columns=names)
    df.insert(0, "timestamp", data[ts_row])
    n_expect = int(RECORD_SEC * fs * 0.95)
    if len(df) < n_expect:
        print(f"   PERINGATAN: hanya {len(df)} sampel (harusnya ~{RECORD_SEC * fs}). Ulangi bila perlu.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nama", required=True, help="nama subjek, mis. budi")
    ap.add_argument("--board", default="muse_2", choices=["muse_2", "muse_s"])
    ap.add_argument("--n", type=int, default=N_PER_STATE, help="rekaman per kondisi (paper: 20)")
    ap.add_argument("--out", default="data_muse")
    args = ap.parse_args()

    folders, done = {}, {}
    for state in ("relax", "con"):
        folders[state] = os.path.join(args.out, args.nama, state)
        os.makedirs(folders[state], exist_ok=True)
        done[state] = len([f for f in os.listdir(folders[state]) if f.endswith(".csv")])
    todo = ["relax"] * (args.n - done["relax"]) + ["con"] * (args.n - done["con"])
    print(f"Sudah ada: relax {done['relax']}/{args.n}, con {done['con']}/{args.n}. "
          f"Sisa {len(todo)} rekaman (blok relax lalu blok con).")

    board = connect(args.board)
    try:
        i = 0
        while i < len(todo):
            state = todo[i]
            print(f"\n-- [{i + 1}/{len(todo)}] {state.upper()}  "
                  f"(relax {done['relax']}/{args.n}, con {done['con']}/{args.n})")
            df = record_trial(board, state)
            ans = input("   Simpan? [Enter=ya / u=ulang] ").strip().lower()
            if ans == "u":
                continue
            fname = time.strftime("%y%m%d%H%M%S") + ".csv"
            df.to_csv(os.path.join(folders[state], fname), index=False)
            done[state] += 1
            i += 1
            print(f"   tersimpan {state}/{fname} ({len(df)} sampel)")
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
