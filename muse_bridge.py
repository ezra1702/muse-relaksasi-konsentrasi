"""
Jembatan Muse -> Streamlit.  [no-jurnal | terpaksa] (alat Muse)

Kenapa perlu: library Bluetooth BrainFlow (SimpleBLE) di macOS bisa deadlock bila
prepare_session() dipanggil dari thread Streamlit. Di proses biasa (thread utama)
koneksi Muse berjalan normal. Jadi koneksi Bluetooth dikerjakan di proses ini,
lalu data diteruskan ke aplikasi lewat fitur streaming bawaan BrainFlow
(streaming_board, multicast lokal). Koneksi mengikuti pengambilan_eeg.py.

Dijalankan otomatis oleh app_muse.py. Manual:
    python muse_bridge.py --board muse_2
Keluar (exit code != 0) bila Muse tidak ditemukan / data berhenti, supaya app mencoba lagi.
"""
import argparse
import sys
import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

BOARDS = {"muse_2": BoardIds.MUSE_2_BOARD, "muse_s": BoardIds.MUSE_S_BOARD,
          "synthetic": BoardIds.SYNTHETIC_BOARD}
STREAM_URL = "streaming_board://225.1.1.1:6677"        # alamat multicast lokal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="muse_2", choices=list(BOARDS))
    ap.add_argument("--serial", default="", help="nama perangkat, mis. Muse-1A2B (opsional)")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    params.timeout = args.timeout                       # waktu cari Bluetooth (detik)
    if args.serial:
        params.serial_number = args.serial
    board = BoardShim(BOARDS[args.board], params)
    try:
        board.prepare_session()                         # cari & sambungkan Muse (thread utama)
    except Exception as e:
        print(f"NOT_FOUND {e}", flush=True)
        sys.exit(1)
    board.add_streamer(STREAM_URL)                      # teruskan data ke app_muse.py
    board.start_stream()
    print("CONNECTED", flush=True)

    last_n, last_change = -1, time.time()
    try:
        while True:                                     # jaga koneksi: data harus terus masuk
            time.sleep(1)
            n = board.get_board_data_count()
            if n != last_n:
                last_n, last_change = n, time.time()
            elif time.time() - last_change > 10:
                print("STALLED", flush=True)
                sys.exit(2)
            if n > 256 * 60:                            # buang buffer lokal (data sudah di-stream)
                board.get_board_data()
                last_n = -1
    finally:
        try:
            board.stop_stream(); board.release_session()
        except Exception:
            pass


if __name__ == "__main__":
    main()
