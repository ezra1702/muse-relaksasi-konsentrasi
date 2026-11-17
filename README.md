# Klasifikasi Relaksasi vs Konsentrasi dengan EEG Muse (SVM & BPNN)

Implementasi metode paper **You, S.D. (2021). *Classification of Relaxation and Concentration
Mental States with EEG*. Information, 12(5), 187** untuk alat **Muse (Muse 2 / Muse S, satuan µV)**.
Paper aslinya memakai NeuroSky. Di sini semua langkah mengikuti paper, dan setiap bagian yang
terpaksa berbeda karena alatnya Muse ditandai `[no-jurnal | terpaksa]` atau `[no-jurnal | tambahan]` di kode.

Isinya:
- **Notebook** `EEG_Muse_Paper_Model.ipynb`: semua eksperimen paper (Tabel 1–6, Gambar 4–7), F1 score,
  dan pembuatan model `.pkl`.
- **Aplikasi Streamlit** `app_muse.py`: tersambung otomatis ke Muse, menampilkan grafik EEG µV
  real-time, lalu mengklasifikasi rekaman 10 detik dengan **SVM** dan **BPNN** sekaligus.

## Struktur folder

```
.
├── app_muse.py                  # aplikasi Streamlit (grafik µV + klasifikasi SVM & BPNN)
├── muse_bridge.py               # koneksi Bluetooth Muse (dijalankan otomatis oleh app)
├── paper_eeg.py                 # semua metode paper + konversi sinyal Muse -> spektrogram
├── rekam_muse_paper.py          # perekam data Muse dengan protokol paper (CLI)
├── EEG_Muse_Paper_Model.ipynb   # eksperimen paper + pelatihan model
├── models/
│   ├── svm_muse.pkl             # SVM RBF (C=32, gamma=8), dipakai app
│   ├── bpnn_muse.pkl            # BPNN 10 hidden unit (MSE), dipakai app
│   └── svm_jahzeel              # model SVM alternatif (tidak dipakai default)
├── eeg/extracted/Run_1, Run_2   # dataset paper (github.com/NTUT-LabASPL/EEG)
├── referensi/pengambilan_eeg.py # contoh acuan pengambilan data mentah Muse
├── .streamlit/config.toml       # tema terang
└── requirements.txt
```

## Instalasi

Butuh **Python 3.11**.

```bash
git clone <URL-repo-ini>
cd <nama-folder-repo>

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Atau dengan conda:

```bash
conda create -n eeg-muse python=3.11 -y
conda activate eeg-muse
pip install -r requirements.txt
```

> Versi `scikit-learn` dikunci (1.9.1) karena model `.pkl` dibuat dengan versi itu.

## Menjalankan aplikasi

```bash
streamlit run app_muse.py
```

1. **Nyalakan Muse.** Aplikasi mencari dan menyambung sendiri, tidak perlu menekan tombol.
   Tanpa alat, pilih **"Simulasi (tanpa Muse)"** di sidebar.
2. Empat grafik menampilkan sinyal µV per kanal:
   - TP9: belakang telinga kiri
   - AF7: dahi kiri
   - AF8: dahi kanan
   - TP10: belakang telinga kanan

   AF7 dan AF8 dipakai untuk klasifikasi. Sinyal yang baik punya RMS di bawah ±100 µV.
3. Pilih kondisi, lalu tekan **"▶ Mulai rekaman 10 detik"** dan ikuti protokolnya:
   - Mata tertutup.
   - Detik 0–5: rileks.
   - Detik 5–10, untuk kondisi konsentrasi: sebutkan 5 digit yang muncul secara terbalik di dalam hati.
   - Detik 5–10, untuk kondisi relaksasi: tetap rileks.
4. Hasil SVM dan BPNN tampil berdampingan.

Di macOS, izinkan akses **Bluetooth** untuk Terminal/IDE saat diminta.

## Menjalankan notebook

```bash
jupyter notebook EEG_Muse_Paper_Model.ipynb
```

Jalankan semua sel (*Run All*). Notebook memuat `eeg/extracted/Run_1` dan `Run_2`, menjalankan
eksperimen paper, lalu menyimpan `models/svm_muse.pkl` dan `models/bpnn_muse.pkl`.

## Merekam data Muse sendiri (opsional)

```bash
python rekam_muse_paper.py --nama budi
```

Hasilnya tersimpan di `data_muse/budi/relax/*.csv` dan `data_muse/budi/con/*.csv`, berisi 20 rekaman per kondisi.
Folder `data_muse/` tidak ikut diunggah ke GitHub. Untuk melatih model pribadi, set
`SUMBER_LATIH = "muse:budi"` di bagian 12 notebook.

## Metode (ringkas, sesuai paper)

| Tahap | Paper | Di proyek ini |
|---|---|---|
| Spektrogram | 0.125 s × 0.25 Hz, 0–64 Hz | sama (STFT dari µV Muse) `[terpaksa]` |
| Segmen | detik ke-6 s/d ke-8 | sama |
| Fitur | Pers. (1), band α+β+γ, bandwidth 4 Hz | sama |
| SVM | RBF, γ = 8, C = 32 | sama |
| BPNN | 10 hidden unit, MSE | sama |
| Evaluasi | 70/30, 40 trial, F1 | sama |

### Hasil (F1, %, α+β+γ 4 Hz)

| Model | Individual | Umum (generic) | Paper |
|---|---|---|---|
| SVM  | 65.4 | 64.3 | 84.6 |
| BPNN | 72.1 | 57.6 | 84.0 |

**Model yang dipakai app: `svm_muse.pkl` + `bpnn_muse.pkl`.** Model ini dipilih karena F1 data ujinya terbaik.
`svm_jahzeel` hanya mencapai akurasi 52–68 % pada dataset yang sama. Untuk memakainya, ubah
`MODEL_FILES` di `app_muse.py`.

## Sumber

- Paper: S.D. You, *Classification of Relaxation and Concentration Mental States with EEG*,
  Information 2021, 12, 187. https://doi.org/10.3390/info12050187
- Dataset: https://github.com/NTUT-LabASPL/EEG
- Koneksi Muse: [BrainFlow](https://brainflow.org)
