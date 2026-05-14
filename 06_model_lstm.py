"""
KomodiPrice — 06: Model LSTM (Long Short-Term Memory)
===============================================================
⚠️  SANGAT DISARANKAN: Jalankan di Google Colab dengan GPU
    Di CPU lokal: ~45-60 menit
    Di Colab GPU (T4): ~5-8 menit

CARA PAKAI DI GOOGLE COLAB:
1. Buka colab.research.google.com
2. Runtime → Change runtime type → GPU (T4)
3. Upload folder data/processed/ ke Google Drive
4. Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')
5. Ubah BASE_DIR di bawah menjadi path Drive kamu
   Contoh: BASE_DIR = "/content/drive/MyDrive/komodiprice"
===============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR   = "."
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COMMODITIES = [
    "Bawang Merah", "Bawang Putih", "Beras",
    "Cabai Merah",  "Cabai Rawit",  "Minyak Goreng"
]

COLORS = {
    "Bawang Merah" : "#e74c3c",
    "Bawang Putih" : "#9b59b6",
    "Beras"        : "#f39c12",
    "Cabai Merah"  : "#c0392b",
    "Cabai Rawit"  : "#e67e22",
    "Minyak Goreng": "#27ae60",
}

# Hyperparameter LSTM
EPOCHS        = 50    # tambah ke 100 jika di Colab GPU
BATCH_SIZE    = 32
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.2

# ============================================================
print("=" * 60)
print("  KomodiPrice — 06: Model LSTM")
print("=" * 60)

# Cek GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"\n  ✅ GPU terdeteksi: {gpus[0].name}")
    print("     Training akan jauh lebih cepat!")
else:
    print("\n  ⚠️  Tidak ada GPU — training di CPU")
    print("     Estimasi waktu: 45-60 menit")
    print("     Rekomendasi: jalankan di Google Colab dengan GPU")

print(f"\n  Konfigurasi:")
print(f"    Epochs     : {EPOCHS}")
print(f"    Batch size : {BATCH_SIZE}")
print(f"    LSTM units : {LSTM_UNITS}")
print(f"    Dropout    : {DROPOUT_RATE}")

# ---- Load data preprocessing --------------------------------
print("\n[STEP 1] Loading data dari preprocessing...")
with open(os.path.join(PROC_DIR, "lstm_data.pkl"), "rb") as f:
    lstm_data = pickle.load(f)
with open(os.path.join(PROC_DIR, "scalers_X.pkl"), "rb") as f:
    scalers_X = pickle.load(f)
with open(os.path.join(PROC_DIR, "scalers_y.pkl"), "rb") as f:
    scalers_y = pickle.load(f)

print("  ✓ lstm_data.pkl berhasil dimuat")
print("  ✓ scalers_X.pkl & scalers_y.pkl berhasil dimuat")

# ---- Fungsi evaluasi ----------------------------------------
def evaluate(y_true, y_pred, name=""):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    if name:
        print(f"    MAE  : Rp {mae:>10,.0f}")
        print(f"    RMSE : Rp {rmse:>10,.0f}")
        print(f"    MAPE : {mape:>8.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# ---- Fungsi bangun model LSTM -------------------------------
def build_lstm_model(seq_len, n_features, units=64, dropout=0.2):
    """
    Arsitektur LSTM:
    - Layer 1: Bidirectional LSTM (tangkap pola dari 2 arah)
    - BatchNorm: stabilkan training
    - Dropout: cegah overfitting
    - Layer 2: LSTM biasa
    - Dense output: prediksi 1 nilai harga
    """
    model = Sequential([
        Bidirectional(
            LSTM(units, return_sequences=True),
            input_shape=(seq_len, n_features)
        ),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="KomodiPrice_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",     # lebih robust terhadap outlier vs MSE
        metrics=["mae"]
    )
    return model

# ============================================================
print("\n[STEP 2] Training LSTM per komoditas...")
results_lstm = {}
histories    = {}

for comm in COMMODITIES:
    print(f"\n  → {comm}")
    d = lstm_data[comm]

    X_train = d["X_train"]
    y_train = d["y_train"]
    X_val   = d["X_val"]
    y_val   = d["y_val"]
    X_test  = d["X_test"]
    y_test  = d["y_test"]

    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]
    print(f"    Input shape: ({seq_len} hari × {n_features} fitur)")

    # --- Bangun model ---
    model = build_lstm_model(seq_len, n_features, LSTM_UNITS, DROPOUT_RATE)

    if comm == COMMODITIES[0]:
        model.summary()  # Tampilkan arsitektur hanya sekali

    # --- Callbacks ---
    checkpoint_path = os.path.join(MODEL_DIR, f"lstm_{comm.replace(' ','_')}.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=0
        ),
        ModelCheckpoint(
            checkpoint_path, save_best_only=True,
            monitor="val_loss", verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=0
        ),
    ]

    # --- Training ---
    history = model.fit(
        X_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (X_val, y_val),
        callbacks       = callbacks,
        verbose         = 1,
    )
    histories[comm] = history.history

    # --- Evaluasi pada test set ---
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform (kembalikan ke Rp asli)
    scaler_y = scalers_y[comm]
    y_test_rp = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_rp = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_pred_rp = np.maximum(y_pred_rp, 0)

    metrics = evaluate(y_test_rp, y_pred_rp, name=comm)

    # --- Simpan ---
    results_lstm[comm] = {
        "model"       : model,
        "y_test_rp"   : y_test_rp,
        "y_pred_rp"   : y_pred_rp,
        "metrics"     : metrics,
        "last_sequence": d.get("last_sequence"),
        "test_dates"  : d.get("test_dates"),
    }

# ============================================================
print("\n\n[STEP 3] Membuat visualisasi forecast...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle(
    f"LSTM — Prediksi vs Aktual (Test Set)\n"
    f"Arsitektur: Bidirectional LSTM ({LSTM_UNITS} units) | Epochs: {EPOCHS} | Huber Loss",
    fontsize=13, fontweight="bold", y=0.98
)
axes = axes.flatten()

for i, comm in enumerate(COMMODITIES):
    ax  = axes[i]
    r   = results_lstm[comm]
    col = COLORS[comm]

    y_true      = r["y_test_rp"]
    y_pred      = r["y_pred_rp"]
    test_dates  = r.get("test_dates")

    x_axis = test_dates if test_dates is not None else range(len(y_true))

    ax.plot(x_axis, y_true, color="black", linewidth=1.5, label="Aktual")
    ax.plot(x_axis, y_pred, color=col, linewidth=2,
            linestyle="--", label="Prediksi LSTM")

    # Simple confidence band (±1 std dari error)
    errors = y_true - y_pred
    std_e  = np.std(errors)
    ax.fill_between(x_axis, y_pred - std_e, y_pred + std_e,
                    color=col, alpha=0.15, label="±1 Std Error")

    mape = r["metrics"]["MAPE"]
    mae  = r["metrics"]["MAE"]
    ax.set_title(f"{comm}  |  MAPE: {mape:.1f}%  |  MAE: Rp{mae:,.0f}",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Harga (Rp/kg)")
    ax.legend(fontsize=8, loc="upper left")
    if test_dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rp{x/1000:.0f}K"))

plt.tight_layout()
path_plot = os.path.join(OUT_DIR, "model_lstm_forecast.png")
plt.savefig(path_plot, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✓ Grafik disimpan: {path_plot}")

# ============================================================
print("\n[STEP 4] Visualisasi training history...")

fig2, axes2 = plt.subplots(3, 2, figsize=(14, 10))
fig2.suptitle("LSTM — Training Loss per Komoditas", fontsize=14, fontweight="bold")
axes2 = axes2.flatten()

for i, comm in enumerate(COMMODITIES):
    ax = axes2[i]
    h  = histories[comm]
    ax.plot(h["loss"],     label="Train Loss", color=COLORS[comm])
    ax.plot(h["val_loss"], label="Val Loss",   color=COLORS[comm],
            linestyle="--", alpha=0.7)
    ax.set_title(comm, fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path_hist = os.path.join(OUT_DIR, "model_lstm_training_history.png")
plt.savefig(path_hist, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✓ Training history disimpan: {path_hist}")
print("    → Grafik ini menunjukkan apakah model konvergen dengan baik")
print("      (train & val loss turun bersama = tidak overfitting)")

# ============================================================
print("\n[STEP 5] Ringkasan evaluasi LSTM...")
rows = []
for comm in COMMODITIES:
    m = results_lstm[comm]["metrics"]
    rows.append({
        "Komoditas"  : comm,
        "MAE (Rp)"   : f"{m['MAE']:,.0f}",
        "RMSE (Rp)"  : f"{m['RMSE']:,.0f}",
        "MAPE (%)"   : f"{m['MAPE']:.2f}%",
        "Kategori"   : "✅ Baik" if m["MAPE"] < 10 else
                       "⚠️ Cukup" if m["MAPE"] < 20 else "❌ Perlu tuning",
    })

df_results = pd.DataFrame(rows)
print(df_results.to_string(index=False))

# ============================================================
print("\n[STEP 6] Menyimpan hasil evaluasi LSTM...")
save_data = {
    comm: {
        "metrics"      : results_lstm[comm]["metrics"],
        "y_test_rp"    : results_lstm[comm]["y_test_rp"],
        "y_pred_rp"    : results_lstm[comm]["y_pred_rp"],
        "last_sequence": results_lstm[comm]["last_sequence"],
        "test_dates"   : results_lstm[comm]["test_dates"],
    }
    for comm in COMMODITIES
}
with open(os.path.join(MODEL_DIR, "lstm_results.pkl"), "wb") as f:
    pickle.dump(save_data, f)
print("  ✓ lstm_results.pkl tersimpan")
print(f"  ✓ Model per komoditas disimpan di folder: {MODEL_DIR}/")

# ============================================================
print("\n" + "=" * 60)
print("  INTERPRETASI UNTUK LAPORAN")
print("=" * 60)
avg_mape = np.mean([results_lstm[c]["metrics"]["MAPE"] for c in COMMODITIES])
print(f"\n  Rata-rata MAPE semua komoditas: {avg_mape:.2f}%")
print("""
  KELEBIHAN LSTM yang teramati dari data ini:
  ✓ Arsitektur deep learning paling canggih di antara ketiganya
  ✓ Bisa belajar pola non-linear yang kompleks
  ✓ Multivariat: menggunakan SEMUA 6 komoditas + 9 fitur kalender sekaligus
    → bisa belajar hubungan antar komoditas (misal: cabai merah naik → cabai rawit naik)
  ✓ Bidirectional: membaca urutan waktu dari dua arah → tangkap pola lebih baik

  KELEMAHAN LSTM yang teramati dari data ini:
  ✗ BUTUH GPU untuk training yang wajar → hambatan besar untuk deployment
  ✗ Overfitting lebih mudah terjadi — perlu tuning extra
  ✗ Black box: sulit dijelaskan ke dosen/pembimbing secara intuitif
  ✗ Ukuran model besar (.keras file) → deployment lebih berat
  ✗ Untuk prediksi 30 hari ke depan, error bisa membesar (cascading prediction)
  ✗ Training ulang ketika ada data baru membutuhkan waktu lama

  KESIMPULAN:
  LSTM adalah model paling powerful, TAPI untuk proyek ini:
  - Kompleksitas tidak sebanding dengan manfaat tambahan
  - Tidak bisa memanfaatkan confidence interval otomatis
  - Deployment jauh lebih sulit dibanding Prophet
  → Bukan pilihan utama untuk proyek skala praktikum ini
""")

print("=" * 60)
print("  ✅ LSTM selesai!")
print("=" * 60)
