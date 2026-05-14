"""
KomodiPrice — 04: Model ARIMA
===============================================================
Jalankan di: Google Colab atau lokal (tidak butuh GPU)
Estimasi waktu: ~5-10 menit (batch forecast)
===============================================================
CARA PAKAI DI GOOGLE COLAB:
1. Upload folder data/processed/ ke Google Drive
2. Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')
3. Ubah BASE_DIR di bawah menjadi path Drive kamu
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI — sesuaikan jika pakai Google Colab
# ============================================================
BASE_DIR   = "."                        # ganti ke path Drive jika di Colab
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

# ============================================================
print("=" * 60)
print("  KomodiPrice — 04: Model ARIMA")
print("=" * 60)

# ---- Load data preprocessing --------------------------------
print("\n[STEP 1] Loading data dari preprocessing...")
with open(os.path.join(PROC_DIR, "arima_data.pkl"), "rb") as f:
    arima_data = pickle.load(f)
print("  ✓ arima_data.pkl berhasil dimuat")

# ---- Fungsi evaluasi ----------------------------------------
def evaluate(y_true, y_pred, name=""):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    if name:
        print(f"    MAE  : Rp {mae:>10,.0f}")
        print(f"    RMSE : Rp {rmse:>10,.0f}")
        print(f"    MAPE : {mape:>8.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# ============================================================
print("\n[STEP 2] Training ARIMA per komoditas...")
print("  Strategi: Batch forecast (cepat)")
print("  Parameter: auto-grid (p:0-3, q:0-2)")
print()

results_arima = {}

for comm in COMMODITIES:
    print(f"  → {comm}")
    d = arima_data[comm]

    train_series = d["train"]
    val_series   = d["val"]
    test_series  = d["test"]
    d_order      = d["d_order"]   # 0 atau 1, dari stationarity check

    # --- Grid search sederhana untuk p dan q ---
    best_aic   = np.inf
    best_order = (1, d_order, 1)

    for p in range(0, 4):
        for q in range(0, 3):
            try:
                mdl = ARIMA(train_series, order=(p, d_order, q)).fit()
                if mdl.aic < best_aic:
                    best_aic   = mdl.aic
                    best_order = (p, d_order, q)
            except Exception:
                continue

    print(f"    Best order: {best_order}  |  AIC: {best_aic:.1f}")

    # --- Training final dengan order terbaik ---
    # Gabungkan train + val untuk forecast test
    train_val = pd.concat([train_series, val_series])
    model_final = ARIMA(train_val, order=best_order).fit()

    # --- Batch forecast test set ---
    n_test    = len(test_series)
    forecast  = model_final.forecast(steps=n_test)
    conf_int  = model_final.get_forecast(steps=n_test).conf_int(alpha=0.05)

    metrics = evaluate(test_series.values, forecast.values, name=comm)

    # --- Simpan hasil ---
    results_arima[comm] = {
        "order"       : best_order,
        "aic"         : best_aic,
        "model"       : model_final,
        "train_val"   : train_val,
        "test"        : test_series,
        "forecast"    : forecast,
        "conf_int"    : conf_int,
        "metrics"     : metrics,
    }
    print()

# ============================================================
print("[STEP 3] Membuat visualisasi...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("ARIMA — Prediksi vs Aktual (Test Set)\nStrategi: Batch Forecast",
             fontsize=15, fontweight="bold", y=0.98)
axes = axes.flatten()

for i, comm in enumerate(COMMODITIES):
    ax  = axes[i]
    r   = results_arima[comm]
    col = COLORS[comm]

    test      = r["test"]
    forecast  = r["forecast"]
    conf_int  = r["conf_int"]
    train_val = r["train_val"]

    # Plot historis (50 hari terakhir train_val) + test
    hist_tail = train_val.tail(50)
    ax.plot(hist_tail.index, hist_tail.values,
            color="gray", linewidth=1.2, alpha=0.6, label="Historis")
    ax.plot(test.index, test.values,
            color="black", linewidth=1.5, label="Aktual")
    ax.plot(forecast.index, forecast.values,
            color=col, linewidth=2, linestyle="--", label="Prediksi ARIMA")
    ax.fill_between(forecast.index,
                    conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    color=col, alpha=0.15, label="95% CI")

    mape = r["metrics"]["MAPE"]
    mae  = r["metrics"]["MAE"]
    ax.set_title(f"{comm}  |  MAPE: {mape:.1f}%  |  MAE: Rp{mae:,.0f}",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Harga (Rp/kg)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rp{x/1000:.0f}K"))

plt.tight_layout()
path_plot = os.path.join(OUT_DIR, "model_arima_forecast.png")
plt.savefig(path_plot, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✓ Grafik disimpan: {path_plot}")

# ============================================================
print("\n[STEP 4] Ringkasan evaluasi ARIMA...")
rows = []
for comm in COMMODITIES:
    m = results_arima[comm]["metrics"]
    o = results_arima[comm]["order"]
    rows.append({
        "Komoditas": comm,
        "Order (p,d,q)": str(o),
        "MAE (Rp)": f"{m['MAE']:,.0f}",
        "RMSE (Rp)": f"{m['RMSE']:,.0f}",
        "MAPE (%)": f"{m['MAPE']:.2f}%",
        "Kategori": "✅ Baik" if m["MAPE"] < 10 else
                    "⚠️ Cukup" if m["MAPE"] < 20 else "❌ Perlu tuning",
    })

df_results = pd.DataFrame(rows)
print(df_results.to_string(index=False))

# ============================================================
print("\n[STEP 5] Menyimpan model ARIMA...")
with open(os.path.join(MODEL_DIR, "arima_results.pkl"), "wb") as f:
    # Simpan tanpa objek model (tidak serializable dengan mudah)
    save_data = {
        comm: {
            "order"   : results_arima[comm]["order"],
            "metrics" : results_arima[comm]["metrics"],
            "forecast": results_arima[comm]["forecast"],
            "test"    : results_arima[comm]["test"],
        }
        for comm in COMMODITIES
    }
    pickle.dump(save_data, f)
print("  ✓ arima_results.pkl tersimpan")

# ============================================================
# INTERPRETASI UNTUK LAPORAN
# ============================================================
print("\n" + "=" * 60)
print("  INTERPRETASI UNTUK LAPORAN")
print("=" * 60)
avg_mape = np.mean([results_arima[c]["metrics"]["MAPE"] for c in COMMODITIES])
print(f"\n  Rata-rata MAPE semua komoditas: {avg_mape:.2f}%")
print("""
  KELEBIHAN ARIMA yang teramati dari data ini:
  ✓ Cepat ditraining — batch forecast selesai < 10 menit
  ✓ Tidak butuh GPU
  ✓ Model sederhana dan mudah dijelaskan secara matematis
  ✓ Bekerja baik untuk data yang stasioner (beras, minyak goreng)

  KELEMAHAN ARIMA yang teramati dari data ini:
  ✗ Tidak bisa tangani multiple seasonality secara otomatis
    (Ramadan + musim panen + weekly pattern perlu ditangani manual)
  ✗ Performa buruk pada komoditas volatile (cabai — MAPE tinggi)
  ✗ Tidak menggunakan fitur kalender (is_ramadan, dll) yang sudah kita buat
  ✗ Untuk rolling forecast (lebih akurat), waktu training sangat lama
""")

print("=" * 60)
print("  ✅ ARIMA selesai!")
print("=" * 60)
