"""
KomodiPrice — predict.py
================================================
FILE INI ADALAH INFERENCE ENGINE.
Role 3 (App Developer) hanya perlu import file ini.
Tidak perlu mengerti cara kerja Prophet sama sekali.

CARA PAKAI (dari file app manapun):
    from predict import get_prediction, get_all_latest_prices

    # Prediksi 14 hari ke depan untuk Beras
    result = get_prediction("Beras", horizon=14)
    print(result["dates"])        # list tanggal
    print(result["predictions"])  # list harga prediksi (Rp)
    print(result["lower"])        # batas bawah confidence interval
    print(result["upper"])        # batas atas confidence interval

    # Ambil harga terkini semua komoditas
    prices = get_all_latest_prices()
================================================
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prophet.serialize import model_from_json

# ============================================================
# PATH — sesuaikan jika struktur folder berbeda
# ============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
CLEAN_DATA  = os.path.join(PROC_DIR, "dataset_komodiprice_clean.csv")
PROPHET_PKL = os.path.join(MODEL_DIR, "prophet_results.pkl")

# ============================================================
# KONSTANTA
# ============================================================
COMMODITIES = [
    "Bawang Merah", "Bawang Putih", "Beras",
    "Cabai Merah",  "Cabai Rawit",  "Minyak Goreng"
]

CALENDAR_FEATS = [
    "is_ramadan", "is_lebaran_window", "is_nataru",
    "is_harvest_season", "is_weekend"
]

# Periode Ramadan untuk tahun mendatang (update jika perlu)
RAMADAN_PERIODS = [
    ("2026-02-18", "2026-03-19"),
    ("2027-02-07", "2027-03-08"),
]
LEBARAN_WINDOWS = [
    ("2026-03-13", "2026-03-26"),
    ("2027-03-02", "2027-03-15"),
]
NATARU_PERIODS = [
    ("2025-12-20", "2026-01-07"),
    ("2026-12-20", "2027-01-07"),
]
HARVEST_MONTHS = [3, 4, 9, 10]

# ============================================================
# LOAD MODEL (dilakukan sekali saat import)
# ============================================================
print("[predict.py] Loading Prophet models...")
with open(PROPHET_PKL, "rb") as f:
    _prophet_store = pickle.load(f)

_models = {}
for comm in COMMODITIES:
    if comm in _prophet_store:
        _models[comm] = model_from_json(_prophet_store[comm]["model_json"])

print(f"[predict.py] ✓ {len(_models)} model berhasil dimuat: {list(_models.keys())}")

# ============================================================
# LOAD HISTORICAL DATA
# ============================================================
_df_clean = pd.read_csv(CLEAN_DATA, parse_dates=["date"])


# ============================================================
# HELPER: tambahkan fitur kalender ke future dataframe
# ============================================================
def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom kalender ke dataframe dengan kolom 'ds'."""
    df = df.copy()
    df["is_weekend"]      = (df["ds"].dt.dayofweek >= 5).astype(int)
    df["is_ramadan"]      = 0
    df["is_lebaran_window"] = 0
    df["is_nataru"]       = 0
    df["is_harvest_season"] = df["ds"].dt.month.isin(HARVEST_MONTHS).astype(int)

    for start, end in RAMADAN_PERIODS:
        mask = (df["ds"] >= pd.Timestamp(start)) & (df["ds"] <= pd.Timestamp(end))
        df.loc[mask, "is_ramadan"] = 1

    for start, end in LEBARAN_WINDOWS:
        mask = (df["ds"] >= pd.Timestamp(start)) & (df["ds"] <= pd.Timestamp(end))
        df.loc[mask, "is_lebaran_window"] = 1

    for start, end in NATARU_PERIODS:
        mask = (df["ds"] >= pd.Timestamp(start)) & (df["ds"] <= pd.Timestamp(end))
        df.loc[mask, "is_nataru"] = 1

    return df


# ============================================================
# FUNGSI UTAMA 1: get_prediction()
# ============================================================
def get_prediction(commodity: str, horizon: int = 14) -> dict:
    """
    Prediksi harga komoditas N hari ke depan menggunakan Prophet.

    Args:
        commodity (str): Nama komoditas. Pilihan:
                         "Bawang Merah", "Bawang Putih", "Beras",
                         "Cabai Merah", "Cabai Rawit", "Minyak Goreng"
        horizon (int)  : Jumlah hari ke depan (7, 14, atau 30)

    Returns:
        dict dengan keys:
            commodity   : nama komoditas
            horizon     : jumlah hari
            dates       : list string tanggal ["2026-05-16", ...]
            predictions : list float harga prediksi (Rp)
            lower       : list float batas bawah 95% CI
            upper       : list float batas atas 95% CI
            last_actual : float harga terakhir yang diketahui
            last_date   : string tanggal terakhir data historis
            metrics     : dict MAE, RMSE, MAPE dari evaluasi training
            trend       : "naik" | "turun" | "stabil"
            pct_change  : float persentase perubahan dari hari ini ke hari terakhir prediksi
    """
    if commodity not in _models:
        raise ValueError(
            f"Komoditas '{commodity}' tidak ditemukan. "
            f"Pilihan: {COMMODITIES}"
        )
    if horizon not in [7, 14, 30]:
        horizon = min([7, 14, 30], key=lambda x: abs(x - horizon))

    model = _models[commodity]

    # Tanggal mulai prediksi = hari setelah data terakhir
    last_date = _df_clean["date"].max()
    future_dates = pd.date_range(
        start = last_date + timedelta(days=1),
        periods = horizon,
        freq = "D"
    )

    # Buat future dataframe
    future_df = pd.DataFrame({"ds": future_dates})
    future_df = _add_calendar_features(future_df)

    # Prediksi
    forecast = model.predict(future_df)

    predictions = np.maximum(forecast["yhat"].values,   0).tolist()
    lower       = np.maximum(forecast["yhat_lower"].values, 0).tolist()
    upper       = forecast["yhat_upper"].values.tolist()
    dates       = [str(d.date()) for d in future_dates]

    # Harga terakhir yang diketahui
    last_actual = float(_df_clean[commodity].iloc[-1])
    last_price  = last_actual

    # Trend
    first_pred  = predictions[0]
    last_pred   = predictions[-1]
    pct_change  = ((last_pred - last_price) / last_price) * 100

    if pct_change > 2:
        trend = "naik"
    elif pct_change < -2:
        trend = "turun"
    else:
        trend = "stabil"

    # Metrics dari training
    metrics = _prophet_store[commodity].get("metrics", {})

    return {
        "commodity"  : commodity,
        "horizon"    : horizon,
        "dates"      : dates,
        "predictions": predictions,
        "lower"      : lower,
        "upper"      : upper,
        "last_actual": last_actual,
        "last_date"  : str(last_date.date()),
        "metrics"    : metrics,
        "trend"      : trend,
        "pct_change" : round(pct_change, 2),
    }


# ============================================================
# FUNGSI UTAMA 2: get_all_latest_prices()
# ============================================================
def get_all_latest_prices() -> dict:
    """
    Ambil harga terkini semua komoditas + perubahan vs kemarin.

    Returns:
        dict: {
            "Beras": {
                "price_today"    : 14500,
                "price_yesterday": 14500,
                "change"         : 0,
                "change_pct"     : 0.0,
                "trend"          : "stabil"
            },
            ...
        }
    """
    result = {}
    df = _df_clean.sort_values("date")

    for comm in COMMODITIES:
        today_price = float(df[comm].iloc[-1])
        yest_price  = float(df[comm].iloc[-2]) if len(df) > 1 else today_price
        change      = today_price - yest_price
        change_pct  = (change / yest_price * 100) if yest_price != 0 else 0.0

        if change_pct > 0.5:
            trend = "naik"
        elif change_pct < -0.5:
            trend = "turun"
        else:
            trend = "stabil"

        result[comm] = {
            "price_today"    : today_price,
            "price_yesterday": yest_price,
            "change"         : round(change, 0),
            "change_pct"     : round(change_pct, 2),
            "trend"          : trend,
        }

    return result


# ============================================================
# FUNGSI UTAMA 3: get_history()
# ============================================================
def get_history(commodity: str, days: int = 90) -> dict:
    """
    Ambil data historis harga untuk grafik.

    Args:
        commodity (str): Nama komoditas
        days (int)     : Jumlah hari ke belakang (default 90)

    Returns:
        dict:
            dates  : list string tanggal
            prices : list float harga
    """
    if commodity not in COMMODITIES:
        raise ValueError(f"Komoditas '{commodity}' tidak valid.")

    df = _df_clean[["date", commodity]].dropna()
    df = df.sort_values("date").tail(days)

    return {
        "commodity": commodity,
        "dates"    : [str(d.date()) for d in df["date"]],
        "prices"   : df[commodity].tolist(),
    }


# ============================================================
# FUNGSI UTAMA 4: get_model_metrics()
# ============================================================
def get_model_metrics() -> dict:
    """
    Ambil metrik evaluasi model Prophet untuk semua komoditas.
    Berguna untuk halaman 'Tentang Model' di dashboard.

    Returns:
        dict: {commodity: {MAE, RMSE, MAPE}}
    """
    result = {}
    for comm in COMMODITIES:
        metrics = _prophet_store.get(comm, {}).get("metrics", {})
        result[comm] = {
            "MAE" : round(metrics.get("MAE",  0), 0),
            "RMSE": round(metrics.get("RMSE", 0), 0),
            "MAPE": round(metrics.get("MAPE", 0), 2),
        }
    return result


# ============================================================
# TEST — jalankan file ini langsung untuk verifikasi
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  TEST predict.py — verifikasi sebelum diserahkan ke Role 3")
    print("="*55)

    # Test 1: harga terkini
    print("\n[TEST 1] get_all_latest_prices()")
    prices = get_all_latest_prices()
    for comm, info in prices.items():
        arrow = "↑" if info["trend"] == "naik" else ("↓" if info["trend"] == "turun" else "→")
        print(f"  {comm:<15}: Rp{info['price_today']:>8,.0f}  "
              f"{arrow} {info['change_pct']:+.1f}%")

    # Test 2: prediksi 14 hari
    print("\n[TEST 2] get_prediction('Beras', horizon=14)")
    pred = get_prediction("Beras", horizon=14)
    print(f"  Komoditas  : {pred['commodity']}")
    print(f"  Horizon    : {pred['horizon']} hari")
    print(f"  Trend      : {pred['trend']}  ({pred['pct_change']:+.1f}%)")
    print(f"  {'Tanggal':<13} {'Prediksi':>12} {'Batas Bawah':>13} {'Batas Atas':>11}")
    print(f"  {'-'*52}")
    for dt, pr, lo, hi in zip(pred["dates"], pred["predictions"],
                               pred["lower"],  pred["upper"]):
        print(f"  {dt:<13} Rp{pr:>9,.0f} Rp{lo:>10,.0f} Rp{hi:>8,.0f}")

    # Test 3: histori
    print("\n[TEST 3] get_history('Cabai Merah', days=7)")
    hist = get_history("Cabai Merah", days=7)
    for dt, pr in zip(hist["dates"], hist["prices"]):
        print(f"  {dt}  Rp{pr:>10,.0f}")

    # Test 4: model metrics
    print("\n[TEST 4] get_model_metrics()")
    metrics = get_model_metrics()
    print(f"  {'Komoditas':<15} {'MAE':>12} {'RMSE':>12} {'MAPE':>8}")
    print(f"  {'-'*50}")
    for comm, m in metrics.items():
        print(f"  {comm:<15} Rp{m['MAE']:>9,.0f} Rp{m['RMSE']:>9,.0f} {m['MAPE']:>6.2f}%")

    print("\n  ✅ Semua fungsi berjalan normal. Siap diserahkan ke Role 3!")
    print("="*55)
