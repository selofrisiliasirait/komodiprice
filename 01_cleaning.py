"""
KomodiPrice — Step 1: Data Cleaning
Role 1: Data Specialist
============================================================
Input  : 6 file CSV raw di folder data/raw/
Output : data/processed/dataset_komodiprice_clean.csv  (wide)
         data/processed/dataset_komodiprice_long.csv   (long)
============================================================
"""

import os
import pandas as pd
import numpy as np

# Suppress TF warnings (tidak wajib, hanya agar output bersih)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# KONFIGURASI — sesuaikan jika nama file berbeda
# ============================================================
RAW_DIR  = "data/raw"
OUT_DIR  = "data/processed"

FILES = {
    "Bawang Merah"  : f"{RAW_DIR}/komoditas_bawang_merah_2022_2026.csv",
    "Bawang Putih"  : f"{RAW_DIR}/komoditas_bawang_putih_2022_2026.csv",
    "Beras"         : f"{RAW_DIR}/komoditas_beras_2022_2026.csv",
    "Cabai Merah"   : f"{RAW_DIR}/komoditas_cabai_merah_2022_2026.csv",
    "Cabai Rawit"   : f"{RAW_DIR}/komoditas_cabai_rawit_2022_2026.csv",
    "Minyak Goreng" : f"{RAW_DIR}/komoditas_minyak_goreng_2022_2026.csv",
}

TARGET_PROVINCE = "Sumatera Utara"

PRICE_COLS = [
    "Bawang Merah", "Bawang Putih", "Beras",
    "Cabai Merah", "Cabai Rawit", "Minyak Goreng"
]

CALENDAR_COLS = [
    "year", "month", "day", "day_of_week", "week_of_year",
    "quarter", "is_weekend", "is_ramadan",
    "is_lebaran_window", "is_nataru", "is_harvest_season"
]

# ============================================================
print("=" * 60)
print("  KomodiPrice — 01: Data Cleaning")
print("=" * 60)

# ── STEP 1: Load & filter Sumatera Utara ────────────────────
print("\n[STEP 1] Loading & filtering Sumatera Utara...")

all_series = {}

for commodity_name, filepath in FILES.items():
    df = pd.read_csv(filepath)

    df_sumut = df[df["Province_Name"] == TARGET_PROVINCE].copy()

    df_sumut = df_sumut.drop(columns=[
        "Date_Scraped", "Commodity_ID", "Province_ID",
        "Province_Name", "Commodity_Name", "Price_Type"
    ])

    df_sumut = df_sumut.rename(columns={"Date_Param": "date", "Price": commodity_name})
    df_sumut["date"] = pd.to_datetime(df_sumut["date"])
    df_sumut = df_sumut.set_index("date").sort_index()

    all_series[commodity_name] = df_sumut[commodity_name]

    print(f"  ✓ {commodity_name:15s} | {len(df_sumut)} baris | "
          f"{df_sumut.index.min().date()} → {df_sumut.index.max().date()}")

# ── STEP 2: Full date range + forward-fill missing dates ────
print("\n[STEP 2] Membuat date range lengkap & mengisi missing dates...")

global_min = min(s.index.min() for s in all_series.values())
global_max = max(s.index.max() for s in all_series.values())
full_range = pd.date_range(start=global_min, end=global_max, freq="D")

print(f"  Rentang   : {global_min.date()} → {global_max.date()}")
print(f"  Total hari: {len(full_range)}")

df_wide = pd.DataFrame(index=full_range)
df_wide.index.name = "date"

for commodity_name, series in all_series.items():
    reindexed = series.reindex(full_range)
    filled    = reindexed.ffill().bfill()
    df_wide[commodity_name] = filled
    n_filled = reindexed.isna().sum()
    print(f"  ✓ {commodity_name:15s} | {n_filled} tanggal di-forward-fill")

# ── STEP 3: Fitur kalender & kontekstual ────────────────────
print("\n[STEP 3] Menambahkan fitur kalender...")

df_wide = df_wide.reset_index()

df_wide["year"]         = df_wide["date"].dt.year
df_wide["month"]        = df_wide["date"].dt.month
df_wide["day"]          = df_wide["date"].dt.day
df_wide["day_of_week"]  = df_wide["date"].dt.dayofweek
df_wide["week_of_year"] = df_wide["date"].dt.isocalendar().week.astype(int)
df_wide["quarter"]      = df_wide["date"].dt.quarter
df_wide["is_weekend"]   = (df_wide["day_of_week"] >= 5).astype(int)

# Ramadan (estimasi per tahun)
ramadan_periods = [
    ("2022-04-02", "2022-05-01"),
    ("2023-03-23", "2023-04-21"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
]
df_wide["is_ramadan"] = 0
for start, end in ramadan_periods:
    mask = (df_wide["date"] >= start) & (df_wide["date"] <= end)
    df_wide.loc[mask, "is_ramadan"] = 1

# Lebaran window H-7 s/d H+7
lebaran_dates = ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31"]
df_wide["is_lebaran_window"] = 0
for ld in lebaran_dates:
    ts = pd.Timestamp(ld)
    mask = (df_wide["date"] >= ts - pd.Timedelta(days=7)) & \
           (df_wide["date"] <= ts + pd.Timedelta(days=7))
    df_wide.loc[mask, "is_lebaran_window"] = 1

# Natal & Tahun Baru
df_wide["is_nataru"] = (
    ((df_wide["month"] == 12) & (df_wide["day"] >= 20)) |
    ((df_wide["month"] == 1)  & (df_wide["day"] <= 7))
).astype(int)

# Musim panen (estimasi Sumatera Utara)
df_wide["is_harvest_season"] = df_wide["month"].isin([3, 4, 5, 9, 10]).astype(int)

print("  ✓ year, month, day, day_of_week, week_of_year, quarter")
print("  ✓ is_weekend, is_ramadan, is_lebaran_window, is_nataru, is_harvest_season")

# ── STEP 4: Validasi & susun kolom ──────────────────────────
print("\n[STEP 4] Validasi akhir...")

df_wide = df_wide[["date"] + PRICE_COLS + CALENDAR_COLS]

assert df_wide.isnull().sum().sum() == 0,         "❌ Ada null values!"
assert df_wide.duplicated(subset=["date"]).sum() == 0, "❌ Ada duplikat tanggal!"

print(f"  ✓ Shape         : {df_wide.shape}")
print(f"  ✓ Null values   : 0")
print(f"  ✓ Duplikat      : 0")
print(f"  ✓ Tanggal       : {df_wide['date'].min().date()} → {df_wide['date'].max().date()}")

# ── STEP 5: Export ───────────────────────────────────────────
print("\n[STEP 5] Menyimpan dataset...")

os.makedirs(OUT_DIR, exist_ok=True)

# Wide format
path_wide = f"{OUT_DIR}/dataset_komodiprice_clean.csv"
df_wide.to_csv(path_wide, index=False)
print(f"  ✓ Wide : {path_wide}")
print(f"    → {len(df_wide)} baris × {len(df_wide.columns)} kolom")

# Long format
df_long = df_wide.melt(
    id_vars=["date"] + CALENDAR_COLS,
    value_vars=PRICE_COLS,
    var_name="commodity",
    value_name="price"
)
df_long = df_long.sort_values(["commodity", "date"]).reset_index(drop=True)

path_long = f"{OUT_DIR}/dataset_komodiprice_long.csv"
df_long.to_csv(path_long, index=False)
print(f"  ✓ Long : {path_long}")
print(f"    → {len(df_long)} baris × {len(df_long.columns)} kolom")

# ── Ringkasan statistik harga ────────────────────────────────
print("\n" + "=" * 60)
print("  STATISTIK HARGA (Rp/kg) — Sumatera Utara")
print("=" * 60)
print(f"  {'Komoditas':15s} | {'Rata-rata':>12s} | {'Min':>10s} | {'Max':>10s}")
print("  " + "-" * 54)
for col in PRICE_COLS:
    print(f"  {col:15s} | Rp {df_wide[col].mean():9,.0f} | "
          f"Rp {df_wide[col].min():7,.0f} | Rp {df_wide[col].max():7,.0f}")

print("\n  ✅ Cleaning selesai!")
print("=" * 60)
