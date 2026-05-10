"""
KomodiPrice — Step 3: Preprocessing untuk AI Training
Role 1: Data Specialist
============================================================
Input  : data/processed/dataset_komodiprice_clean.csv
Output : data/processed/arima_data.pkl
         data/processed/prophet_data.pkl
         data/processed/lstm_data.pkl
         data/processed/scalers_X.pkl
         data/processed/scalers_y.pkl
         outputs/prep_01_acf_pacf.png
         outputs/prep_02_split_lstm.png
============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
INPUT     = "data/processed/dataset_komodiprice_clean.csv"
OUT_DIR   = "data/processed"
PLOT_DIR  = "outputs"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

COMMODITIES = ['Bawang Merah','Bawang Putih','Beras',
               'Cabai Merah','Cabai Rawit','Minyak Goreng']
COLORS      = ['#E85D24','#F2A623','#3B8BD4',
               '#D4537E','#1D9E75','#7F77DD']

CALENDAR_FEATS = [
    'day_of_week','month','week_of_year','quarter',
    'is_weekend','is_ramadan','is_lebaran_window',
    'is_nataru','is_harvest_season'
]

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
SEQ_LEN     = 60        # panjang window LSTM (hari)

print("=" * 60)
print("  KomodiPrice — 03: Preprocessing")
print("=" * 60)

df = pd.read_csv(INPUT)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

n        = len(df)
n_train  = int(n * TRAIN_RATIO)
n_val    = int(n * VAL_RATIO)
n_test   = n - n_train - n_val

print(f"\n  Dataset : {n} hari")
print(f"  Train   : {n_train} hari ({TRAIN_RATIO*100:.0f}%)")
print(f"  Val     : {n_val} hari ({VAL_RATIO*100:.0f}%)")
print(f"  Test    : {n_test} hari ({(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%)")

# ──────────────────────────────────────────────────────────────
# BAGIAN A — Stationarity Check
# ──────────────────────────────────────────────────────────────
print("\n─" * 30)
print("[A] Stationarity Check (ADF + KPSS)")

stationarity = {}
for comm in COMMODITIES:
    series = df[comm].values
    adf_p  = adfuller(series, autolag='AIC')[1]
    kpss_p = kpss(series, regression='c', nlags='auto')[1]
    adf_ok  = adf_p  < 0.05
    kpss_ok = kpss_p > 0.05
    d = 0 if (adf_ok and kpss_ok) else 1
    verdict = "STATIONARY" if d==0 else "NON-STATIONARY → d=1"
    stationarity[comm] = d
    print(f"  {comm:15s} | ADF p={adf_p:.4f} {'✓' if adf_ok else '✗'} | "
          f"KPSS p={kpss_p:.4f} {'✓' if kpss_ok else '✗'} | {verdict}")

# ──────────────────────────────────────────────────────────────
# BAGIAN B — ACF / PACF Plot
# ──────────────────────────────────────────────────────────────
print("\n─" * 30)
print("[B] Membuat plot ACF & PACF...")

fig, axes = plt.subplots(len(COMMODITIES), 2, figsize=(16, 22))
fig.patch.set_facecolor('#0F1117')

for i, (comm, color) in enumerate(zip(COMMODITIES, COLORS)):
    series_diff = df[comm].diff().dropna()
    ax_acf  = axes[i][0]
    ax_pacf = axes[i][1]
    for ax in [ax_acf, ax_pacf]:
        ax.set_facecolor('#1A1D27')
        for sp in ax.spines.values(): sp.set_color('#2E3145')
    plot_acf( series_diff, lags=40, ax=ax_acf,  color=color, alpha=0.3,
              vlines_kwargs={'colors': color}, title='')
    plot_pacf(series_diff, lags=40, ax=ax_pacf, color=color, alpha=0.3,
              vlines_kwargs={'colors': color}, title='', method='ywm')
    for ax in [ax_acf, ax_pacf]:
        ax.tick_params(colors='#8890A4', labelsize=8)
        ax.set_facecolor('#1A1D27')
        for coll in ax.collections:
            coll.set_facecolor('#252838')
    ax_acf.set_title( f'{comm} — ACF (diff 1)',  color=color, fontsize=10, fontweight='bold')
    ax_pacf.set_title(f'{comm} — PACF (diff 1)', color=color, fontsize=10, fontweight='bold')

fig.suptitle('ACF & PACF per Komoditas — Menentukan order ARIMA(p,d,q)',
             color='white', fontsize=13, fontweight='bold', y=1.005)
plt.tight_layout()
path = f"{PLOT_DIR}/prep_01_acf_pacf.png"
plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ──────────────────────────────────────────────────────────────
# BAGIAN C — Split Visualization
# ──────────────────────────────────────────────────────────────
print("[C] Membuat plot train/val/test split...")

fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('#0F1117')
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

axes_split = []
for i in range(6):
    r, c = divmod(i, 2)
    axes_split.append(fig.add_subplot(gs[r, c]))

for i, (comm, color) in enumerate(zip(COMMODITIES, COLORS)):
    ax = axes_split[i]
    ax.set_facecolor('#1A1D27')
    for sp in ax.spines.values(): sp.set_color('#2E3145')
    dates  = df['date']
    prices = df[comm]
    ax.plot(dates[:n_train], prices[:n_train], color=color, linewidth=1.0, label='Train')
    ax.plot(dates[n_train:n_train+n_val], prices[n_train:n_train+n_val],
            color='#F2A623', linewidth=1.0, label='Validasi')
    ax.plot(dates[n_train+n_val:], prices[n_train+n_val:],
            color='white', linewidth=1.2, label='Test')
    ax.axvline(dates.iloc[n_train],       color='#F2A623', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.axvline(dates.iloc[n_train+n_val], color='white',   linewidth=1.2, linestyle='--', alpha=0.6)
    ax.axvspan(dates.iloc[0], dates.iloc[n_train-1],           alpha=0.06, color=color)
    ax.axvspan(dates.iloc[n_train], dates.iloc[n_train+n_val-1], alpha=0.10, color='#F2A623')
    ax.axvspan(dates.iloc[n_train+n_val], dates.iloc[-1],      alpha=0.12, color='white')
    ax.set_title(comm, color=color, fontsize=11, fontweight='bold', pad=6)
    ax.tick_params(colors='#8890A4', labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
    ax.set_ylabel('Rp/kg', color='#8890A4', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(color='#2E3145', linewidth=0.4, alpha=0.5)
    if i == 0:
        ax.legend(fontsize=8, facecolor='#252838',
                  edgecolor='#3E4260', labelcolor='white', loc='upper left')

# LSTM window diagram
ax_lstm = fig.add_subplot(gs[3, :])
ax_lstm.set_facecolor('#1A1D27')
for sp in ax_lstm.spines.values(): sp.set_color('#2E3145')
ax_lstm.set_xlim(0, 200); ax_lstm.set_ylim(0, 60); ax_lstm.axis('off')
ax_lstm.set_title(f'Cara LSTM Membaca Data — Sliding Window (seq_len={SEQ_LEN} hari)',
                  color='white', fontsize=11, fontweight='bold', pad=8)
for j in range(6):
    xs = 10 + j*30
    ax_lstm.add_patch(plt.Rectangle((xs,25),24,20,linewidth=1.0,
                      edgecolor='#7F77DD',facecolor='#252838',alpha=0.9))
    ax_lstm.text(xs+12,35,f'{SEQ_LEN}\nhari',ha='center',va='center',
                 color='#B0B8CC',fontsize=7.5)
    ax_lstm.annotate('',xy=(xs+26,35),xytext=(xs+24,35),
                     arrowprops=dict(arrowstyle='->',color='#7F77DD',lw=1.2))
    ax_lstm.add_patch(plt.Rectangle((xs+26,29),10,12,linewidth=1.0,
                      edgecolor='#E85D24',facecolor='#2A1A1A',alpha=0.9))
    ax_lstm.text(xs+31,35,'H+1',ha='center',va='center',color='#E85D24',fontsize=7)
ax_lstm.text(5,15,'→ Setiap window geser 1 hari (sliding)',color='#8890A4',fontsize=9)
n_seq = n - SEQ_LEN
ax_lstm.text(5,8,f'→ Total: {n_seq} sequences | Input shape: ({int(n_seq*TRAIN_RATIO)}, {SEQ_LEN}, {len(COMMODITIES)+len(CALENDAR_FEATS)}) | Output: harga H+1',
             color='#8890A4',fontsize=9)
ax_lstm.add_patch(plt.Rectangle((130,38),5,8,color='#7F77DD',alpha=0.8))
ax_lstm.text(137,42,'= Input: 60 hari × 15 fitur',color='#B0B8CC',fontsize=8.5,va='center')
ax_lstm.add_patch(plt.Rectangle((130,25),5,8,color='#E85D24',alpha=0.8))
ax_lstm.text(137,29,'= Output: harga hari ke-61',color='#B0B8CC',fontsize=8.5,va='center')

fig.suptitle('Data Split & LSTM Sliding Window — KomodiPrice',
             color='white', fontsize=14, fontweight='bold', y=1.005)
path = f"{PLOT_DIR}/prep_02_split_lstm.png"
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ──────────────────────────────────────────────────────────────
# BAGIAN D — ARIMA data
# ──────────────────────────────────────────────────────────────
print("\n─" * 30)
print("[D] Menyiapkan data ARIMA...")

arima_data = {}
for comm in COMMODITIES:
    series = df[['date', comm]].rename(columns={comm:'price'}).set_index('date')
    train = series.iloc[:n_train]
    val   = series.iloc[n_train:n_train+n_val]
    test  = series.iloc[n_train+n_val:]
    arima_data[comm] = {
        'full': series, 'train': train, 'val': val, 'test': test,
        'd_order': stationarity[comm],
        'p_range': range(0,4), 'q_range': range(0,4),
    }
    print(f"  {comm:15s} | train={len(train)} val={len(val)} test={len(test)} | d={stationarity[comm]}")

with open(f"{OUT_DIR}/arima_data.pkl",'wb') as f: pickle.dump(arima_data, f)
print(f"  ✓ Disimpan: {OUT_DIR}/arima_data.pkl")

# ──────────────────────────────────────────────────────────────
# BAGIAN E — Prophet data
# ──────────────────────────────────────────────────────────────
print("\n─" * 30)
print("[E] Menyiapkan data Prophet...")

prophet_data = {}
for comm in COMMODITIES:
    pdf = df[['date', comm] + CALENDAR_FEATS].copy()
    pdf = pdf.rename(columns={'date':'ds', comm:'y'})
    train     = pdf.iloc[:n_train].reset_index(drop=True)
    val       = pdf.iloc[n_train:n_train+n_val].reset_index(drop=True)
    test      = pdf.iloc[n_train+n_val:].reset_index(drop=True)
    train_val = pdf.iloc[:n_train+n_val].reset_index(drop=True)
    prophet_data[comm] = {
        'train': train, 'val': val, 'test': test,
        'train_val': train_val, 'full': pdf,
        'regressors': CALENDAR_FEATS,
    }
    print(f"  {comm:15s} | train={len(train)} val={len(val)} test={len(test)}")

with open(f"{OUT_DIR}/prophet_data.pkl",'wb') as f: pickle.dump(prophet_data, f)
print(f"  ✓ Disimpan: {OUT_DIR}/prophet_data.pkl")

# ──────────────────────────────────────────────────────────────
# BAGIAN F — LSTM data
# ──────────────────────────────────────────────────────────────
print("\n─" * 30)
print(f"[F] Menyiapkan data LSTM (seq_len={SEQ_LEN})...")

LSTM_FEATURES = COMMODITIES + CALENDAR_FEATS   # 15 fitur total

def create_sequences(X_sc, y_sc, seq_len):
    Xs, ys = [], []
    for i in range(len(X_sc) - seq_len):
        Xs.append(X_sc[i:i+seq_len])
        ys.append(y_sc[i+seq_len])
    return np.array(Xs), np.array(ys)

lstm_data = {}
scalers_X = {}
scalers_y = {}

for comm in COMMODITIES:
    X_df  = df[LSTM_FEATURES].values
    y_arr = df[comm].values.reshape(-1,1)

    scaler_X = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    scaler_X.fit(X_df[:n_train])
    scaler_y.fit(y_arr[:n_train])

    X_sc = scaler_X.transform(X_df)
    y_sc = scaler_y.transform(y_arr).flatten()

    X_seq, y_seq = create_sequences(X_sc, y_sc, SEQ_LEN)

    seq_n_train = n_train - SEQ_LEN
    seq_n_val   = n_val

    X_train = X_seq[:seq_n_train];           y_train = y_seq[:seq_n_train]
    X_val   = X_seq[seq_n_train:seq_n_train+seq_n_val]
    y_val   = y_seq[seq_n_train:seq_n_train+seq_n_val]
    X_test  = X_seq[seq_n_train+seq_n_val:]; y_test  = y_seq[seq_n_train+seq_n_val:]

    lstm_data[comm] = {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'seq_len':       SEQ_LEN,
        'n_features':    len(LSTM_FEATURES),
        'feature_names': LSTM_FEATURES,
        'scaler_X':      scaler_X,
        'scaler_y':      scaler_y,
        'last_sequence': X_sc[-SEQ_LEN:],
    }
    scalers_X[comm] = scaler_X
    scalers_y[comm] = scaler_y

    print(f"  {comm:15s} | X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")

with open(f"{OUT_DIR}/lstm_data.pkl",  'wb') as f: pickle.dump(lstm_data,  f)
with open(f"{OUT_DIR}/scalers_X.pkl",  'wb') as f: pickle.dump(scalers_X,  f)
with open(f"{OUT_DIR}/scalers_y.pkl",  'wb') as f: pickle.dump(scalers_y,  f)
print(f"  ✓ Disimpan: lstm_data.pkl, scalers_X.pkl, scalers_y.pkl")

# ──────────────────────────────────────────────────────────────
# RINGKASAN
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RINGKASAN PREPROCESSING")
print("=" * 60)
print(f"  Data split      : {n_train} train / {n_val} val / {n_test} test")
print(f"  LSTM seq_len    : {SEQ_LEN} hari")
print(f"  LSTM fitur      : {len(LSTM_FEATURES)} (6 harga + 9 kalender)")
print(f"  File output     : arima_data.pkl, prophet_data.pkl,")
print(f"                    lstm_data.pkl, scalers_X.pkl, scalers_y.pkl")
print(f"  Grafik output   : prep_01_acf_pacf.png, prep_02_split_lstm.png")
print(f"\n  ✅ Preprocessing selesai! Data siap untuk Role 2 (Model Training).")
print("=" * 60)
