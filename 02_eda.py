"""
KomodiPrice — Step 2: Exploratory Data Analysis (EDA)
Role 1: Data Specialist
============================================================
Input  : data/processed/dataset_komodiprice_clean.csv
Output : outputs/eda_01_tren_harga.png
         outputs/eda_02_korelasi.png
         outputs/eda_03_seasonality.png
         outputs/eda_04_distribusi.png
         outputs/eda_05_yoy_ramadan.png
         outputs/eda_06_index_outlier.png
============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (aman untuk semua OS)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
INPUT   = "data/processed/dataset_komodiprice_clean.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

COMMODITIES = ['Bawang Merah','Bawang Putih','Beras',
               'Cabai Merah','Cabai Rawit','Minyak Goreng']
COLORS      = ['#E85D24','#F2A623','#3B8BD4',
               '#D4537E','#1D9E75','#7F77DD']
SHORT_NAMES = ['B.Merah','B.Putih','Beras','C.Merah','C.Rawit','M.Goreng']
MONTHS      = ['Jan','Feb','Mar','Apr','Mei','Jun',
               'Jul','Agu','Sep','Okt','Nov','Des']

RAMADAN_PERIODS = [
    ("2022-04-02","2022-05-01"),("2023-03-23","2023-04-21"),
    ("2024-03-11","2024-04-09"),("2025-03-01","2025-03-30"),
]

print("=" * 60)
print("  KomodiPrice — 02: EDA")
print("=" * 60)

df = pd.read_csv(INPUT)
df['date'] = pd.to_datetime(df['date'])
print(f"\n  Dataset dimuat: {df.shape[0]} baris × {df.shape[1]} kolom\n")

# ── GRAFIK 1: Tren harga semua komoditas ────────────────────
print("[1/6] Tren harga per komoditas...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0F1117')
axes = axes.flatten()

for i, (comm, color) in enumerate(zip(COMMODITIES, COLORS)):
    ax = axes[i]
    ax.set_facecolor('#1A1D27')
    for sp in ax.spines.values(): sp.set_color('#2E3145')

    for start, end in RAMADAN_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color='#F2A623', zorder=0)

    ax.plot(df['date'], df[comm], color=color, linewidth=1.1, alpha=0.9)
    roll = df[comm].rolling(30).mean()
    ax.plot(df['date'], roll, color='white', linewidth=1.8,
            alpha=0.5, linestyle='--', label='MA-30 hari')

    mean_p = df[comm].mean()
    ax.axhline(mean_p, color=color, linewidth=0.8, linestyle=':', alpha=0.6)

    stats = (f"Rata-rata: Rp {mean_p:,.0f}\n"
             f"Maks: Rp {df[comm].max():,.0f}\n"
             f"Min:  Rp {df[comm].min():,.0f}")
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=8,
            color='#B0B8CC', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#252838',
                      alpha=0.8, edgecolor='#3E4260'))

    ax.set_title(comm, color='white', fontsize=13, fontweight='bold', pad=8)
    ax.tick_params(colors='#8890A4', labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
    ax.set_ylabel('Rp/kg', color='#8890A4', fontsize=9)
    ax.grid(axis='y', color='#2E3145', linewidth=0.5, alpha=0.7)
    if i == 0:
        ax.plot([], [], color='#F2A623', alpha=0.5, linewidth=8, label='Ramadan')
        ax.legend(fontsize=8, facecolor='#252838',
                  edgecolor='#3E4260', labelcolor='white')

fig.suptitle('Tren Harga Komoditas — Sumatera Utara (Jan 2022 – Feb 2026)',
             color='white', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
path = f"{OUT_DIR}/eda_01_tren_harga.png"
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ── GRAFIK 2: Korelasi ───────────────────────────────────────
print("[2/6] Heatmap korelasi...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0F1117')

# Heatmap
ax1 = axes[0]
ax1.set_facecolor('#1A1D27')
corr = df[COMMODITIES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220,20,as_cmap=True),
            vmax=1, vmin=-1, center=0, annot=True, fmt='.2f',
            linewidths=0.5, linecolor='#1A1D27', ax=ax1,
            annot_kws={"size":11,"color":"white","fontweight":"bold"},
            cbar_kws={"shrink":0.8})
ax1.set_facecolor('#1A1D27')
ax1.set_title('Korelasi Harga Antar Komoditas', color='white',
              fontsize=13, fontweight='bold', pad=12)
ax1.set_xticklabels(SHORT_NAMES, rotation=30, ha='right', color='#B0B8CC')
ax1.set_yticklabels(SHORT_NAMES, rotation=0, color='#B0B8CC')
ax1.collections[0].colorbar.ax.tick_params(colors='#8890A4')

# Scatter cabai
ax2 = axes[1]
ax2.set_facecolor('#1A1D27')
for sp in ax2.spines.values(): sp.set_color('#2E3145')
sc = ax2.scatter(df['Cabai Merah'], df['Cabai Rawit'],
                 c=df['date'].map(lambda x: x.toordinal()),
                 cmap='plasma', alpha=0.5, s=15)
z = np.polyfit(df['Cabai Merah'], df['Cabai Rawit'], 1)
xline = np.linspace(df['Cabai Merah'].min(), df['Cabai Merah'].max(), 100)
ax2.plot(xline, np.poly1d(z)(xline), color='#F2A623', linewidth=2,
         linestyle='--', label='Tren linear')
fig.colorbar(sc, ax=ax2, shrink=0.8).ax.tick_params(colors='#8890A4', labelsize=8)
r_val = corr.loc['Cabai Merah','Cabai Rawit']
ax2.set_title(f'Cabai Merah vs Cabai Rawit (r = {r_val:.2f})', color='white',
              fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Cabai Merah (Rp/kg)', color='#8890A4', fontsize=10)
ax2.set_ylabel('Cabai Rawit (Rp/kg)', color='#8890A4', fontsize=10)
ax2.tick_params(colors='#8890A4', labelsize=9)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
ax2.grid(color='#2E3145', linewidth=0.5, alpha=0.5)
ax2.legend(fontsize=9, facecolor='#252838', edgecolor='#3E4260', labelcolor='white')

fig.suptitle('Analisis Korelasi — Sumatera Utara', color='white',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = f"{OUT_DIR}/eda_02_korelasi.png"
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ── GRAFIK 3: Seasonality bulanan ───────────────────────────
print("[3/6] Pola musiman bulanan...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#0F1117')
axes = axes.flatten()

for i, (comm, color) in enumerate(zip(COMMODITIES, COLORS)):
    ax = axes[i]
    ax.set_facecolor('#1A1D27')
    for sp in ax.spines.values(): sp.set_color('#2E3145')

    monthly = df.groupby('month')[comm].mean()
    overall = df[comm].mean()

    bars = ax.bar(range(1,13), monthly.values, color=color,
                  alpha=0.75, edgecolor='none', width=0.7, zorder=2)
    for bar, val in zip(bars, monthly.values):
        if val > overall * 1.05:
            bar.set_alpha(1.0)
            bar.set_edgecolor('white')
            bar.set_linewidth(0.8)

    ax.axhline(overall, color='white', linewidth=1.2, linestyle='--',
               alpha=0.6, label=f'Rata-rata Rp {overall:,.0f}')

    for m in [3, 4]:
        ax.axvspan(m-0.5, m+0.5, alpha=0.12, color='#F2A623', zorder=0)
    for m in [12, 1]:
        ax.axvspan(m-0.5, m+0.5, alpha=0.10, color='#3B8BD4', zorder=0)

    peak = monthly.idxmax()
    ax.text(peak, monthly[peak]*1.01, f'↑{MONTHS[peak-1]}',
            ha='center', fontsize=8, color='white', fontweight='bold')

    ax.set_title(comm, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(MONTHS, color='#8890A4', fontsize=8)
    ax.tick_params(colors='#8890A4', labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
    ax.set_ylabel('Rp/kg', color='#8890A4', fontsize=9)
    ax.grid(axis='y', color='#2E3145', linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7.5, facecolor='#252838', edgecolor='#3E4260', labelcolor='white')

axes[0].plot([], [], color='#F2A623', alpha=0.5, linewidth=8, label='Ramadan (Mar–Apr)')
axes[0].plot([], [], color='#3B8BD4', alpha=0.5, linewidth=8, label='Nataru (Des–Jan)')
axes[0].legend(fontsize=8, facecolor='#252838', edgecolor='#3E4260',
               labelcolor='white', loc='upper left')

fig.suptitle('Pola Musiman Bulanan — Rata-rata Harga per Bulan (2022–2026)',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
path = f"{OUT_DIR}/eda_03_seasonality.png"
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ── GRAFIK 4: Distribusi & volatilitas ──────────────────────
print("[4/6] Distribusi & volatilitas...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0F1117')

ax1 = axes[0]
ax1.set_facecolor('#1A1D27')
for sp in ax1.spines.values(): sp.set_color('#2E3145')
bp = ax1.boxplot([df[c].values for c in COMMODITIES], patch_artist=True,
                 notch=True,
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(color='#8890A4'),
                 capprops=dict(color='#8890A4'),
                 flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_xticks(range(1,7))
ax1.set_xticklabels(SHORT_NAMES, color='#B0B8CC', fontsize=9, rotation=20)
ax1.tick_params(colors='#8890A4', labelsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
ax1.set_ylabel('Harga (Rp/kg)', color='#8890A4', fontsize=10)
ax1.set_title('Distribusi Harga per Komoditas', color='white',
              fontsize=13, fontweight='bold', pad=10)
ax1.grid(axis='y', color='#2E3145', linewidth=0.5, alpha=0.6)

ax2 = axes[1]
ax2.set_facecolor('#1A1D27')
for sp in ax2.spines.values(): sp.set_color('#2E3145')
cv_vals = [(df[c].std()/df[c].mean())*100 for c in COMMODITIES]
bars = ax2.barh(range(len(COMMODITIES)), cv_vals, color=COLORS,
                alpha=0.8, edgecolor='none', height=0.6)
for bar, val in zip(bars, cv_vals):
    ax2.text(val+0.3, bar.get_y()+bar.get_height()/2,
             f'{val:.1f}%', va='center', ha='left',
             color='white', fontsize=10, fontweight='bold')
ax2.set_yticks(range(len(COMMODITIES)))
ax2.set_yticklabels(SHORT_NAMES, color='#B0B8CC', fontsize=10)
ax2.set_xlabel('Coefficient of Variation (%)', color='#8890A4', fontsize=10)
ax2.set_title('Tingkat Volatilitas Harga (CV %)', color='white',
              fontsize=13, fontweight='bold', pad=10)
ax2.tick_params(colors='#8890A4', labelsize=9)
ax2.grid(axis='x', color='#2E3145', linewidth=0.5, alpha=0.6)
ax2.axvline(10, color='#F2A623', linewidth=1, linestyle='--', alpha=0.7)
ax2.axvline(20, color='#E85D24', linewidth=1, linestyle='--', alpha=0.7)
ax2.text(10.2, -0.7, 'Sedang', color='#F2A623', fontsize=8)
ax2.text(20.2, -0.7, 'Tinggi', color='#E85D24', fontsize=8)

fig.suptitle('Distribusi & Volatilitas — Sumatera Utara', color='white',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = f"{OUT_DIR}/eda_04_distribusi.png"
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ── GRAFIK 5: YoY & Efek Ramadan ────────────────────────────
print("[5/6] YoY & efek Ramadan...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0F1117')

# YoY
ax1 = axes[0]
ax1.set_facecolor('#1A1D27')
for sp in ax1.spines.values(): sp.set_color('#2E3145')
x = np.arange(len(COMMODITIES))
width = 0.2
yr_colors = ['#3B8BD4','#1D9E75','#F2A623','#E85D24']
for yi, year in enumerate([2023,2024,2025,2026]):
    y_data = df[df['year']==year]
    prev   = df[df['year']==year-1]
    if len(y_data)==0 or len(prev)==0: continue
    changes = [(y_data[c].mean()-prev[c].mean())/prev[c].mean()*100 for c in COMMODITIES]
    offset  = (yi-1.5)*width
    bars = ax1.bar(x+offset, changes, width, alpha=0.8,
                   edgecolor='none', label=str(year), color=yr_colors[yi])
ax1.axhline(0, color='white', linewidth=0.8, alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(SHORT_NAMES, color='#B0B8CC', fontsize=9, rotation=20)
ax1.set_ylabel('Perubahan YoY (%)', color='#8890A4', fontsize=10)
ax1.set_title('Perubahan Harga Year-over-Year', color='white',
              fontsize=13, fontweight='bold', pad=10)
ax1.tick_params(colors='#8890A4', labelsize=9)
ax1.legend(fontsize=9, facecolor='#252838', edgecolor='#3E4260', labelcolor='white')
ax1.grid(axis='y', color='#2E3145', linewidth=0.5, alpha=0.6)

# Ramadan effect
ax2 = axes[1]
ax2.set_facecolor('#1A1D27')
for sp in ax2.spines.values(): sp.set_color('#2E3145')
ram_mean    = df[df['is_ramadan']==1][COMMODITIES].mean()
normal_mean = df[df['is_ramadan']==0][COMMODITIES].mean()
effects     = (ram_mean - normal_mean) / normal_mean * 100
c_bars = ['#E85D24' if v>0 else '#3B8BD4' for v in effects.values]
bars = ax2.bar(range(len(COMMODITIES)), effects.values, color=c_bars,
               alpha=0.85, edgecolor='none', width=0.6)
for bar, val in zip(bars, effects.values):
    ypos = val+0.3 if val>=0 else val-1.8
    ax2.text(bar.get_x()+bar.get_width()/2, ypos,
             f'{val:+.1f}%', ha='center', fontsize=9.5,
             color='white', fontweight='bold')
ax2.axhline(0, color='white', linewidth=0.8, alpha=0.5)
ax2.set_xticks(range(len(COMMODITIES)))
ax2.set_xticklabels(SHORT_NAMES, color='#B0B8CC', fontsize=9, rotation=20)
ax2.set_ylabel('Selisih vs Hari Biasa (%)', color='#8890A4', fontsize=10)
ax2.set_title('Efek Ramadan terhadap Harga', color='white',
              fontsize=13, fontweight='bold', pad=10)
ax2.tick_params(colors='#8890A4', labelsize=9)
ax2.grid(axis='y', color='#2E3145', linewidth=0.5, alpha=0.6)

fig.suptitle('YoY & Efek Musiman — Sumatera Utara', color='white',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = f"{OUT_DIR}/eda_05_yoy_ramadan.png"
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ── GRAFIK 6: Indeks ternormalisasi & outlier ────────────────
print("[6/6] Indeks & deteksi outlier...")

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.patch.set_facecolor('#0F1117')

ax1 = axes[0]
ax1.set_facecolor('#1A1D27')
for sp in ax1.spines.values(): sp.set_color('#2E3145')
for comm, color in zip(COMMODITIES, COLORS):
    base    = df[df['date']==df['date'].min()][comm].values[0]
    indexed = df[comm] / base * 100
    ax1.plot(df['date'], indexed, color=color, linewidth=1.4,
             alpha=0.85, label=comm)
for start, end in RAMADAN_PERIODS:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.10, color='#F2A623', zorder=0)
ax1.axhline(100, color='white', linewidth=1, linestyle=':', alpha=0.4)
ax1.set_title('Indeks Harga Ternormalisasi (Jan 2022 = 100)',
              color='white', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Indeks', color='#8890A4', fontsize=10)
ax1.tick_params(colors='#8890A4', labelsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax1.grid(color='#2E3145', linewidth=0.5, alpha=0.5)
ax1.legend(fontsize=9, facecolor='#252838', edgecolor='#3E4260',
           labelcolor='white', ncol=3, loc='upper left')

ax2 = axes[1]
ax2.set_facecolor('#1A1D27')
for sp in ax2.spines.values(): sp.set_color('#2E3145')
for comm, color in zip(COMMODITIES, COLORS):
    daily = df[comm].pct_change() * 100
    Q1, Q3 = daily.quantile(0.25), daily.quantile(0.75)
    IQR = Q3 - Q1
    outliers = daily[(daily < Q1-3*IQR) | (daily > Q3+3*IQR)]
    ax2.plot(df['date'], daily, color=color, linewidth=0.6,
             alpha=0.5, label=f'{comm} ({len(outliers)} outlier)')
    if len(outliers):
        ax2.scatter(df['date'][outliers.index], outliers.values,
                    color=color, s=30, zorder=5, alpha=0.9)
ax2.axhline(0, color='white', linewidth=0.8, alpha=0.4)
ax2.set_ylim(-30, 30)
ax2.set_title('Perubahan Harga Harian (%) & Deteksi Outlier',
              color='white', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('Perubahan (%)', color='#8890A4', fontsize=10)
ax2.tick_params(colors='#8890A4', labelsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax2.grid(color='#2E3145', linewidth=0.5, alpha=0.5)
ax2.legend(fontsize=8.5, facecolor='#252838', edgecolor='#3E4260',
           labelcolor='white', ncol=3, loc='upper right')

fig.suptitle('Indeks & Dinamika Harian — Sumatera Utara',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
path = f"{OUT_DIR}/eda_06_index_outlier.png"
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print(f"  ✓ Disimpan: {path}")

# ── Ringkasan insight ────────────────────────────────────────
print("\n" + "=" * 60)
print("  RINGKASAN INSIGHT EDA")
print("=" * 60)
print(f"  {'Komoditas':15s} | {'CV%':>6s} | {'Efek Ramadan':>13s} | {'Outlier':>7s}")
print("  " + "-" * 50)
for comm in COMMODITIES:
    cv  = df[comm].std()/df[comm].mean()*100
    ram = (df[df['is_ramadan']==1][comm].mean() -
           df[df['is_ramadan']==0][comm].mean()) / \
           df[df['is_ramadan']==0][comm].mean() * 100
    daily = df[comm].pct_change()*100
    Q1,Q3 = daily.quantile(0.25), daily.quantile(0.75)
    IQR   = Q3-Q1
    n_out = len(daily[(daily<Q1-3*IQR)|(daily>Q3+3*IQR)])
    print(f"  {comm:15s} | {cv:5.1f}% | {ram:+12.1f}% | {n_out:7d}")

print("\n  ✅ EDA selesai! 6 grafik tersimpan di folder outputs/")
print("=" * 60)
