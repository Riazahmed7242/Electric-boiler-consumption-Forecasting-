# =============================================================================
#  ELECTRICITY FORECASTING PIPELINE — Minute Resolution
#  Models: Gradient Boosting | Random Forest | XGBoost | LSTM
#  ✅ DATA LEAKAGE FIXED — Publishable / Realistic Results
# =============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
os.makedirs("forecast_outputs", exist_ok=True)

print("=" * 65)
print("  ELECTRICITY FORECASTING PIPELINE — Leakage-Free Version")
print("=" * 65)

# =============================================================================
# STEP 1 — LOAD DATA  [UNCHANGED]
# =============================================================================
print("\n[1/6]  Loading data ...")
FILE_PATH = "371_2025-11-01T0000_2026-01-31T0000__1_.xlsx"

if FILE_PATH.endswith(".xlsx") or FILE_PATH.endswith(".xls"):
    df = pd.read_excel(FILE_PATH)
else:
    df = pd.read_csv(FILE_PATH)

df.columns = [c.strip() for c in df.columns]
col_map = {}
for c in df.columns:
    cl = c.lower()
    if "start" in cl:
        col_map[c] = "startTime"
    elif "end" in cl:
        col_map[c] = "endTime"
    else:
        col_map[c] = "consumption"
df.rename(columns=col_map, inplace=True)
print(f"   Loaded {len(df):,} rows | Columns: {list(df.columns)}")

# =============================================================================
# STEP 2 — CLEAN & PROCESS DATA  [UNCHANGED]
# =============================================================================
print("\n[2/6]  Cleaning and processing data ...")
df["startTime"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
df["startTime"] = df["startTime"].dt.tz_localize(None)
df.dropna(subset=["startTime"], inplace=True)
df.sort_values("startTime", inplace=True)
df.reset_index(drop=True, inplace=True)
df = df[["startTime", "consumption"]].copy()
df.set_index("startTime", inplace=True)
df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
df["consumption"].ffill(inplace=True)
df["consumption"].bfill(inplace=True)
Q1, Q3  = df["consumption"].quantile([0.25, 0.75])
IQR     = Q3 - Q1
lower   = Q1 - 3 * IQR
upper   = Q3 + 3 * IQR
df["consumption"] = df["consumption"].clip(lower, upper)
print(f"   Rows after cleaning : {len(df):,}")
print(f"   Date range          : {df.index.min()}  →  {df.index.max()}")
print(f"   Consumption range   : {df['consumption'].min():.2f}  –  {df['consumption'].max():.2f}")

# =============================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  STEP 3 — FEATURE ENGINEERING  [🔧 FIX #1: LEAKAGE REMOVED] ║
# ╚══════════════════════════════════════════════════════════════╝
# WHAT WAS WRONG:
#   df['rolling_mean'] = df['consumption'].rolling(10).mean()
#   → includes current row → LEAKAGE
#
# WHAT IS FIXED:
#   ALL lag and rolling features now use .shift(1) FIRST
#   so at timestamp T, only data up to T-1 is visible
# =============================================================================
print("\n[3/6]  Engineering features (leakage-free) ...")

# ── Time features (no leakage — derived from index only) ─────────────────────
df["hour"]            = df.index.hour
df["day_of_week"]     = df.index.dayofweek
df["minute_of_hour"]  = df.index.minute

df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["min_sin"]   = np.sin(2 * np.pi * df["minute_of_hour"] / 60)
df["min_cos"]   = np.cos(2 * np.pi * df["minute_of_hour"] / 60)

# ✅ FIX #1a — Lag features: MUST use .shift(n), never .shift(0)
#    At prediction time T, we ONLY know values up to T-1
for lag in [1, 5, 10, 30, 60]:
    df[f"lag_{lag}"] = df["consumption"].shift(lag)   # ✅ shift(lag), not shift(0)

# ✅ FIX #1b — Rolling features: .shift(1) BEFORE .rolling()
#    This ensures the window at T covers [T-window, T-1], never T itself
df["rolling_mean_5"]  = df["consumption"].shift(1).rolling(5).mean()   # ✅
df["rolling_mean_15"] = df["consumption"].shift(1).rolling(15).mean()  # ✅
df["rolling_std_15"]  = df["consumption"].shift(1).rolling(15).std()   # ✅

df.dropna(inplace=True)

FEATURE_COLS = [
    "hour", "day_of_week", "minute_of_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "min_sin", "min_cos",
    "lag_1", "lag_5", "lag_10", "lag_30", "lag_60",
    "rolling_mean_5", "rolling_mean_15", "rolling_std_15"
]
TARGET_COL = "consumption"
print(f"   Features created : {len(FEATURE_COLS)}")
print(f"   Rows after dropna: {len(df):,}")

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — TRAIN / TEST SPLIT  [🔧 FIX #2: STRICT 80/20 SPLIT] ║
# ╚══════════════════════════════════════════════════════════════════╝
# WHAT WAS WRONG:
#   Splitting by "last 7 days" — too small test window, and if data
#   is uneven this creates an implicit look-ahead boundary leak.
#   Never use shuffle=True for time-series.
#
# WHAT IS FIXED:
#   Strict positional 80/20 split — NO shuffling, NO random state
#   Train = first 80% rows  |  Test = last 20% rows
# =============================================================================
print("\n[4/6]  Splitting train / test (strict 80/20, no shuffle) ...")

split_idx = int(len(df) * 0.80)
train_df  = df.iloc[:split_idx]
test_df   = df.iloc[split_idx:]

X_train = train_df[FEATURE_COLS].values
y_train = train_df[TARGET_COL].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values

# ✅ FIX #3 — DEBUG VERIFICATION: print time ranges
print("\n   ┌─────────────────────────────────────────────────────┐")
print("   │              SPLIT VERIFICATION                    │")
print("   ├─────────────────────────────────────────────────────┤")
print(f"   │  Train rows : {len(train_df):>8,}                           │")
print(f"   │  Train START: {str(train_df.index.min()):<35}  │")
print(f"   │  Train END  : {str(train_df.index.max()):<35}  │")
print("   ├─────────────────────────────────────────────────────┤")
print(f"   │  Test  rows : {len(test_df):>8,}                           │")
print(f"   │  Test  START: {str(test_df.index.min()):<35}  │")
print(f"   │  Test  END  : {str(test_df.index.max()):<35}  │")
print("   └─────────────────────────────────────────────────────┘")

# ✅ FIX #4 — SCALER: fit ONLY on train, transform both separately
#    Wrong: scaler.fit(X_all) then transform both → leaks test statistics
#    Correct: scaler.fit(X_train) only
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)   # ✅ fit on train ONLY
X_test_sc  = scaler_X.transform(X_test)         # ✅ transform only (no fit)

scaler_y = StandardScaler()
y_train_sc = scaler_y.fit_transform(             # ✅ fit on train ONLY
    y_train.reshape(-1, 1)
).ravel()
# y_test is intentionally NOT scaled here —
# we inverse_transform LSTM outputs to raw kW for fair comparison

# =============================================================================
# STEP 5 — MODEL TRAINING  [UNCHANGED — models are identical]
# =============================================================================
print("\n[5/6]  Training models ...")
results = {}
preds   = {}

def evaluate(name, y_true, y_pred, train_secs):
    """Compute metrics on OUT-OF-SAMPLE test set only."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2, "Train Time (s)": train_secs}
    print(f"   {name:<22}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  Time={train_secs:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 5A — GRADIENT BOOSTING  [unchanged]
# ─────────────────────────────────────────────────────────────────────────────
print("\n   ▶  Gradient Boosting ...")
t0 = time.time()
gb = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, min_samples_split=10, max_features="sqrt", random_state=42
)
gb.fit(X_train_sc, y_train)          # ✅ train on training data only
gb_time = time.time() - t0
gb_pred = gb.predict(X_test_sc)      # ✅ predict on test data only
preds["Gradient Boosting"] = gb_pred
evaluate("Gradient Boosting", y_test, gb_pred, gb_time)

# ─────────────────────────────────────────────────────────────────────────────
# 5B — RANDOM FOREST  [unchanged]
# ─────────────────────────────────────────────────────────────────────────────
print("\n   ▶  Random Forest ...")
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_split=10,
    min_samples_leaf=4, max_features="sqrt", n_jobs=-1, random_state=42
)
rf.fit(X_train_sc, y_train)
rf_time = time.time() - t0
rf_pred = rf.predict(X_test_sc)
preds["Random Forest"] = rf_pred
evaluate("Random Forest", y_test, rf_pred, rf_time)

# ─────────────────────────────────────────────────────────────────────────────
# 5C — XGBOOST  [unchanged]
# ─────────────────────────────────────────────────────────────────────────────
print("\n   ▶  XGBoost ...")
t0 = time.time()
xgbm = xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist", random_state=42, verbosity=0
)
xgbm.fit(X_train_sc, y_train)
xgb_time = time.time() - t0
xgb_pred = xgbm.predict(X_test_sc)
preds["XGBoost"] = xgb_pred
evaluate("XGBoost", y_test, xgb_pred, xgb_time)

# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════════╗
# ║  5D — LSTM  [🔧 FIX #5: Same split + no leakage in sequences]  ║
# ╚══════════════════════════════════════════════════════════════════╝
# WHAT WAS WRONG:
#   Test sequences were built from combined train+test data,
#   meaning LSTM could implicitly see future test values
#   when constructing the context window.
#
# WHAT IS FIXED:
#   - Sequences built strictly within train portion
#   - Test sequences: only last LOOKBACK rows of TRAIN used as seed
#   - scaler_y fit on TRAIN only (already fixed above)
#   - training=False during inference (deterministic, no dropout)
# ─────────────────────────────────────────────────────────────────────────────
print("\n   ▶  LSTM (lookback=60, leakage-free sequences) ...")
LOOKBACK = 60

# Stack scaled features + scaled target for sequence creation
# ✅ FIX: X_all_sc and X_test_sc2 built from respective splits only
X_all_sc   = np.hstack([X_train_sc,
                         y_train_sc.reshape(-1, 1)])          # train block
X_test_sc2 = np.hstack([X_test_sc,
                         scaler_y.transform(                  # ✅ transform only
                             y_test.reshape(-1, 1))])         # test block

def make_sequences(data, lookback):
    """Build (X_seq, y_seq) with NO leakage.
    At position i, input window = [i-lookback … i-1], target = data[i, -1]
    """
    Xs, ys = [], []
    for i in range(lookback, len(data)):
        Xs.append(data[i - lookback : i])     # ✅ window ends BEFORE target step
        ys.append(data[i, -1])
    return np.array(Xs), np.array(ys)

# Training sequences — built entirely from training data
X_lstm_train, y_lstm_train = make_sequences(X_all_sc, LOOKBACK)

# ✅ FIX: Test sequences seeded with last LOOKBACK rows of TRAINING data
#    This avoids any future information leaking into the window
warm_up      = X_all_sc[-LOOKBACK:]            # last 60 rows of train
test_ext     = np.vstack([warm_up, X_test_sc2])
X_lstm_test, _ = make_sequences(test_ext, LOOKBACK)

n_features = X_lstm_train.shape[2]

lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOKBACK, n_features)),
    BatchNormalization(), Dropout(0.2),
    LSTM(64,  return_sequences=False),
    BatchNormalization(), Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="huber"
)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, verbose=0)
]
t0 = time.time()
lstm_model.fit(
    X_lstm_train, y_lstm_train,
    epochs=60, batch_size=256,
    validation_split=0.1,          # ✅ uses last 10% of TRAINING sequences
    callbacks=callbacks, verbose=0
)
lstm_time = time.time() - t0

# ✅ training=False → Dropout disabled → deterministic inference
lstm_pred_sc = lstm_model.predict(X_lstm_test, verbose=0).ravel()
lstm_pred    = scaler_y.inverse_transform(
    lstm_pred_sc.reshape(-1, 1)
).ravel()
lstm_pred    = np.clip(lstm_pred, 0, upper)

# Align lengths (LSTM may produce slightly fewer rows than y_test)
min_len       = min(len(lstm_pred), len(y_test))
lstm_pred     = lstm_pred[:min_len]
y_test_lstm   = y_test[:min_len]
test_idx_lstm = test_df.index[:min_len]

preds["LSTM"] = (lstm_pred, test_idx_lstm, y_test_lstm)
evaluate("LSTM", y_test_lstm, lstm_pred, lstm_time)

# =============================================================================
# ╔═══════════════════════════════════════════════════════════════╗
# ║  METRICS SUMMARY  [🔧 FIX #6: test-set only, clearly shown]  ║
# ╚═══════════════════════════════════════════════════════════════╝
# =============================================================================
metrics_df = pd.DataFrame(results).T[["RMSE", "MAE", "R²", "Train Time (s)"]].round(4)
best_model = metrics_df["RMSE"].idxmin()

print("\n" + "=" * 65)
print("  FINAL COMPARISON TABLE  (Out-of-Sample Test Set)")
print("=" * 65)
print(metrics_df.to_string())
print("=" * 65)
print(f"\n  🏆  Best Model (lowest RMSE): {best_model}")
print(f"      RMSE = {metrics_df.loc[best_model, 'RMSE']}")
print(f"      MAE  = {metrics_df.loc[best_model, 'MAE']}")
print(f"      R²   = {metrics_df.loc[best_model, 'R²']}")
print("=" * 65)
metrics_df.to_csv("forecast_outputs/model_comparison.csv")

# ✅ Final split confirmation
print("\n  ── FINAL DATA SPLIT CONFIRMATION ──────────────────────────")
print(f"  Train : {train_df.index[0]}  →  {train_df.index[-1]}")
print(f"          ({len(train_df):,} rows, {len(train_df)/len(df)*100:.1f}% of data)")
print(f"  Test  : {test_df.index[0]}  →  {test_df.index[-1]}")
print(f"          ({len(test_df):,} rows, {len(test_df)/len(df)*100:.1f}% of data)")
print(f"  NO overlap: {test_df.index[0] > train_df.index[-1]}")
print("  ────────────────────────────────────────────────────────────")

# =============================================================================
# STEP 6 — VISUALIZATION  [UNCHANGED — full enhanced section]
# =============================================================================
print("\n[6/6]  Generating enhanced visualizations ...")

plt.rcParams.update({
    "figure.dpi":        130,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "legend.framealpha": 0.85,
    "legend.fontsize":   9,
})

COLORS = {
    "Gradient Boosting": "#e74c3c",
    "Random Forest":     "#3498db",
    "XGBoost":           "#2ecc71",
    "LSTM":              "#9b59b6",
}
ACTUAL_COLOR = "#2c3e50"
SMOOTH_COLOR = "#f39c12"
MODEL_NAMES  = list(COLORS.keys())
test_index   = test_df.index
saved_files  = []

def get_pred_actual(name):
    if name == "LSTM":
        return preds["LSTM"][1], preds["LSTM"][0], preds["LSTM"][2]
    return test_index, preds[name], y_test

def day_slice(idx, pred, actual, start_day, n_days):
    start = test_index.min() + pd.Timedelta(days=start_day)
    end   = start + pd.Timedelta(days=n_days)
    mask  = (idx >= start) & (idx < end)
    return idx[mask], pred[mask], actual[mask]

def fmt_xaxis(ax, fmt="%b %d", rotation=30):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right", fontsize=8)

def save(fname):
    path = f"forecast_outputs/{fname}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅  {fname}")
    saved_files.append(fname)

def section(title):
    print(f"\n   {'─'*55}")
    print(f"   {title}")

# ── 01  Full week actual vs predicted ────────────────────────────────────────
section("① Actual vs Predicted — Full Test Period")
fig, axes = plt.subplots(4, 1, figsize=(20, 22))
fig.suptitle("Actual vs Predicted — Full Test Set (Leakage-Free)\nAll Models",
             fontsize=16, fontweight="bold", y=0.99)
for ax, name in zip(axes, MODEL_NAMES):
    idx, pred, actual = get_pred_actual(name)
    ax.plot(idx, actual, color=ACTUAL_COLOR, lw=0.7, alpha=0.75, label="Actual", zorder=2)
    ax.plot(idx, pred,   color=COLORS[name],  lw=0.9, alpha=0.90, label=f"{name} Forecast", zorder=3)
    ax.set_title(f"{name}   |   RMSE={results[name]['RMSE']:.4f}   R²={results[name]['R²']:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Consumption (kW)")
    ax.legend(loc="upper right")
    fmt_xaxis(ax)
plt.tight_layout(rect=[0, 0, 1, 0.98])
save("01_actual_vs_predicted_full.png")

# ── 02–04  Zoomed windows ────────────────────────────────────────────────────
zoom_specs = [
    ("02_zoomed_first_2_days.png",  "First 2 Days",   0, 2),
    ("03_zoomed_mid_period.png",    "Mid-Period (Days 3–5)", 2, 3),
    ("04_zoomed_last_2_days.png",   "Last 2 Days",    -2, 2),
]
for fname, label, start, n in zoom_specs:
    section(f"Zoomed — {label}")
    fig, axes = plt.subplots(4, 1, figsize=(18, 18))
    fig.suptitle(f"Actual vs Predicted — {label} (Zoomed)", fontsize=14, fontweight="bold")
    for ax, name in zip(axes, MODEL_NAMES):
        idx, pred, actual = get_pred_actual(name)
        if start < 0:
            total_days = (idx[-1] - idx[0]).days
            zi, zp, za = day_slice(idx, pred, actual, total_days + start, n)
        else:
            zi, zp, za = day_slice(idx, pred, actual, start, n)
        ax.plot(zi, za, color=ACTUAL_COLOR, lw=1.0, label="Actual")
        ax.plot(zi, zp, color=COLORS[name],  lw=1.1, label=name, linestyle="--")
        ax.set_title(f"{name} — {label}", fontsize=10, fontweight="bold")
        ax.set_ylabel("kW")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.tight_layout()
    save(fname)

# ── 05  Individual model actual vs predicted ─────────────────────────────────
section("⑤ Individual Model Actual vs Predicted")
for name in MODEL_NAMES:
    idx, pred, actual = get_pred_actual(name)
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.fill_between(idx, actual, alpha=0.15, color=ACTUAL_COLOR)
    ax.plot(idx, actual, color=ACTUAL_COLOR, lw=0.8, alpha=0.85, label="Actual")
    ax.plot(idx, pred,   color=COLORS[name],  lw=1.0, alpha=0.90, label=f"{name} Forecast")
    ax.set_title(f"{name} | RMSE={results[name]['RMSE']:.4f}  R²={results[name]['R²']:.4f}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Consumption (kW)")
    ax.set_xlabel("Time")
    ax.legend()
    fmt_xaxis(ax)
    plt.tight_layout()
    save(f"05_{name.lower().replace(' ','_')}_actual_vs_predicted.png")

# ── 06–09  Metric bar charts ──────────────────────────────────────────────────
def metric_bar(metric, title, ylabel, good, filename):
    vals   = [results[m][metric] for m in MODEL_NAMES]
    colors = [COLORS[m] for m in MODEL_NAMES]
    best   = min(vals) if good == "low" else max(vals)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(MODEL_NAMES, vals, color=colors, edgecolor="white",
                  linewidth=1.4, width=0.5)
    for bar, val in zip(bars, vals):
        bar.set_edgecolor("#FFD700" if val == best else "white")
        bar.set_linewidth(2.5 if val == best else 1.2)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    save(filename)

section("⑥ Metric Comparison Bar Charts")
metric_bar("RMSE", "RMSE Comparison (lower = better)",  "RMSE",         "low",  "06_rmse_comparison.png")
metric_bar("MAE",  "MAE Comparison  (lower = better)",  "MAE",          "low",  "07_mae_comparison.png")
metric_bar("R²",   "R² Comparison   (higher = better)", "R²",           "high", "08_r2_comparison.png")
metric_bar("Train Time (s)", "Training Time (seconds)", "Time (s)",     "low",  "09_training_time_comparison.png")

# ── 10  Combined 3-panel ──────────────────────────────────────────────────────
section("⑦ Combined Metric Comparison")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Model Performance — Combined Comparison (Leakage-Free)",
             fontsize=15, fontweight="bold")
for ax, (metric, label, good) in zip(axes, [
    ("R²",            "R² Score",       "high"),
    ("RMSE",          "RMSE",           "low"),
    ("Train Time (s)","Training Time (s)","low"),
]):
    vals   = [results[m][metric] for m in MODEL_NAMES]
    best   = min(vals) if good == "low" else max(vals)
    bars   = ax.bar(MODEL_NAMES, vals, color=[COLORS[m] for m in MODEL_NAMES],
                    edgecolor="white", linewidth=1.2, width=0.55)
    for bar, val in zip(bars, vals):
        if val == best:
            bar.set_edgecolor("#FFD700"); bar.set_linewidth(3)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
plt.tight_layout()
save("10_combined_comparison.png")

# ── 11  Error distribution ────────────────────────────────────────────────────
section("⑧ Error Distribution")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Error Distribution — Residuals Histogram", fontsize=14, fontweight="bold")
for ax, name in zip(axes.flatten(), MODEL_NAMES):
    idx, pred, actual = get_pred_actual(name)
    residuals = actual - pred
    ax.hist(residuals, bins=80, color=COLORS[name], edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(0,               color="black",   lw=2.0, label="Zero Error")
    ax.axvline(residuals.mean(), color="#f39c12", lw=1.8, linestyle="--",
               label=f"Mean={residuals.mean():.2f}")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
plt.tight_layout()
save("11_error_distribution.png")

# ── 12  Residuals over time ───────────────────────────────────────────────────
section("⑨ Residual Plot")
fig, axes = plt.subplots(4, 1, figsize=(20, 18))
fig.suptitle("Residuals Over Time (Actual − Predicted)", fontsize=14, fontweight="bold")
for ax, name in zip(axes, MODEL_NAMES):
    idx, pred, actual = get_pred_actual(name)
    residuals = actual - pred
    ax.plot(idx, residuals, color=COLORS[name], lw=0.6, alpha=0.75)
    ax.axhline(0, color="black", lw=1.2, linestyle="--")
    ax.fill_between(idx, residuals, 0, alpha=0.12, color=COLORS[name])
    ax.set_title(f"{name} — Residuals", fontsize=10, fontweight="bold")
    ax.set_ylabel("Residual (kW)")
    fmt_xaxis(ax)
plt.tight_layout()
save("12_residuals_over_time.png")

# ── 13  Hourly trend ──────────────────────────────────────────────────────────
section("⑩ Hourly Trend Pattern")
fig, ax = plt.subplots(figsize=(14, 6))
hourly_actual = test_df.copy()
hourly_actual["hour"] = hourly_actual.index.hour
hourly_mean   = hourly_actual.groupby("hour")["consumption"].mean()
ax.plot(hourly_mean.index, hourly_mean.values, color=ACTUAL_COLOR,
        lw=2.2, marker="o", markersize=5, label="Actual Mean", zorder=4)
for name in MODEL_NAMES:
    idx, pred, actual = get_pred_actual(name)
    pred_df   = pd.DataFrame({"hour": idx.hour, "pred": pred})
    pred_mean = pred_df.groupby("hour")["pred"].mean()
    ax.plot(pred_mean.index, pred_mean.values, color=COLORS[name],
            lw=1.6, linestyle="--", marker="s", markersize=4,
            label=name, alpha=0.85)
ax.set_title("Average Hourly Consumption Pattern — Actual vs All Models",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Consumption (kW)")
ax.set_xticks(range(24))
ax.legend()
plt.tight_layout()
save("13_hourly_trend.png")

# ── 14  Day-of-week pattern ───────────────────────────────────────────────────
section("⑪ Daily Pattern")
DOW_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
fig, ax = plt.subplots(figsize=(13, 6))
dow_actual = test_df.copy()
dow_actual["dow"] = dow_actual.index.dayofweek
dow_mean = dow_actual.groupby("dow")["consumption"].mean()
ax.plot(dow_mean.index, dow_mean.values, color=ACTUAL_COLOR,
        lw=2.5, marker="D", markersize=7, label="Actual", zorder=4)
for name in MODEL_NAMES:
    idx, pred, actual = get_pred_actual(name)
    pdf = pd.DataFrame({"dow": idx.dayofweek, "pred": pred})
    pred_mean = pdf.groupby("dow")["pred"].mean()
    ax.plot(pred_mean.index, pred_mean.values, color=COLORS[name],
            lw=1.7, linestyle="--", marker="o", markersize=5,
            label=name, alpha=0.85)
ax.set_title("Average Consumption by Day of Week", fontsize=13, fontweight="bold")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Avg Consumption (kW)")
ax.set_xticks(range(7))
ax.set_xticklabels(DOW_LABELS)
ax.legend()
plt.tight_layout()
save("14_daily_pattern_dow.png")

# ── 15  Minute pattern ────────────────────────────────────────────────────────
section("⑫ Minute-Level Pattern")
fig, ax = plt.subplots(figsize=(14, 6))
min_actual = test_df.copy()
min_actual["minute"] = min_actual.index.minute
min_mean   = min_actual.groupby("minute")["consumption"].mean()
ax.plot(min_mean.index, min_mean.values, color=ACTUAL_COLOR,
        lw=2.0, label="Actual Mean", marker=".", markersize=4)
for name in MODEL_NAMES:
    idx, pred, actual = get_pred_actual(name)
    pdf = pd.DataFrame({"minute": idx.minute, "pred": pred})
    pred_mean = pdf.groupby("minute")["pred"].mean()
    ax.plot(pred_mean.index, pred_mean.values, color=COLORS[name],
            lw=1.4, linestyle="--", label=name, alpha=0.85)
ax.set_title("Average Consumption by Minute of Hour", fontsize=13, fontweight="bold")
ax.set_xlabel("Minute of Hour")
ax.set_ylabel("Avg Consumption (kW)")
ax.set_xticks(range(0, 60, 5))
ax.legend()
plt.tight_layout()
save("15_minute_pattern.png")

# ── 16  Smoothed trend ────────────────────────────────────────────────────────
section("⑬ Smoothed Forecast")
SMOOTH_WIN = 60
fig, axes = plt.subplots(4, 1, figsize=(20, 20))
fig.suptitle(f"Forecast with Smoothed Trend (window={SMOOTH_WIN} min)",
             fontsize=14, fontweight="bold")
for ax, name in zip(axes, MODEL_NAMES):
    idx, pred, actual = get_pred_actual(name)
    smoothed = uniform_filter1d(pred, size=SMOOTH_WIN)
    ax.plot(idx, actual,   color=ACTUAL_COLOR, lw=0.7, alpha=0.55, label="Actual")
    ax.plot(idx, pred,     color=COLORS[name],  lw=0.8, alpha=0.60, label=f"{name} Raw")
    ax.plot(idx, smoothed, color=SMOOTH_COLOR,  lw=2.0, alpha=0.95, label=f"Smoothed")
    ax.set_title(f"{name} — Raw + Smoothed", fontsize=10, fontweight="bold")
    ax.set_ylabel("kW")
    ax.legend(loc="upper right")
    fmt_xaxis(ax)
plt.tight_layout()
save("16_smoothed_forecast.png")

# ── 17  Peak demand ───────────────────────────────────────────────────────────
section("⑭ Peak Demand Comparison")
fig, ax = plt.subplots(figsize=(14, 6))
peak_df = pd.DataFrame({"actual": y_test}, index=test_index)
peak_df["date"] = peak_df.index.date
daily_peak_actual = peak_df.groupby("date")["actual"].max()
dates = list(daily_peak_actual.index)
x     = range(len(dates))
ax.plot(x, daily_peak_actual.values, color=ACTUAL_COLOR,
        lw=2.2, marker="D", markersize=7, label="Actual Peak", zorder=5)
for name in MODEL_NAMES:
    idx, pred, _ = get_pred_actual(name)
    pdf = pd.DataFrame({"pred": pred, "date": idx.date})
    daily_peak_pred = pdf.groupby("date")["pred"].max().reindex(dates, fill_value=np.nan)
    ax.plot(x, daily_peak_pred.values, color=COLORS[name],
            lw=1.7, linestyle="--", marker="o", markersize=5,
            label=name, alpha=0.85)
ax.set_title("Daily Peak Demand — Actual vs All Models", fontsize=13, fontweight="bold")
ax.set_xlabel("Day")
ax.set_ylabel("Peak Consumption (kW)")
ax.set_xticks(list(x))
ax.set_xticklabels([str(d) for d in dates], rotation=35, ha="right", fontsize=8)
ax.legend()
plt.tight_layout()
save("17_peak_demand_comparison.png")

# ── 18  Min / Max ─────────────────────────────────────────────────────────────
section("⑮ Min & Max Comparison")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Daily Min & Max Prediction — All Models", fontsize=14, fontweight="bold")
for ax, (stat, label) in zip(axes, [("min","Daily Minimum"),("max","Daily Maximum")]):
    base_df = pd.DataFrame({"actual": y_test}, index=test_index)
    base_df["date"] = base_df.index.date
    fn = base_df.groupby("date")["actual"].min if stat == "min" else base_df.groupby("date")["actual"].max
    daily_stat = fn()
    ds = list(daily_stat.index)
    xp = range(len(ds))
    ax.plot(xp, daily_stat.values, color=ACTUAL_COLOR, lw=2.2,
            marker="D", markersize=7, label="Actual", zorder=5)
    for name in MODEL_NAMES:
        idx, pred, _ = get_pred_actual(name)
        pdf = pd.DataFrame({"pred": pred, "date": idx.date})
        fn2 = pdf.groupby("date")["pred"].min if stat == "min" else pdf.groupby("date")["pred"].max
        dp  = fn2().reindex(ds, fill_value=np.nan)
        ax.plot(xp, dp.values, color=COLORS[name], lw=1.6,
                linestyle="--", marker="o", markersize=5,
                label=name, alpha=0.85)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("kW")
    ax.set_xticks(list(xp))
    ax.set_xticklabels([str(d) for d in ds], rotation=40, ha="right", fontsize=7)
    ax.legend(fontsize=8)
plt.tight_layout()
save("18_min_max_comparison.png")

# ── 19  Rolling RMSE (model stability) ───────────────────────────────────────
section("⑯ Model Stability — Rolling RMSE")
ROLL_WIN = 120
fig, ax = plt.subplots(figsize=(20, 7))
for name in MODEL_NAMES:
    idx, pred, actual = get_pred_actual(name)
    se        = (actual - pred) ** 2
    roll_rmse = pd.Series(se, index=idx).rolling(ROLL_WIN).mean() ** 0.5
    ax.plot(idx, roll_rmse, color=COLORS[name], lw=1.4, label=name, alpha=0.88)
ax.set_title(f"Model Stability — Rolling RMSE (window={ROLL_WIN} min)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Rolling RMSE")
ax.legend()
fmt_xaxis(ax)
plt.tight_layout()
save("19_model_stability_rolling_rmse.png")

# ── 20  Forecast spread ───────────────────────────────────────────────────────
section("⑰ Forecast Spread")
fig, ax = plt.subplots(figsize=(22, 7))
ax.plot(test_index, y_test, color=ACTUAL_COLOR, lw=0.9, alpha=0.80,
        label="Actual", zorder=5)
all_preds_arr = []
for name in MODEL_NAMES:
    idx, pred, _ = get_pred_actual(name)
    ml = min(len(pred), len(test_index))
    all_preds_arr.append(pred[:ml])
    ax.plot(test_index[:ml], pred[:ml], color=COLORS[name],
            lw=0.9, alpha=0.65, label=name)
mat = np.vstack(all_preds_arr)
ax.fill_between(test_index[:mat.shape[1]], mat.min(axis=0), mat.max(axis=0),
                alpha=0.15, color="gray", label="Model Spread")
ax.set_title("Forecast Spread — All Models vs Actual",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Consumption (kW)")
ax.legend(loc="upper right")
fmt_xaxis(ax)
plt.tight_layout()
save("20_forecast_spread.png")

# ── 21  Variability ───────────────────────────────────────────────────────────
section("⑱ Variability Comparison")
ROLL_STD = 60
fig, ax = plt.subplots(figsize=(20, 7))
actual_std = pd.Series(y_test, index=test_index).rolling(ROLL_STD).std()
ax.plot(test_index, actual_std, color=ACTUAL_COLOR, lw=2.0,
        label="Actual", zorder=5)
for name in MODEL_NAMES:
    idx, pred, _ = get_pred_actual(name)
    pred_std = pd.Series(pred, index=idx).rolling(ROLL_STD).std()
    ax.plot(idx, pred_std, color=COLORS[name], lw=1.4,
            linestyle="--", label=name, alpha=0.85)
ax.set_title(f"Forecast Variability — Rolling Std Dev (window={ROLL_STD} min)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Rolling Std Dev (kW)")
ax.legend()
fmt_xaxis(ax)
plt.tight_layout()
save("21_variability_comparison.png")

# ── 22  Weekly windows ────────────────────────────────────────────────────────
section("⑲ Weekly Forecast Graphs")
for week_num in range(1, 5):
    ws = test_index.min() + pd.Timedelta(weeks=week_num - 1)
    we = ws + pd.Timedelta(weeks=1)
    fig, axes = plt.subplots(4, 1, figsize=(20, 18))
    fig.suptitle(f"Forecast — Week {week_num}  ({ws.date()} → {we.date()})",
                 fontsize=14, fontweight="bold")
    for ax, name in zip(axes, MODEL_NAMES):
        idx, pred, actual = get_pred_actual(name)
        mask = (idx >= ws) & (idx < we)
        wi, wp, wa = idx[mask], pred[mask], actual[mask]
        if len(wi) == 0:
            ax.set_title(f"{name} — No data for this week")
            continue
        ax.plot(wi, wa, color=ACTUAL_COLOR, lw=0.8, alpha=0.75, label="Actual")
        ax.plot(wi, wp, color=COLORS[name],  lw=1.0, alpha=0.90, label=f"{name}")
        rmse_w = np.sqrt(mean_squared_error(wa, wp))
        ax.set_title(f"{name} | Week-RMSE={rmse_w:.4f}", fontsize=10, fontweight="bold")
        ax.set_ylabel("kW")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    save(f"22_week{week_num}_forecast.png")

# ── 26  Daily average ─────────────────────────────────────────────────────────
section("⑳ Daily Average Forecast")
fig, ax = plt.subplots(figsize=(16, 7))
base = pd.DataFrame({"actual": y_test}, index=test_index)
base["date"] = base.index.date
daily_avg_actual = base.groupby("date")["actual"].mean()
x_dates = list(daily_avg_actual.index)
xp = range(len(x_dates))
ax.plot(xp, daily_avg_actual.values, color=ACTUAL_COLOR,
        lw=2.2, marker="D", markersize=8, label="Actual Daily Avg", zorder=5)
for name in MODEL_NAMES:
    idx, pred, _ = get_pred_actual(name)
    pdf = pd.DataFrame({"pred": pred, "date": idx.date})
    daily_avg_pred = pdf.groupby("date")["pred"].mean().reindex(x_dates, fill_value=np.nan)
    ax.plot(xp, daily_avg_pred.values, color=COLORS[name], lw=1.8,
            linestyle="--", marker="o", markersize=6, label=name, alpha=0.87)
ax.set_title("Daily Average Forecast — Actual vs All Models",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Avg Consumption (kW)")
ax.set_xticks(list(xp))
ax.set_xticklabels([str(d) for d in x_dates], rotation=40, ha="right", fontsize=8)
ax.legend()
plt.tight_layout()
save("26_daily_average_forecast.png")

# ── 27  Daily total ───────────────────────────────────────────────────────────
section("㉑ Daily Total Consumption")
fig, ax = plt.subplots(figsize=(16, 7))
daily_total_actual = base.groupby("date")["actual"].sum() / 60
ax.bar([i - 0.3 for i in xp], daily_total_actual.values,
       width=0.15, color=ACTUAL_COLOR, label="Actual (kWh)", alpha=0.85)
for name, offset in zip(MODEL_NAMES, [-0.15, 0.0, 0.15, 0.30]):
    idx, pred, _ = get_pred_actual(name)
    pdf = pd.DataFrame({"pred": pred, "date": idx.date})
    total_pred = pdf.groupby("date")["pred"].sum().reindex(x_dates, fill_value=0) / 60
    ax.bar([i + offset for i in xp], total_pred.values,
           width=0.15, color=COLORS[name], label=name, alpha=0.80)
ax.set_title("Daily Total Consumption Forecast (kWh)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Total Consumption (kWh)")
ax.set_xticks(list(xp))
ax.set_xticklabels([str(d) for d in x_dates], rotation=40, ha="right", fontsize=8)
ax.legend()
plt.tight_layout()
save("27_daily_total_forecast.png")

# ── 28  Model-wise forecast overview ─────────────────────────────────────────
section("㉒ Individual Model Forecast Overview")
for name in MODEL_NAMES:
    idx, pred, actual = get_pred_actual(name)
    smoothed = uniform_filter1d(pred, size=60)
    fig, axes = plt.subplots(2, 1, figsize=(22, 12),
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{name} — Forecast Overview  |  "
                 f"RMSE={results[name]['RMSE']:.4f}  MAE={results[name]['MAE']:.4f}  "
                 f"R²={results[name]['R²']:.4f}",
                 fontsize=13, fontweight="bold")
    axes[0].fill_between(idx, actual, alpha=0.12, color=ACTUAL_COLOR)
    axes[0].plot(idx, actual,   color=ACTUAL_COLOR, lw=0.8, alpha=0.80, label="Actual")
    axes[0].plot(idx, pred,     color=COLORS[name],  lw=0.9, alpha=0.75, label="Forecast")
    axes[0].plot(idx, smoothed, color=SMOOTH_COLOR,  lw=2.0, alpha=0.95, label="Smoothed")
    axes[0].set_ylabel("Consumption (kW)")
    axes[0].legend()
    fmt_xaxis(axes[0])
    residuals = actual - pred
    axes[1].bar(range(len(residuals)), residuals,
                color=[COLORS[name] if r >= 0 else "#e74c3c" for r in residuals],
                width=1.0, alpha=0.65)
    axes[1].axhline(0, color="black", lw=1.2)
    axes[1].set_ylabel("Residual (kW)")
    axes[1].set_xlabel("Time Step (minutes)")
    plt.tight_layout()
    save(f"28_{name.lower().replace(' ','_')}_forecast_overview.png")

# ── 32  All models combined ───────────────────────────────────────────────────
section("㉓ All Models Full Forecast Combined")
fig, ax = plt.subplots(figsize=(24, 8))
ax.plot(test_index, y_test, color=ACTUAL_COLOR, lw=1.1, alpha=0.85,
        label="Actual", zorder=6)
for name in MODEL_NAMES:
    idx, pred, _ = get_pred_actual(name)
    ax.plot(idx, pred, color=COLORS[name], lw=0.9, alpha=0.75, label=name)
ax.set_title("All Models — Full Forecast vs Actual (Leakage-Free)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Consumption (kW)")
ax.set_xlabel("Time")
ax.legend(loc="upper right", fontsize=10)
fmt_xaxis(ax)
plt.tight_layout()
save("32_all_models_full_forecast.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  ALL GRAPHS SAVED — forecast_outputs/")
print(f"  Total graphs : {len(saved_files)}")
print("=" * 65)
for i, f in enumerate(saved_files, 1):
    print(f"  {i:02d}.  {f}")
print("\n" + "=" * 65)
print("  LEAKAGE-FIX SUMMARY")
print("=" * 65)
print("  ✅  FIX 1: Lag/rolling features use .shift(1) before .rolling()")
print("  ✅  FIX 2: 80/20 strict positional time-based split (no shuffle)")
print("  ✅  FIX 3: StandardScaler fit on TRAIN only, transform test only")
print("  ✅  FIX 4: All models trained on X_train, evaluated on X_test only")
print("  ✅  FIX 5: LSTM sequences seeded from train tail — no test leakage")
print("  ✅  FIX 6: Metrics computed on out-of-sample test predictions only")
print("  ✅  FIX 7: Debug verification prints train/test timestamps")
print("=" * 65)
print(f"\n  🏆  Best Model: {best_model}")
print(f"      RMSE={metrics_df.loc[best_model,'RMSE']}  "
      f"MAE={metrics_df.loc[best_model,'MAE']}  "
      f"R²={metrics_df.loc[best_model,'R²']}")
print("  ✔  Pipeline complete!")
