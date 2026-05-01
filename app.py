"""
Planetary Regression — NATAL ASPECTS Edition
=============================================
Y  : Log(price level)
X  : constant + time trend + transit-to-natal aspect dummies (OLS)
     + ANN on detrended residuals

Models
  1. OLS  : log_price ~ const + trend + aspect_dummies
  2. ANN  : (trend-residuals) ~ aspect_dummies  → combined with trend

Ephemeris is loaded from GitHub (set EPHEMERIS_URL below).
"""

import warnings
import datetime
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import statsmodels.api as sm
import streamlit as st

from sklearn.neural_network  import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  EPHEMERIS URL  ← set to your raw GitHub CSV
# ══════════════════════════════════════════════════════════════════════════════
EPHEMERIS_URL = (
    "https://raw.githubusercontent.com/goncuahm/kozmiknatal/"
    "main/planet_degrees.csv"
)

# ── Colours ───────────────────────────────────────────────────────────────────
BG     = "#080818"
GOLD   = "#C8A84B"
TEAL   = "#00D4B4"
WHITE  = "#E8E8F4"
GREY   = "#2A2A4A"
ORANGE = "#FF8844"
PURPLE = "#CC44FF"

# ── Planet universe ───────────────────────────────────────────────────────────
ALL_PLANETS = [
    "sun", "moon", "mercury", "venus", "mars",
    "jupiter", "saturn", "uranus", "neptune",
    "pluto", "true_node", "chiron",
]
ASPECTS   = [0, 60, 90, 120, 180]
ASP_NAMES = {0: "Conj", 60: "Sext", 90: "Sqr", 120: "Trin", 180: "Opp"}
SIGNS     = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# ── ANN architecture ─────────────────────────────────────────────────────────
ANN_LAYERS = (512, 32)
TRAIN_FRAC = 0.85

# ── Defaults for Silver (SI=F) ────────────────────────────────────────────────
DEFAULT_TICKER      = "SI=F"
DEFAULT_NAME        = "Silver Futures"
DEFAULT_NATAL_DATE  = "1933-07-05"
DEFAULT_STOCK_START = "1987-01-01"
DEFAULT_ORB_APPLY   = 4.0
DEFAULT_ORB_SEP     = 1.0
# Default ASC: Leo 29° = 4*30+29 = 149°
DEFAULT_ASC_DEG     = float(4 * 30 + 29)   # 149.0
DEFAULT_MC_DEG      = None                  # not used for silver


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Natal Planetary Regression",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Rajdhani:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #080818;
        color: #E8E8F4;
    }
    .stApp { background-color: #080818; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0d0d2e 0%,#080818 100%);
        border-right: 1px solid #2A2A4A;
    }
    h1, h2, h3 { font-family: 'Cinzel', serif; color: #C8A84B; }
    .stButton > button {
        background: linear-gradient(135deg,#C8A84B,#a07830);
        color: #080818; font-family:'Cinzel',serif;
        font-weight:700; border:none; border-radius:4px;
        padding:10px 28px; font-size:14px; letter-spacing:1px; width:100%;
    }
    .stButton > button:hover { opacity:.88; }
    .metric-card {
        background:#0d0d28; border:1px solid #2A2A4A;
        border-radius:6px; padding:14px 18px; margin:4px 0;
    }
    .metric-label { color:#888; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
    .metric-value { color:#C8A84B; font-size:22px; font-family:'Cinzel',serif; font-weight:600; }
    .metric-sub   { color:#aaa; font-size:11px; }
    .forecast-box {
        background:#0d0d28; border:1px solid #C8A84B40;
        border-radius:6px; padding:14px 18px; margin:6px 0;
    }
    .up-signal   { color:#44DD88; font-weight:700; }
    .down-signal { color:#FF4466; font-weight:700; }
    hr.gold { border:none; border-top:1px solid #C8A84B40; margin:16px 0; }
    div[data-testid="stDataFrame"] { background:#0d0d28; }
    .natal-card {
        background:#0d0d28; border:1px solid #C8A84B55;
        border-radius:6px; padding:14px 18px; margin:6px 0;
        font-size:13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def metric_html(label, value, sub=""):
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-sub'>{sub}</div></div>"
    )

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    return buf

def angular_diff(lon_transit, lon_target):
    d = (lon_transit - lon_target) % 360
    return np.where(d > 180, d - 360, d)

def angle_sign_str(lon):
    sign = SIGNS[int(lon // 30)]
    deg  = int(lon % 30)
    return f"{sign} {deg:02d}°  ({lon:.1f}°)"

def add_ct(X_asp, trend_vec):
    """Prepend constant and trend columns."""
    const = np.ones((len(trend_vec), 1))
    trend = trend_vec.reshape(-1, 1)
    return np.hstack([const, trend, X_asp])


# ── Build feature matrix (transit → natal) ────────────────────────────────────
def build_features(date_index, eph, natal_targets, orb_apply, orb_sep):
    """
    natal_targets : dict  {label: longitude}
                    e.g. {'sun': 120.3, 'moon': 45.7, 'asc': 149.0}
    Returns DataFrame of binary aspect dummies.
    """
    eph_al   = eph.reindex(date_index, method="ffill")
    avail_tp = [p for p in ALL_PLANETS if p in eph_al.columns]

    feat_cols = {}
    for tp in avail_tp:
        t_lons = eph_al[tp].values.astype(float) % 360
        motion = np.gradient(np.unwrap(t_lons, period=360))
        for nlabel, n_lon in natal_targets.items():
            for asp in ASPECTS:
                target     = (n_lon + asp) % 360
                gap        = angular_diff(t_lons, target)
                abs_gap    = np.abs(gap)
                applying   = (((motion > 0) & (gap < 0)) |
                              ((motion < 0) & (gap > 0)))
                separating = ~applying
                key_a = f"{tp}__{nlabel}__{asp}__apply"
                key_s = f"{tp}__{nlabel}__{asp}__sep"
                feat_cols[key_a] = (applying   & (abs_gap <= orb_apply)).astype(float)
                feat_cols[key_s] = (separating & (abs_gap <= orb_sep)).astype(float)

    feat_df = pd.DataFrame(feat_cols, index=date_index)
    feat_df = feat_df.loc[:, (feat_df > 0).any(axis=0)]
    return feat_df


def build_forecast_features(fut_dates, eph, natal_targets, feature_cols,
                             orb_apply, orb_sep):
    eph_fut  = eph.reindex(fut_dates, method="ffill")
    avail_tp = [p for p in ALL_PLANETS if p in eph_fut.columns]

    feat_cols_d = {}
    for tp in avail_tp:
        t_lons = eph_fut[tp].values.astype(float) % 360
        motion = np.gradient(np.unwrap(t_lons, period=360))
        for nlabel, n_lon in natal_targets.items():
            for asp in ASPECTS:
                target     = (n_lon + asp) % 360
                gap        = angular_diff(t_lons, target)
                abs_gap    = np.abs(gap)
                applying   = (((motion > 0) & (gap < 0)) |
                              ((motion < 0) & (gap > 0)))
                separating = ~applying
                feat_cols_d[f"{tp}__{nlabel}__{asp}__apply"] = (
                    applying   & (abs_gap <= orb_apply)).astype(float)
                feat_cols_d[f"{tp}__{nlabel}__{asp}__sep"] = (
                    separating & (abs_gap <= orb_sep)).astype(float)

    fut_df = pd.DataFrame(feat_cols_d, index=fut_dates)
    return fut_df.reindex(columns=feature_cols, fill_value=0.0).values


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center;margin-bottom:4px'>🌟 Natal</h2>"
        "<h2 style='text-align:center;margin-top:0'>Regression</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

    # ── Stock ─────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Asset Settings")
    ticker = st.text_input("Stock Ticker", value=DEFAULT_TICKER,
                           help="Yahoo Finance ticker, e.g. GC=F, SI=F, AAPL, XU100.IS")
    stock_name = st.text_input("Display Name", value=DEFAULT_NAME,
                               help="Used in chart titles")

    today       = datetime.date.today()
    default_end = today + datetime.timedelta(days=365)

    stock_start = st.date_input(
        "Data Start Date", value=datetime.date(1987, 1, 1),
        min_value=datetime.date(1950, 1, 1), max_value=today,
    )
    forecast_end = st.date_input(
        "Forecast End Date", value=default_end,
        min_value=today + datetime.timedelta(days=1),
    )

    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

    # ── Natal / Birth chart ───────────────────────────────────────────────────
    st.markdown("### 🎂 Natal / Birth Chart")
    natal_date_str = st.text_input(
        "Natal Date (YYYY-MM-DD)", value=DEFAULT_NATAL_DATE,
        help="Birth / listing date of the asset. Planets on this date form the natal chart.",
    )

    st.markdown("**Optional Angles** *(leave blank to skip)*")
    asc_input = st.text_input(
        "ASC Longitude (0–360°)", value=str(DEFAULT_ASC_DEG),
        help="Ascendant degree in absolute longitude. e.g. 149 = Leo 29°. Leave blank to skip.",
    )
    mc_input = st.text_input(
        "MC Longitude (0–360°)", value="",
        help="Midheaven degree in absolute longitude. Leave blank to skip.",
    )

    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

    # ── Orbs ──────────────────────────────────────────────────────────────────
    st.markdown("### 🔭 Orb Settings")
    orb_apply = st.slider("Applying Orb (°)", 0.5, 10.0, DEFAULT_ORB_APPLY, 0.5)
    orb_sep   = st.slider("Separating Orb (°)", 0.5, 5.0, DEFAULT_ORB_SEP, 0.5)

    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

    # ── ANN ───────────────────────────────────────────────────────────────────
    st.markdown("### 🧠 ANN Settings")
    ann_l1 = st.number_input("Layer 1 neurons", min_value=8, max_value=1024,
                              value=512, step=8)
    ann_l2 = st.number_input("Layer 2 neurons", min_value=0, max_value=512,
                              value=32, step=8,
                              help="Set to 0 for a single hidden layer.")
    train_frac = st.slider("Train fraction", 0.60, 0.95, TRAIN_FRAC, 0.05)

    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Run Analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center;letter-spacing:3px;margin-bottom:4px'>"
    "NATAL PLANETARY REGRESSION</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#888;font-size:14px;letter-spacing:2px'>"
    "TRANSIT-TO-NATAL ASPECTS · OLS + ANN · LOG PRICE LEVEL</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

if not run_btn:
    st.markdown(
        """
        <div style='text-align:center;padding:60px 20px;color:#555'>
            <div style='font-size:64px;margin-bottom:16px'>🌟</div>
            <div style='font-family:Cinzel,serif;font-size:20px;color:#C8A84B;margin-bottom:12px'>
                Configure Settings &amp; Run Analysis
            </div>
            <div style='font-size:14px;line-height:2.0'>
                1. Enter asset ticker, display name, and data start date<br>
                2. Enter the asset natal / birth date<br>
                3. Optionally set ASC and MC longitudes<br>
                4. Adjust orb angles and ANN architecture<br>
                5. Click <strong style='color:#C8A84B'>⚡ Run Analysis</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PARSE INPUTS
# ══════════════════════════════════════════════════════════════════════════════
# Parse natal date
try:
    natal_date = pd.Timestamp(natal_date_str)
except Exception:
    st.error(f"Invalid natal date: '{natal_date_str}'. Use YYYY-MM-DD format.")
    st.stop()

# Parse ASC / MC
natal_angles = {}
if asc_input.strip():
    try:
        natal_angles["asc"] = float(asc_input.strip()) % 360
    except ValueError:
        st.warning(f"ASC value '{asc_input}' is not a valid number — skipping ASC.")
if mc_input.strip():
    try:
        natal_angles["mc"] = float(mc_input.strip()) % 360
    except ValueError:
        st.warning(f"MC value '{mc_input}' is not a valid number — skipping MC.")

angle_labels = {"asc": "ASC", "mc": "MC"}

# ANN layers tuple
ann_layers_list = [ann_l1]
if ann_l2 > 0:
    ann_layers_list.append(ann_l2)
ann_layers = tuple(ann_layers_list)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD EPHEMERIS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_ephemeris(url: str) -> pd.DataFrame:
    return pd.read_csv(url, index_col="date", parse_dates=True)

with st.spinner("Loading ephemeris from GitHub …"):
    try:
        eph = load_ephemeris(EPHEMERIS_URL)
        st.success(
            f"✓ Ephemeris: {eph.shape[0]:,} rows · {eph.shape[1]} planets · "
            f"{eph.index[0].date()} → {eph.index[-1].date()}"
        )
    except Exception as e:
        st.error(
            f"Could not load ephemeris.\n\n**Error:** {e}\n\n"
            f"**URL:** `{EPHEMERIS_URL}`\n\nUpdate `EPHEMERIS_URL` at the top of the script."
        )
        st.stop()

# Extract natal chart
if natal_date not in eph.index:
    # Try nearest available date
    nearest = eph.index[eph.index.get_indexer([natal_date], method="nearest")[0]]
    st.warning(
        f"Natal date {natal_date.date()} not in ephemeris. "
        f"Using nearest: {nearest.date()}"
    )
    natal_date = nearest

natal_row    = eph.loc[natal_date]
natal_planets = {}
for p in ALL_PLANETS:
    if p in natal_row.index:
        natal_planets[p] = float(natal_row[p]) % 360

# Merge angles into natal targets
natal_targets = dict(natal_planets)
natal_targets.update(natal_angles)   # 'asc' / 'mc' added if present


# ══════════════════════════════════════════════════════════════════════════════
#  SHOW NATAL CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🎂 Natal Chart</h2>", unsafe_allow_html=True)

natal_cols = st.columns(4)
planet_items = list(natal_planets.items())
for i, (p, lon) in enumerate(planet_items):
    sign = SIGNS[int(lon // 30)]
    deg  = int(lon % 30)
    natal_cols[i % 4].markdown(
        f"<div class='natal-card'>"
        f"<span style='color:#888;font-size:11px;text-transform:uppercase'>{p}</span><br>"
        f"<span style='color:#C8A84B;font-size:16px;font-family:Cinzel,serif'>{sign} {deg:02d}°</span><br>"
        f"<span style='color:#555;font-size:11px'>{lon:.2f}°</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

if natal_angles:
    st.markdown("**Optional Angles:**")
    ang_cols = st.columns(len(natal_angles))
    for i, (k, lon) in enumerate(natal_angles.items()):
        ang_cols[i].markdown(
            f"<div class='natal-card'>"
            f"<span style='color:#888;font-size:11px;text-transform:uppercase'>{angle_labels[k]}</span><br>"
            f"<span style='color:#C8A84B;font-size:16px;font-family:Cinzel,serif'>{angle_sign_str(lon)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD STOCK DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"Downloading {ticker} …"):
    try:
        stock_end_str = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(
            ticker,
            start=stock_start.strftime("%Y-%m-%d"),
            end=stock_end_str,
            progress=False,
            auto_adjust=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw["Close"]     = pd.to_numeric(raw["Close"], errors="coerce")
        raw              = raw[["Close"]].dropna()
        raw["log_price"] = np.log(raw["Close"])
        st.success(
            f"✓ {ticker}: {len(raw):,} trading days "
            f"({raw.index[0].date()} → {raw.index[-1].date()})"
        )
    except Exception as e:
        st.error(f"Failed to download {ticker}: {e}")
        st.stop()

price_df  = raw
dates_all = price_df.index
y_all     = price_df["log_price"].values
close_all = price_df["Close"].values
n_total   = len(dates_all)


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Building transit-to-natal aspect features …"):
    feat_all     = build_features(dates_all, eph, natal_targets, orb_apply, orb_sep)
    feature_cols = feat_all.columns.tolist()
    n_features   = len(feature_cols)
    n_apply_cols = sum(1 for c in feature_cols if c.endswith("__apply"))
    n_sep_cols   = sum(1 for c in feature_cols if c.endswith("__sep"))

st.markdown(
    f"<div class='metric-card'>"
    f"<span class='metric-label'>Features built (transit→natal)</span><br>"
    f"<span class='metric-value'>{n_features}</span> "
    f"<span class='metric-sub'>"
    f"{n_apply_cols} applying + {n_sep_cols} separating · "
    f"Natal targets: {len(natal_targets)} "
    f"({len(natal_planets)} planets"
    f"{', ' + ', '.join(angle_labels[k] for k in natal_angles) if natal_angles else ''}) · "
    f"Orb: apply≤{orb_apply}° / sep≤{orb_sep}°"
    f"</span></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLIT + DESIGN MATRIX
# ══════════════════════════════════════════════════════════════════════════════
n_train     = int(n_total * train_frac)
dates_train = dates_all[:n_train]
dates_test  = dates_all[n_train:]
y_train     = y_all[:n_train]
y_test      = y_all[n_train:]

trend_all   = np.arange(n_total) / n_total
trend_train = trend_all[:n_train]
trend_test  = trend_all[n_train:]

X_train_asp = feat_all.values[:n_train]
X_test_asp  = feat_all.values[n_train:]
X_all_asp   = feat_all.values

X_train = add_ct(X_train_asp, trend_train)
X_test  = add_ct(X_test_asp,  trend_test)
X_all   = add_ct(X_all_asp,   trend_all)

col_names = ["const", "trend"] + feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — OLS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Fitting OLS …"):
    ols_res = sm.OLS(y_train, X_train).fit()

y_fit_ols_train = np.asarray(ols_res.predict(X_train))
y_fit_ols_test  = np.asarray(ols_res.predict(X_test))

def metrics_dict(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r    = float(np.corrcoef(y_true, y_pred)[0, 1])
    return dict(r2=r2, rmse=rmse, mae=mae, r=r)

m_ols_train = metrics_dict(y_train, y_fit_ols_train)
m_ols_test  = metrics_dict(y_test,  y_fit_ols_test)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — ANN on detrended residuals
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"Fitting ANN {ann_layers} on detrended residuals …"):
    X_trend_train = X_train[:, :2]
    X_trend_test  = X_test[:, :2]
    X_trend_all   = X_all[:, :2]

    ols_trend       = sm.OLS(y_train, X_trend_train).fit()
    trend_fit_train = np.asarray(ols_trend.predict(X_trend_train))
    trend_fit_test  = np.asarray(ols_trend.predict(X_trend_test))
    trend_fit_all   = np.asarray(ols_trend.predict(X_trend_all))

    resid_train = y_train - trend_fit_train

    sc_resid        = StandardScaler()
    X_asp_tr_sc     = sc_resid.fit_transform(X_train[:, 2:])
    X_asp_te_sc     = sc_resid.transform(X_test[:, 2:])

    ann = MLPRegressor(
        hidden_layer_sizes=ann_layers,
        activation="relu", solver="adam", alpha=0.01,
        learning_rate="adaptive", learning_rate_init=0.001,
        max_iter=9000, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=50,
        random_state=42, verbose=False,
    )
    ann.fit(X_asp_tr_sc, resid_train)

resid_pred_train = ann.predict(X_asp_tr_sc)
resid_pred_test  = ann.predict(X_asp_te_sc)
y_fit_ann_train  = trend_fit_train + resid_pred_train
y_fit_ann_test   = trend_fit_test  + resid_pred_test

m_ann_train = metrics_dict(y_train, y_fit_ann_train)
m_ann_test  = metrics_dict(y_test,  y_fit_ann_test)


# ══════════════════════════════════════════════════════════════════════════════
#  REFIT FULL SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Refitting on full sample for forecast …"):
    ols_full       = sm.OLS(y_all, X_all).fit()
    y_fit_ols_full = np.asarray(ols_full.fittedvalues)

    ols_trend_full     = sm.OLS(y_all, X_trend_all).fit()
    trend_fit_all_full = np.asarray(ols_trend_full.predict(X_trend_all))
    resid_all          = y_all - trend_fit_all_full

    sc_resid_full   = StandardScaler()
    X_asp_all_sc    = sc_resid_full.fit_transform(X_all[:, 2:])

    ann_full = MLPRegressor(
        hidden_layer_sizes=ann_layers,
        activation="relu", solver="adam", alpha=0.01,
        learning_rate="adaptive", learning_rate_init=0.001,
        max_iter=5000, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=50,
        random_state=42, verbose=False,
    )
    ann_full.fit(X_asp_all_sc, resid_all)

resid_pred_all = ann_full.predict(X_asp_all_sc)
y_fit_ann_full = trend_fit_all_full + resid_pred_all

fit_ols_all  = np.exp(y_fit_ols_full)
fit_ann_all  = np.exp(y_fit_ann_full)
last_fit_ols = float(fit_ols_all[-1])
last_fit_ann = float(fit_ann_all[-1])
last_actual  = float(close_all[-1])


# ══════════════════════════════════════════════════════════════════════════════
#  FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Building forecast …"):
    forecast_start = (dates_all[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fut_dates      = pd.date_range(
        start=forecast_start,
        end=forecast_end.strftime("%Y-%m-%d"),
        freq="B",
    )
    n_fut = len(fut_dates)

    X_fut_asp    = build_forecast_features(
        fut_dates, eph, natal_targets, feature_cols, orb_apply, orb_sep
    )
    trend_fut    = (n_total + np.arange(n_fut)) / n_total
    X_fut        = add_ct(X_fut_asp, trend_fut)
    y_fore_ols   = np.asarray(ols_full.predict(X_fut))
    fore_ols_idx = np.exp(y_fore_ols)

    trend_fore   = np.asarray(ols_trend_full.predict(X_fut[:, :2]))
    X_fut_asp_sc = sc_resid_full.transform(X_fut[:, 2:])
    resid_fore   = ann_full.predict(X_fut_asp_sc)
    y_fore_ann   = trend_fore + resid_fore
    fore_ann_idx = np.exp(y_fore_ann)

ols_full_r2 = r2_score(y_all, y_fit_ols_full)
ann_full_r2 = r2_score(y_all, y_fit_ann_full)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📊 Model Performance</h2>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(metric_html("Last Actual",       f"{last_actual:,.4f}",    f"{ticker} · {dates_all[-1].date()}"),       unsafe_allow_html=True)
c2.markdown(metric_html("OLS Full R²",       f"{ols_full_r2:.4f}",     "Full sample"),                              unsafe_allow_html=True)
c3.markdown(metric_html("ANN Full R²",       f"{ann_full_r2:.4f}",     "Full sample"),                              unsafe_allow_html=True)
c4.markdown(metric_html("OLS Test R²",       f"{m_ols_test['r2']:.4f}", f"r={m_ols_test['r']:.4f}"),               unsafe_allow_html=True)
c5.markdown(metric_html("ANN Test R²",       f"{m_ann_test['r2']:.4f}", f"r={m_ann_test['r']:.4f}"),               unsafe_allow_html=True)
c6.markdown(metric_html("Features (active)", f"{n_features}",           f"Natal targets: {len(natal_targets)}"),    unsafe_allow_html=True)

# Detailed performance table
perf_data = {
    "Model":    ["OLS", "OLS", "ANN", "ANN"],
    "Split":    ["Train", "Test", "Train", "Test"],
    "R²":       [m_ols_train["r2"],  m_ols_test["r2"],
                 m_ann_train["r2"],  m_ann_test["r2"]],
    "Pearson r":[m_ols_train["r"],   m_ols_test["r"],
                 m_ann_train["r"],   m_ann_test["r"]],
    "RMSE":     [m_ols_train["rmse"],m_ols_test["rmse"],
                 m_ann_train["rmse"],m_ann_test["rmse"]],
    "MAE":      [m_ols_train["mae"], m_ols_test["mae"],
                 m_ann_train["mae"], m_ann_test["mae"]],
}
perf_df = pd.DataFrame(perf_data)

def style_perf(df):
    def row_style(row):
        col = ORANGE if row["Model"] == "OLS" else "#BB33FF"
        bg  = "#0d0d28"
        return [f"color:{col};background-color:{bg}"] * len(row)
    return df.style.apply(row_style, axis=1).format({
        "R²": "{:.4f}", "Pearson r": "{:.4f}", "RMSE": "{:.6f}", "MAE": "{:.6f}"
    })

st.dataframe(style_perf(perf_df), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  NEXT 3 TRADING DAYS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📅 Next 3 Trading Days Forecast</h2>", unsafe_allow_html=True)

cols3 = st.columns(3)
for k in range(min(3, n_fut)):
    ols_lvl  = fore_ols_idx[k]
    ann_lvl  = fore_ann_idx[k]
    ols_vs   = ols_lvl - last_fit_ols
    ann_vs   = ann_lvl - last_fit_ann
    ols_pct  = (ols_lvl / last_fit_ols - 1) * 100
    ann_pct  = (ann_lvl / last_fit_ann - 1) * 100
    ols_dir  = "▲ UP" if ols_vs > 0 else "▼ DOWN"
    ann_dir  = "▲ UP" if ann_vs > 0 else "▼ DOWN"
    o_cls    = "up-signal" if ols_vs > 0 else "down-signal"
    a_cls    = "up-signal" if ann_vs > 0 else "down-signal"
    cons     = "CONSENSUS" if ols_dir == ann_dir else "SPLIT"
    cons_c   = "#44DD88" if cons == "CONSENSUS" else "#FF8844"
    with cols3[k]:
        st.markdown(
            f"""<div class='forecast-box'>
            <div style='color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Day +{k+1}</div>
            <div style='font-family:Cinzel,serif;font-size:14px;color:#C8A84B;margin:4px 0'>{fut_dates[k].strftime('%B %d, %Y')}</div>
            <div style='font-size:18px;font-weight:700;margin:8px 0'>
                <span style='color:{cons_c}'>{cons}</span>
            </div>
            <div style='margin:4px 0'>
                <span style='color:{ORANGE};font-size:13px'>OLS</span>
                <span style='color:#888;font-size:11px'> → </span>
                <span class='{o_cls}'>{ols_dir}</span>
                <span style='color:#aaa;font-size:11px'> ({ols_pct:+.3f}%)</span>
            </div>
            <div style='margin:4px 0'>
                <span style='color:#CC44FF;font-size:13px'>ANN</span>
                <span style='color:#888;font-size:11px'> → </span>
                <span class='{a_cls}'>{ann_dir}</span>
                <span style='color:#aaa;font-size:11px'> ({ann_pct:+.3f}%)</span>
            </div>
            <div style='margin-top:10px;padding-top:8px;border-top:1px solid #2A2A4A;font-size:11px;color:#555'>
                OLS: {ols_lvl:,.4f} · ANN: {ann_lvl:,.4f}
            </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 1 — Last 12M Fitted + 3M Forecast
#  Fitted lines rebased to actual price at window start for comparable scaling
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📈 Chart 1 — Last 12 Months Fitted + 3-Month Forecast</h2>", unsafe_allow_html=True)

TRADING_MONTH   = 63
last_year_start = dates_all[-1] - pd.DateOffset(months=12)
mask_ly         = dates_all >= last_year_start
dates_ly        = dates_all[mask_ly]
actual_ly       = close_all[mask_ly]

# Rebase fitted series to actual price at window start
w_pos = int(np.where(mask_ly)[0][0])

# OLS rebased: cumulate log-fitted increments within window, start at actual_ly[0]
ols_log_win = y_fit_ols_full[w_pos:]
ols_fit_ly  = actual_ly[0] * np.exp(np.cumsum(ols_log_win) - ols_log_win[0])

# ANN rebased: same approach
ann_log_win = y_fit_ann_full[w_pos:]
ann_fit_ly  = actual_ly[0] * np.exp(np.cumsum(ann_log_win) - ann_log_win[0])

fore_dates_3m = fut_dates[:TRADING_MONTH]
fore_ols_3m   = fore_ols_idx[:TRADING_MONTH]
fore_ann_3m   = fore_ann_idx[:TRADING_MONTH]

r_ols_ly    = float(np.corrcoef(y_all[mask_ly], y_fit_ols_full[mask_ly])[0, 1])
r_ann_ly    = float(np.corrcoef(y_all[mask_ly], y_fit_ann_full[mask_ly])[0, 1])
rmse_ols_ly = float(np.sqrt(np.mean((y_all[mask_ly] - y_fit_ols_full[mask_ly])**2)))
rmse_ann_ly = float(np.sqrt(np.mean((y_all[mask_ly] - y_fit_ann_full[mask_ly])**2)))

all_vals  = np.concatenate([
    actual_ly, ols_fit_ly, ann_fit_ly,
    fore_ols_3m if len(fore_ols_3m) else np.array([actual_ly[-1]]),
    fore_ann_3m if len(fore_ann_3m) else np.array([actual_ly[-1]]),
])
y_top_val = float(all_vals.max())

fig1, ax = plt.subplots(figsize=(18, 6), facecolor=BG)
ax.set_facecolor("#0D0D28")
for sp in ax.spines.values(): sp.set_color(GREY)
ax.tick_params(colors=WHITE, labelsize=8)

ax.plot(dates_ly, actual_ly,  color=TEAL,   lw=2.5, zorder=6, label=f"Actual {stock_name}")
ax.plot(dates_ly, ols_fit_ly, color=ORANGE, lw=1.4, alpha=0.85, zorder=4,
        label=f"OLS fitted  r={r_ols_ly:.3f}  RMSE(log)={rmse_ols_ly:.4f}  (rebased)")
ax.plot(dates_ly, ann_fit_ly, color=PURPLE, lw=1.4, alpha=0.85, zorder=4,
        label=f"ANN fitted  r={r_ann_ly:.3f}  RMSE(log)={rmse_ann_ly:.4f}  (rebased)")

if len(fore_dates_3m):
    ax.axvline(dates_ly[-1], color=GOLD, lw=1.5, ls=":", alpha=0.9, zorder=5)
    ax.text(dates_ly[-1], y_top_val * 1.002, "  Forecast→",
            color=GOLD, fontsize=8, va="bottom", fontweight="bold")
    ax.axvspan(fore_dates_3m[0], fore_dates_3m[-1], alpha=0.06, color=GOLD, zorder=1)

    # Connectors from rebased fitted end to forecast start (anchored at last_actual)
    ax.plot([dates_ly[-1], fore_dates_3m[0]], [float(ols_fit_ly[-1]), fore_ols_3m[0]],
            color=ORANGE, lw=1.2, alpha=0.7, zorder=4)
    ax.plot(fore_dates_3m, fore_ols_3m, color=ORANGE, lw=2.0, ls="--", zorder=5,
            label=(f"OLS 3M  end={fore_ols_3m[-1]:,.4f}  "
                   f"({(fore_ols_3m[-1]/last_fit_ols-1)*100:+.1f}% vs fitted)"))
    ax.scatter([fore_dates_3m[-1]], [fore_ols_3m[-1]], color=ORANGE, s=60, zorder=8)
    ax.annotate(f"{fore_ols_3m[-1]:,.4f}",
                xy=(fore_dates_3m[-1], fore_ols_3m[-1]),
                xytext=(8, 4), textcoords="offset points",
                color=ORANGE, fontsize=8, fontweight="bold")

    ax.plot([dates_ly[-1], fore_dates_3m[0]], [float(ann_fit_ly[-1]), fore_ann_3m[0]],
            color=PURPLE, lw=1.2, alpha=0.7, zorder=4)
    ax.plot(fore_dates_3m, fore_ann_3m, color=PURPLE, lw=2.0, ls="--", zorder=5,
            label=(f"ANN 3M  end={fore_ann_3m[-1]:,.4f}  "
                   f"({(fore_ann_3m[-1]/last_fit_ann-1)*100:+.1f}% vs fitted)"))
    ax.scatter([fore_dates_3m[-1]], [fore_ann_3m[-1]], color=PURPLE, s=60, zorder=8)
    ax.annotate(f"{fore_ann_3m[-1]:,.4f}",
                xy=(fore_dates_3m[-1], fore_ann_3m[-1]),
                xytext=(8, -14), textcoords="offset points",
                color=PURPLE, fontsize=8, fontweight="bold")

ax.scatter([dates_ly[-1]], [last_actual], color=TEAL, s=70, zorder=8,
           label=f"Last actual: {last_actual:,.4f}")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Price Level", color=WHITE, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.4f}"))
angle_sub = ""
if natal_angles:
    angle_sub = "  |  Angles: " + "  ".join(
        f"{angle_labels[k]}={angle_sign_str(v)}" for k, v in natal_angles.items()
    )
ax.set_title(
    f"{stock_name} ({ticker}) — Last 12M Fitted + 3M Forecast  (OLS & ANN)\n"
    f"Natal: {natal_date.date()}  ·  Fitted rebased to window-start price  ·  "
    f"Features: {n_features}  ·  Orb: apply≤{orb_apply}°/sep≤{orb_sep}°{angle_sub}",
    color=GOLD, fontsize=11, fontweight="bold",
)
ax.legend(fontsize=8, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left")
fig1.tight_layout()
st.image(fig_to_buf(fig1), use_container_width=True)
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 2 — 90-Day Forecast with Peaks & Troughs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🔍 Chart 2 — 90-Day Forecast (Peaks &amp; Troughs)</h2>", unsafe_allow_html=True)

FORECAST_DAYS = 90
fore_dates_90 = fut_dates[:FORECAST_DAYS]
fore_ols_90   = fore_ols_idx[:FORECAST_DAYS]
fore_ann_90   = fore_ann_idx[:FORECAST_DAYS]
fore_avg_90   = (fore_ols_90 + fore_ann_90) / 2.0

last_6m_start = dates_all[-1] - pd.DateOffset(months=6)
mask_6m       = dates_all >= last_6m_start
dates_6m      = dates_all[mask_6m]
actual_6m     = close_all[mask_6m]
fit_ols_6m    = fit_ols_all[mask_6m]
fit_ann_6m    = fit_ann_all[mask_6m]

r_ols_6m = float(np.corrcoef(y_all[mask_6m], y_fit_ols_full[mask_6m])[0, 1])
r_ann_6m = float(np.corrcoef(y_all[mask_6m], y_fit_ann_full[mask_6m])[0, 1])

peak_idx, trough_idx = [], []
if len(fore_avg_90) >= 7:
    W_pt = 3
    for i in range(W_pt, len(fore_avg_90) - W_pt):
        w = fore_avg_90[i - W_pt:i + W_pt + 1]
        if fore_avg_90[i] == w.max() and fore_avg_90[i] > fore_avg_90[i-1] and fore_avg_90[i] > fore_avg_90[i+1]:
            peak_idx.append(i)
        if fore_avg_90[i] == w.min() and fore_avg_90[i] < fore_avg_90[i-1] and fore_avg_90[i] < fore_avg_90[i+1]:
            trough_idx.append(i)
    ai = int(np.argmax(fore_avg_90)); ni = int(np.argmin(fore_avg_90))
    if ai not in peak_idx:   peak_idx.append(ai)
    if ni not in trough_idx: trough_idx.append(ni)
    peak_idx   = sorted(set(peak_idx))
    trough_idx = sorted(set(trough_idx))

fig2, ax = plt.subplots(figsize=(22, 8), facecolor=BG)
ax.set_facecolor("#0D0D28")
for sp in ax.spines.values(): sp.set_color(GREY)
ax.tick_params(colors=WHITE, labelsize=8.5)

ax.plot(dates_6m, actual_6m, color=TEAL,   lw=2.5, zorder=6, label=f"Actual {stock_name} (6M)")
ax.plot(dates_6m, fit_ols_6m, color=ORANGE, lw=1.4, alpha=0.80, zorder=4,
        label=f"OLS fitted  r={r_ols_6m:.3f}")
ax.plot(dates_6m, fit_ann_6m, color=PURPLE, lw=1.4, alpha=0.80, zorder=4,
        label=f"ANN fitted  r={r_ann_6m:.3f}")

if len(fore_dates_90):
    ax.axvline(fore_dates_90[0], color=GOLD, lw=2.0, ls=":", alpha=0.95, zorder=5)
    ax.axvspan(fore_dates_90[0], fore_dates_90[-1], alpha=0.07, color=GOLD, zorder=1)

    ax.plot([dates_6m[-1], fore_dates_90[0]], [last_fit_ols, fore_ols_90[0]],
            color=ORANGE, lw=1.2, alpha=0.7, zorder=4)
    ax.plot([dates_6m[-1], fore_dates_90[0]], [last_fit_ann, fore_ann_90[0]],
            color=PURPLE, lw=1.2, alpha=0.7, zorder=4)

    ax.plot(fore_dates_90, fore_ols_90, color=ORANGE, lw=2.2, ls="--", zorder=5,
            label=(f"OLS 90d  end={fore_ols_90[-1]:,.4f}  "
                   f"({(fore_ols_90[-1]/last_fit_ols-1)*100:+.2f}%)"))
    ax.plot(fore_dates_90, fore_ann_90, color=PURPLE, lw=2.2, ls="--", zorder=5,
            label=(f"ANN 90d  end={fore_ann_90[-1]:,.4f}  "
                   f"({(fore_ann_90[-1]/last_fit_ann-1)*100:+.2f}%)"))

    all_y = np.concatenate([actual_6m, fit_ols_6m, fit_ann_6m,
                             fore_ols_90, fore_ann_90])
    y_min_ax = all_y.min() * 0.993; y_max_ax = all_y.max() * 1.012
    lyt = y_max_ax * 0.9985; lyb = y_min_ax * 1.0015

    for i in peak_idx:
        d = fore_dates_90[i]
        ov = fore_ols_90[i]; av = fore_ann_90[i]
        ax.axvline(d, color="#FF4466", lw=1.3, ls="--", alpha=0.85, zorder=6)
        ax.scatter([d, d], [ov, av], color="#FF4466", s=55, zorder=8, marker="^")
        ax.text(d, lyt, f"  ▲ {d.strftime('%b %d')}\n  OLS {ov:,.2f}\n  ANN {av:,.2f}",
                color="#FF8899", fontsize=7, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a0010",
                          edgecolor="#FF4466", alpha=0.85))

    for i in trough_idx:
        d = fore_dates_90[i]
        ov = fore_ols_90[i]; av = fore_ann_90[i]
        ax.axvline(d, color="#44FF88", lw=1.3, ls="--", alpha=0.85, zorder=6)
        ax.scatter([d, d], [ov, av], color="#44FF88", s=55, zorder=8, marker="v")
        ax.text(d, lyb, f"  ▼ {d.strftime('%b %d')}\n  OLS {ov:,.2f}\n  ANN {av:,.2f}",
                color="#88FFAA", fontsize=7, va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#001a0a",
                          edgecolor="#44FF88", alpha=0.85))

    ax.set_ylim(y_min_ax, y_max_ax)

ax.scatter([dates_6m[-1]], [last_actual], color=TEAL, s=80, zorder=8,
           label=f"Last actual: {last_actual:,.4f}")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
ax.xaxis.grid(True, which="major", color=GREY, alpha=0.35, lw=0.6)
ax.yaxis.grid(True, color=GREY, alpha=0.25, lw=0.5)
ax.set_ylabel("Price Level", color=WHITE, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.4f}"))
ax.set_title(
    f"{stock_name} ({ticker}) — 90-Day OLS & ANN Forecast  (peaks & troughs)\n"
    f"▲Red=Peak (avg OLS+ANN)  ▼Green=Trough  ·  "
    f"Natal: {natal_date.date()}  ·  Features: {n_features}{angle_sub}",
    color=GOLD, fontsize=11, fontweight="bold",
)
ax.legend(fontsize=9, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left", ncol=2)
fig2.tight_layout(pad=2.0)
st.image(fig_to_buf(fig2), use_container_width=True)
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 3 — OLS Full-Period Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📉 Chart 3 — OLS Full-Period Forecast</h2>", unsafe_allow_html=True)

fig3, ax = plt.subplots(figsize=(20, 7), facecolor=BG)
ax.set_facecolor("#0D0D28")
for sp in ax.spines.values(): sp.set_color(GREY)
ax.tick_params(colors=WHITE, labelsize=8.5)

ax.fill_between(fut_dates, last_actual, fore_ols_idx,
                where=(fore_ols_idx >= last_actual),
                color="#00AA55", alpha=0.18, zorder=2, label="Cumulative gain")
ax.fill_between(fut_dates, last_actual, fore_ols_idx,
                where=(fore_ols_idx < last_actual),
                color="#FF3355", alpha=0.18, zorder=2, label="Cumulative loss")
ax.axhline(last_actual, color=GOLD, lw=1.2, ls=":", alpha=0.8, zorder=3,
           label=f"Last actual: {last_actual:,.4f}")
ax.plot(fut_dates, fore_ols_idx, color=ORANGE, lw=2.5, zorder=5, label="OLS forecast")

peak_yr   = int(np.argmax(fore_ols_idx))
trough_yr = int(np.argmin(fore_ols_idx))
for idx, col, marker, label_s in [
    (peak_yr,   "#FF4466", "^", "Peak"),
    (trough_yr, "#44FF88", "v", "Trough"),
]:
    d = fut_dates[idx]; v = fore_ols_idx[idx]
    ax.scatter([d], [v], color=col, s=100, zorder=9, marker=marker)
    ax.axvline(d, color=col, lw=1.0, ls="--", alpha=0.6, zorder=4)
    va = "bottom" if marker == "^" else "top"
    dy = v * 1.003 if marker == "^" else v * 0.997
    ax.annotate(
        f"  {label_s}\n  {d.strftime('%b %d, %Y')}\n  {v:,.4f}",
        xy=(d, v), xytext=(d, dy), color="#FF8899" if marker == "^" else "#88FFAA",
        fontsize=8.5, va=va, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#1a0010" if marker == "^" else "#001a0a",
                  edgecolor=col, alpha=0.9),
    )

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8.5)
ax.xaxis.grid(True, which="major", color=GREY, alpha=0.40, lw=0.7)
ax.yaxis.grid(True, color=GREY, alpha=0.25, lw=0.5)
ax.set_ylabel("Price Level (OLS)", color=WHITE, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.4f}"))
chg3 = (fore_ols_idx[-1] / last_actual - 1) * 100
ax.set_title(
    f"{stock_name} ({ticker}) — OLS Forecast  "
    f"{fut_dates[0].strftime('%b %d, %Y')} → {fut_dates[-1].strftime('%b %d, %Y')}\n"
    f"End: {fore_ols_idx[-1]:,.4f}  ({chg3:+.2f}% vs last actual)  ·  "
    f"Natal: {natal_date.date()}  ·  Features: {n_features}",
    color=GOLD, fontsize=11, fontweight="bold",
)
ax.legend(fontsize=9, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left", ncol=2)
fig3.tight_layout(pad=2.0)
st.image(fig_to_buf(fig3), use_container_width=True)
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 4 — ANN Full-Period Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🧠 Chart 4 — ANN Full-Period Forecast</h2>", unsafe_allow_html=True)

fig4, ax = plt.subplots(figsize=(20, 7), facecolor=BG)
ax.set_facecolor("#0D0D28")
for sp in ax.spines.values(): sp.set_color(GREY)
ax.tick_params(colors=WHITE, labelsize=8.5)

ax.fill_between(fut_dates, last_actual, fore_ann_idx,
                where=(fore_ann_idx >= last_actual),
                color="#00AA55", alpha=0.18, zorder=2, label="Cumulative gain")
ax.fill_between(fut_dates, last_actual, fore_ann_idx,
                where=(fore_ann_idx < last_actual),
                color="#FF3355", alpha=0.18, zorder=2, label="Cumulative loss")
ax.axhline(last_actual, color=GOLD, lw=1.2, ls=":", alpha=0.8, zorder=3,
           label=f"Last actual: {last_actual:,.4f}")
ax.plot(fut_dates, fore_ann_idx, color=PURPLE, lw=2.5, zorder=5,
        label=f"ANN forecast  {ann_layers}")

peak_ann   = int(np.argmax(fore_ann_idx))
trough_ann = int(np.argmin(fore_ann_idx))
for idx, col, marker, label_s in [
    (peak_ann,   "#FF4466", "^", "Peak"),
    (trough_ann, "#44FF88", "v", "Trough"),
]:
    d = fut_dates[idx]; v = fore_ann_idx[idx]
    ax.scatter([d], [v], color=col, s=100, zorder=9, marker=marker)
    ax.axvline(d, color=col, lw=1.0, ls="--", alpha=0.6, zorder=4)
    va = "bottom" if marker == "^" else "top"
    dy = v * 1.003 if marker == "^" else v * 0.997
    ax.annotate(
        f"  {label_s}\n  {d.strftime('%b %d, %Y')}\n  {v:,.4f}",
        xy=(d, v), xytext=(d, dy), color="#FF8899" if marker == "^" else "#88FFAA",
        fontsize=8.5, va=va, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#1a0010" if marker == "^" else "#001a0a",
                  edgecolor=col, alpha=0.9),
    )

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8.5)
ax.xaxis.grid(True, which="major", color=GREY, alpha=0.40, lw=0.7)
ax.yaxis.grid(True, color=GREY, alpha=0.25, lw=0.5)
ax.set_ylabel("Price Level (ANN)", color=WHITE, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.4f}"))
chg4 = (fore_ann_idx[-1] / last_actual - 1) * 100
ax.set_title(
    f"{stock_name} ({ticker}) — ANN Forecast  "
    f"{fut_dates[0].strftime('%b %d, %Y')} → {fut_dates[-1].strftime('%b %d, %Y')}\n"
    f"End: {fore_ann_idx[-1]:,.4f}  ({chg4:+.2f}% vs last actual)  ·  "
    f"ANN {ann_layers}  ·  Natal: {natal_date.date()}  ·  Features: {n_features}",
    color=GOLD, fontsize=11, fontweight="bold",
)
ax.legend(fontsize=9, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left", ncol=2)
fig4.tight_layout(pad=2.0)
st.image(fig_to_buf(fig4), use_container_width=True)
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
#  ACTIVE ASPECTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🔭 Active Aspects — Last 7 Days + Next 30 Days</h2>", unsafe_allow_html=True)

ols_full_params = pd.Series(np.asarray(ols_full.params),  index=col_names)
ols_full_pvals  = pd.Series(np.asarray(ols_full.pvalues), index=col_names)

with st.spinner("Computing active transit-to-natal aspects …"):
    last_date    = dates_all[-1]
    window_start = last_date - pd.Timedelta(days=10)
    window_end   = last_date + pd.Timedelta(days=30)
    past_dates_w = dates_all[dates_all >= window_start]
    fut_dates_w  = pd.date_range(
        start=last_date + pd.Timedelta(days=1), end=window_end, freq="B"
    )
    all_win_d = past_dates_w.append(fut_dates_w)

    eph_win  = eph.reindex(all_win_d, method="ffill")
    avail_tp = [p for p in ALL_PLANETS if p in eph_win.columns]

    rows = []
    for tp in avail_tp:
        t_lons = eph_win[tp].values.astype(float) % 360
        motion = np.gradient(np.unwrap(t_lons, period=360))
        for nlabel, n_lon in natal_targets.items():
            for asp in ASPECTS:
                target     = (n_lon + asp) % 360
                gap        = angular_diff(t_lons, target)
                abs_gap    = np.abs(gap)
                applying   = (((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0)))
                separating = ~applying
                for col_key, phase, mask, olim in [
                    (f"{tp}__{nlabel}__{asp}__apply", "Applying",   applying,   orb_apply),
                    (f"{tp}__{nlabel}__{asp}__sep",   "Separating", separating, orb_sep),
                ]:
                    if col_key not in ols_full_params.index:
                        continue
                    coef = ols_full_params[col_key]
                    for i, date in enumerate(all_win_d):
                        if mask[i] and abs_gap[i] <= olim:
                            rows.append({
                                "Date":      date.date(),
                                "Period":    "PAST" if date <= last_date else "FUTURE",
                                "Transit":   tp,
                                "Aspect":    ASP_NAMES[asp],
                                "Natal":     angle_labels.get(nlabel, nlabel),
                                "Phase":     phase,
                                "Orb °":     round(float(abs_gap[i]), 2),
                                "OLS Coef":  round(float(coef), 6),
                                "Effect %":  round(float(coef) * 100, 3),
                                "Direction": "▲ Bullish" if coef > 0 else "▼ Bearish",
                                "p-value":   round(float(ols_full_pvals[col_key]), 4),
                            })

    asp_table = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

# Daily net effect
if len(asp_table):
    daily_net = (
        asp_table.groupby(["Date", "Period"])
        .agg(N_Aspects=("OLS Coef", "count"), Net_Coef=("OLS Coef", "sum"))
        .reset_index()
        .sort_values("Date")
    )
    daily_net["Net Effect %"] = (daily_net["Net_Coef"] * 100).round(3)
    daily_net["Bias"] = daily_net["Net_Coef"].apply(
        lambda x: "▲ Bullish" if x > 0 else "▼ Bearish"
    )

    tab1, tab2, tab3 = st.tabs(["📅 Last 7 Days", "🔮 Next 30 Days", "📊 Daily Net Effect"])

    def style_aspect_df(df):
        def row_color(row):
            color = "#0a1a0f" if row["Direction"] == "▲ Bullish" else "#1a0a0f"
            dir_c = "#44DD88" if row["Direction"] == "▲ Bullish" else "#FF4466"
            return (["background-color:" + color] * (len(row) - 1)
                    + [f"color:{dir_c};background-color:{color}"])
        return df.style.apply(row_color, axis=1).format({
            "Orb °": "{:.2f}", "OLS Coef": "{:+.6f}",
            "Effect %": "{:+.3f}", "p-value": "{:.4f}",
        })

    past_asp = asp_table[asp_table["Period"] == "PAST"].drop(columns=["Period"])
    fut_asp  = asp_table[asp_table["Period"] == "FUTURE"].drop(columns=["Period"])

    with tab1:
        st.markdown(f"**{len(past_asp)} active aspect-days**")
        if len(past_asp):
            st.dataframe(style_aspect_df(past_asp), use_container_width=True, height=400)
        else:
            st.info("No active aspects in the last 7 trading days.")

    with tab2:
        st.markdown(f"**{len(fut_asp)} active aspect-days**")
        if len(fut_asp):
            st.dataframe(style_aspect_df(fut_asp), use_container_width=True, height=400)
        else:
            st.info("No active aspects in the next 30 days.")

    with tab3:
        def style_net(df):
            def row_color(row):
                color = "#0a1a0f" if row["Bias"] == "▲ Bullish" else "#1a0a0f"
                dir_c = "#44DD88" if row["Bias"] == "▲ Bullish" else "#FF4466"
                return (["background-color:" + color] * (len(row) - 1)
                        + [f"color:{dir_c};background-color:{color}"])
            return df.style.apply(row_color, axis=1).format({
                "Net_Coef": "{:+.6f}", "Net Effect %": "{:+.3f}"
            })
        st.markdown(f"**Daily net planetary effect (sum of active OLS coefs)**")
        st.dataframe(style_net(daily_net), use_container_width=True)
else:
    st.info("No active aspects found in the window.")


# ══════════════════════════════════════════════════════════════════════════════
#  LAST 10 DAYS SIGNAL TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🕐 Last 10 Trading Days — Directional Signal</h2>", unsafe_allow_html=True)

def dir_arrow(chg):
    if   chg > 0: return "▲ UP"
    elif chg < 0: return "▼ DOWN"
    else:         return "─ FLAT"

hist_rows = []
for pos in range(max(0, n_total - 10), n_total):
    d          = dates_all[pos]
    close_val  = float(price_df.loc[d, "Close"])
    prev       = float(price_df.iloc[pos - 1]["Close"]) if pos > 0 else float("nan")
    price_chg  = close_val - prev
    price_pct  = price_chg / prev * 100 if (not np.isnan(prev) and prev != 0) else float("nan")
    price_dir  = dir_arrow(price_chg)
    ols_t      = fit_ols_all[pos]
    ols_tm1    = fit_ols_all[pos - 1] if pos > 0 else float("nan")
    ols_chg    = ols_t - ols_tm1
    ols_pct    = ols_chg / ols_tm1 * 100 if (not np.isnan(ols_tm1) and ols_tm1 != 0) else float("nan")
    ols_dir    = dir_arrow(ols_chg)
    ann_t      = fit_ann_all[pos]
    ann_tm1    = fit_ann_all[pos - 1] if pos > 0 else float("nan")
    ann_chg    = ann_t - ann_tm1
    ann_pct    = ann_chg / ann_tm1 * 100 if (not np.isnan(ann_tm1) and ann_tm1 != 0) else float("nan")
    ann_dir    = dir_arrow(ann_chg)
    hist_rows.append({
        "Date":       str(d.date()),
        "Close":      round(close_val, 4),
        "Actual %":   round(price_pct, 2) if not np.isnan(price_pct) else None,
        "Actual Dir": price_dir,
        "OLS %":      round(ols_pct, 3)  if not np.isnan(ols_pct)   else None,
        "OLS Dir":    ols_dir,
        "OLS ✓":      "✓" if ols_dir.strip() == price_dir.strip() else "✗",
        "ANN %":      round(ann_pct, 3)  if not np.isnan(ann_pct)   else None,
        "ANN Dir":    ann_dir,
        "ANN ✓":      "✓" if ann_dir.strip() == price_dir.strip() else "✗",
    })

hist_df = pd.DataFrame(hist_rows)

def style_hist(df):
    def row_style(row):
        styles = []
        bg = "#0d0d28"
        for col in df.columns:
            val = row[col]
            if col in ("OLS ✓", "ANN ✓"):
                c = "#44DD88" if val == "✓" else "#FF4466"
                styles.append(f"color:{c};background-color:{bg};font-weight:700")
            elif col in ("OLS Dir", "ANN Dir", "Actual Dir"):
                c = "#44DD88" if "UP" in str(val) else ("#FF4466" if "DOWN" in str(val) else "#888")
                styles.append(f"color:{c};background-color:{bg}")
            elif col == "OLS %":
                styles.append(f"color:{ORANGE};background-color:{bg}")
            elif col == "ANN %":
                styles.append(f"color:#CC44FF;background-color:{bg}")
            else:
                styles.append(f"background-color:{bg};color:#E8E8F4")
        return styles
    return df.style.apply(row_style, axis=1)

st.dataframe(style_hist(hist_df), use_container_width=True)

ols_hits = sum(1 for r in hist_rows if r["OLS ✓"] == "✓")
ann_hits = sum(1 for r in hist_rows if r["ANN ✓"] == "✓")
n_h      = len(hist_rows)
c1, c2   = st.columns(2)
c1.markdown(metric_html("OLS Hit Rate (last 10d)", f"{ols_hits}/{n_h}", f"{ols_hits/n_h*100:.0f}%"), unsafe_allow_html=True)
c2.markdown(metric_html("ANN Hit Rate (last 10d)", f"{ann_hits}/{n_h}", f"{ann_hits/n_h*100:.0f}%"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD CSVs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>💾 Download Results</h2>", unsafe_allow_html=True)

fitted_df = pd.DataFrame({
    "date":           dates_all,
    "actual_close":   close_all,
    "actual_logprice":y_all,
    "ols_fitted_log": y_fit_ols_full,
    "ann_fitted_log": y_fit_ann_full,
    "ols_fitted_idx": fit_ols_all,
    "ann_fitted_idx": fit_ann_all,
    "split":          (["train"] * n_train + ["test"] * (n_total - n_train)),
})

forecast_df = pd.DataFrame({
    "date":            fut_dates,
    "ols_forecast_log":y_fore_ols,
    "ann_forecast_log":y_fore_ann,
    "ols_forecast_idx":fore_ols_idx,
    "ann_forecast_idx":fore_ann_idx,
})

natal_df = pd.DataFrame([
    {"Point": angle_labels.get(k, k), "Longitude": round(v, 4),
     "Sign": SIGNS[int(v // 30)], "Degree": int(v % 30)}
    for k, v in natal_targets.items()
])

col_a, col_b, col_c = st.columns(3)
col_a.download_button(
    "📥 Fitted Values CSV",
    data=fitted_df.to_csv(index=False).encode(),
    file_name=f"{ticker.replace('=','_')}_natal_fitted.csv",
    mime="text/csv",
)
col_b.download_button(
    "📥 Forecast CSV",
    data=forecast_df.to_csv(index=False).encode(),
    file_name=f"{ticker.replace('=','_')}_natal_forecast.csv",
    mime="text/csv",
)
if len(asp_table):
    col_c.download_button(
        "📥 Active Aspects CSV",
        data=asp_table.to_csv(index=False).encode(),
        file_name=f"{ticker.replace('=','_')}_natal_aspects.csv",
        mime="text/csv",
    )

st.markdown("---")
st.download_button(
    "📥 Natal Chart CSV",
    data=natal_df.to_csv(index=False).encode(),
    file_name=f"{ticker.replace('=','_')}_natal_chart.csv",
    mime="text/csv",
)

st.markdown(
    f"<hr class='gold'/>"
    f"<p style='text-align:center;color:#555;font-size:11px;letter-spacing:1px'>"
    f"NATAL PLANETARY REGRESSION · {stock_name} ({ticker}) · "
    f"Natal: {natal_date.date()} · "
    f"OLS R²={ols_full_r2:.4f} · ANN R²={ann_full_r2:.4f} · "
    f"Features: {n_features} · Apply≤{orb_apply}° Sep≤{orb_sep}° · "
    f"ANN {ann_layers}</p>",
    unsafe_allow_html=True,
)
