"""
Step 3 (step3_prior_calibration): the price-path prior -- simulated stocks with market-level noise and breaks
===================================================================================
Step 2 showed that a model's confidence is set by the answer noise in its training prior.
This step builds a prior with market-level noise: simulated price histories run through the
real pipeline (the same 8 inputs, answer and break flag as the real data), calibrated so
their statistics match real stocks.

DATA (all in the Data folder next to this step's folder)
  Data\\stocks_calibration   20 stocks used ONLY to calibrate the simulator (full history)
  Data\\stocks_test          the 20 test stocks; never used for calibration, only for the
                            final realism check
  The two folders must not share any stock (checked before anything is computed).

THE SIMULATED STOCK
  volatility level (log-normal around a calibrated median); asymmetric GARCH(1,1) clustering
  (falls raise volatility more than rises); Student-t daily shocks; breaks arriving at random,
  each lifting log-volatility and fading with a 98-day half-life; onset shocks, mostly
  downward, inside the variance recursion; a sell-off drift while a break is active;
  overnight gaps (25% of daily variance); day-to-day range noise; daily highs and lows from
  an 8-step Brownian bridge.

CALIBRATION, in four stages
  1. clustering and breaks: 180 combinations
  2. tails, onset-shock size and range noise: 60 combinations
  3. asymmetry (leverage x share of downward onset shocks x sell-off drift): 48 combinations
  4. the volatility level, one exact rescaling to the real median
  Distance: sum of squared differences on the scale-free statistics, each in units of the
  spread across the real stocks. 32 simulated series per combination, 40 for the final
  settings. Exploding combinations are scored as unrealistic. If the winner sits at a grid
  edge, a warning is printed. One extension round of the grid is allowed, and its result is
  accepted. That round has been used (EXTENSION_ROUND = True): the first run ended at the
  edge for breaks per year (5.0, highest), break size (0.15, lowest) and tail degrees of
  freedom (4, lowest), so the grid was widened in those three directions only. Any edge
  that remains is reported, not extended.

RUN MODES (edit MODE, press F5)
  "check"    folder checks, one batch of tasks, the table with the default (uncalibrated)
             settings, and a time estimate (a few minutes)
  "run"      the calibration and the table (about 20 minutes on CPU; resumes if interrupted)
  "analyse"  the table only, from the saved results

OUTPUT
  step3_prior_calibration_results.txt     progress lines and the table
  results\\pricepath_params.json          the calibrated settings and statistics (read by Step 4)
  results\\real_stats.json, results\\synth_cache.json   saved working results (for resuming)

FILES NEEDED
  scripts      this file only
  data         Data\\stocks_calibration (20 price files) and Data\\stocks_test (the 20 original
               test-stock price files, the same as in Step 1), in the folder above this one
  environment  any Python with numpy, pandas, scikit-learn and openpyxl; no GPU
"""
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

MODE = "run"          # "check" | "run" | "analyse"

_HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(_HERE)
DATA = os.path.join(PARENT, "Data")
CALIB_DIR = os.path.join(DATA, "stocks_calibration")
TEST_DIR = os.path.join(DATA, "stocks_test")
N_EXPECTED = {CALIB_DIR: 20, TEST_DIR: 20}
RESULTS = os.path.join(_HERE, "results")
OUTFILE = os.path.join(_HERE, "step3_prior_calibration_results.txt")
PARAMS_FILE = os.path.join(RESULTS, "pricepath_params.json")      # the calibrated settings (Step 4 reads this)
LABEL = "calibration"                                              # prefix of the progress lines

DEFAULT_PARAMS = dict(
    SIGMA_MED=0.32, SIGMA_LOGSD=0.35, DRIFT=0.0002,
    ALPHA=0.15, BETA=0.60, DOF=6.0,
    BREAK_RATE=3.0, BREAK_SIZE=0.3, BREAK_SIZE_SD=0.3, FADE_HALFLIFE=98.0,
    SHOCK_PROB=0.5, SHOCK_SIZE=2.0, SHOCK_DOWN=0.75, LEVERAGE=0.6, BREAK_DRIFT=0.05,
    OVERNIGHT_SHARE=0.25, RANGE_NOISE=0.8,
    INTRADAY_STEPS=8,
)
EXTENSION_ROUND = True    # the one allowed grid extension has been used (see the top of this file)
GRID = dict(GARCH=[(0.05, 0.93), (0.08, 0.90), (0.10, 0.80), (0.15, 0.60), (0.20, 0.45)],
            BREAK_RATE=[0.75, 1.5, 3.0, 5.0, 7.5, 10.0],                       # extended: 7.5, 10.0
            BREAK_SIZE=[0.05, 0.1, 0.15, 0.3, 0.6, 0.9],                        # extended: 0.05, 0.1
            DOF=[3.0, 4.0, 6.0, 10.0, 20.0],                                    # extended: 3.0
            SHOCK_SIZE=[0.0, 2.0, 3.5],
            RANGE_NOISE=[0.0, 0.2, 0.4, 0.8], LEVERAGE=[0.0, 0.3, 0.6, 1.0], SHOCK_DOWN=[0.5, 0.75, 1.0],
            BREAK_DRIFT=[0.0, 0.05, 0.1, 0.2])
NATURAL_LIMITS = dict(SHOCK_SIZE={0.0}, RANGE_NOISE={0.0}, LEVERAGE={0.0, 1.0}, SHOCK_DOWN={0.5, 1.0}, BREAK_DRIFT={0.0})   # not grid edges
STAGE1_BASE = dict(DOF=6.0, SHOCK_SIZE=2.0, RANGE_NOISE=0.8, LEVERAGE=0.6, SHOCK_DOWN=0.75, BREAK_DRIFT=0.05)   # middle values
SIM = dict(N_SERIES=32, N_FINAL=40, N_CHECK=6, N_DAYS=2500, SEED=12345)
TASK = dict(N_DAYS=900, TASKS_PER_SERIES=8, N_CTX_MIN=48, N_CTX_MAX=126, N_Q=8)
LR_START, LR_EVERY = 200, 3          # walk-forward logistic regression: from row 200, every 3rd block
MAX_ANN_VOL = 5.0                     # a simulated series above 500% annual volatility has "exploded"
MIN_DAYS = 600                        # a real series needs at least this many days to be used
KEYS = ["kurtosis", "acf_abs_1", "acf_abs_5", "acf_abs_20", "yes_rate", "y_acf_1", "onsets_126", "lr_auc",
        "lev_corr", "dist_y"]         # the calibrated statistics (all unaffected by the volatility level)

# ---- thesis pipeline constants ---------------------------------------------- #
GAP = 5
LABEL_HORIZON = 5
BASELINE_DAYS = 126
FORECAST_HORIZON = 10
TRAIL_VOL_FAST = 10
TRAIL_VOL_SLOW = 60
BREAK_Z = 2.0
BREAK_PAD = 5
EPS = 1e-12
FEATS = [f"Lag_{i}" for i in range(1, 6)] + ["Momentum_10", "Vol_20", "Dist_SMA_20"]


# ============================================================================= #
# PRICE FILES AND FOLDERS (the loader from Step 1)
# ============================================================================= #
def _read(path, **kw):
    return pd.read_excel(path, **kw) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, **kw)


def load_prices(path):
    """Reads a price file with Date, Open, High, Low, Close columns. Also accepts a lower-case
    'date', dates in the first (unnamed) column, and the three-header-row layout written by
    recent yfinance versions ('Price' / 'Ticker' / 'Date'). Anything else stops with the file
    name and the columns found."""
    df = _read(path)
    first = str(df.columns[0]).strip().lower()
    if first == "price" and len(df) > 2 and str(df.iloc[1, 0]).strip().lower() == "date":
        names_ = ["Date"] + [str(c).strip() for c in df.columns[1:]]          # yfinance 3-row header
        df = _read(path, skiprows=3, header=None)
        df.columns = names_[:df.shape[1]]
    rename = {c: c.strip().capitalize() for c in df.columns if isinstance(c, str)
              and c.strip().lower() in ("date", "open", "high", "low", "close", "volume")}
    df = df.rename(columns=rename)
    if "Date" not in df.columns and str(df.columns[0]).startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "Date"})                       # dates saved as the index
    missing = [c for c in ("Date", "High", "Low", "Close") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: cannot read the price file -- missing column(s) {missing}; "
                         f"columns found: {list(df.columns)[:10]}. Use the original price files.")
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ("Open", "High", "Low", "Close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def price_files(folder):
    return sorted(f for ext in ("*.xlsx", "*.xls", "*.csv") for f in glob.glob(os.path.join(folder, ext)))


def ticker(path):
    return os.path.splitext(os.path.basename(path))[0]


def check_folders():
    """Stop before any computing unless both folders exist, hold the expected number of
    stocks, hold no currencies, and share no stock."""
    problems, found = [], {}
    for folder, n in N_EXPECTED.items():
        if not os.path.isdir(folder):
            problems.append(f"folder not found: {folder}")
            continue
        t = [ticker(f) for f in price_files(folder)]
        found[folder] = t
        if len(t) != n:
            problems.append(f"{folder} holds {len(t)} price files, expected {n}")
        fx = [x for x in t if x.upper().endswith("=X")]
        if fx:
            problems.append(f"{folder} holds currency files: {', '.join(fx)}")
    if len(found) == 2:
        both = {x.upper() for x in found[CALIB_DIR]} & {x.upper() for x in found[TEST_DIR]}
        if both:
            problems.append(f"stocks_calibration and stocks_test share stocks: {', '.join(sorted(both))}")
    if problems:
        raise SystemExit("FOLDER PROBLEMS -- nothing was computed:\n  " + "\n  ".join(problems))
    return found


# ============================================================================= #
# THE REAL PIPELINE (as in Step 1)
# ============================================================================= #
def parkinson_vol(high, low):
    hl = np.log(high / low)
    return np.sqrt(np.clip((hl ** 2) / (4.0 * np.log(2.0)), 1e-12, None))


def causal_break(df):
    pvol = parkinson_vol(df["High"].values, df["Low"].values)
    s = pd.Series(pvol, index=df.index)
    fast = s.rolling(TRAIL_VOL_FAST).mean().shift(1)
    slow = s.rolling(TRAIL_VOL_SLOW).mean().shift(1)
    ratio = np.log((fast / slow).clip(1e-6))
    mu = ratio.rolling(TRAIL_VOL_SLOW).mean().shift(1)
    sd = ratio.rolling(TRAIL_VOL_SLOW).std().shift(1)
    z = (ratio - mu) / sd.replace(0, np.nan)
    onset = ((z > BREAK_Z) & (z.shift(1) <= BREAK_Z)).fillna(False).values
    flag = np.zeros(len(df), dtype=bool)
    for o in np.where(onset)[0]:
        flag[o:min(len(df), o + BREAK_PAD + 1)] = True
    return pd.Series(flag, index=df.index), pd.Series(onset, index=df.index)


def pipeline_frame(df):
    """The 8 inputs, the answer and the break flag, exactly as in the real pipeline."""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    for i in range(1, 6):
        df[f"Lag_{i}"] = df["Return"].shift(i)
    df["Momentum_10"] = df["Close"].pct_change(10).shift(1)
    df["Vol_20"] = df["Return"].rolling(20).std().shift(1)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Dist_SMA_20"] = ((df["Close"] - df["SMA_20"]) / df["SMA_20"]).shift(1)
    df["pvol"] = parkinson_vol(df["High"].values, df["Low"].values)
    df["Future_5d_Vol"] = df["pvol"].rolling(LABEL_HORIZON).mean().shift(-LABEL_HORIZON)
    df["Usual_Vol"] = df["pvol"].rolling(BASELINE_DAYS).median()
    df["Target_Vol_Spike"] = (df["Future_5d_Vol"] > df["Usual_Vol"]).astype(int)
    df["near_transition"], df["onset"] = causal_break(df)
    return df.dropna().reset_index(drop=True)


# ============================================================================= #
# THE SIMULATED STOCK
# ============================================================================= #
def simulate_series(rng, n_days, p):
    """Daily Open/High/Low/Close for one synthetic stock (see the top of this file)."""
    sigma_ann = p["SIGMA_MED"] * np.exp(p["SIGMA_LOGSD"] * rng.standard_normal())
    base_var = sigma_ann ** 2 / 252.0
    decay = np.log(2.0) / p["FADE_HALFLIFE"]
    # breaks: log-volatility lifts that fade
    n_br = rng.poisson(p["BREAK_RATE"] * n_days / 252.0)
    onsets = np.sort(rng.integers(0, n_days, n_br))
    sizes = np.abs(p["BREAK_SIZE"] + p["BREAK_SIZE_SD"] * rng.standard_normal(n_br))
    t = np.arange(n_days)
    logm = np.zeros(n_days)
    for o, s in zip(onsets, sizes):
        k = t >= o
        logm[k] += s * np.exp(-decay * (t[k] - o))
    target_var = base_var * np.exp(2.0 * logm)
    # asymmetric GARCH(1,1) around the moving target level, Student-t shocks with unit variance
    a, b, nu, lev = p["ALPHA"], p["BETA"], p["DOF"], p["LEVERAGE"]
    z = rng.standard_t(nu, n_days) * np.sqrt((nu - 2.0) / nu)
    # onset shocks (in units of that day's volatility), mostly downward, inside the recursion
    shock = np.zeros(n_days)
    has = rng.random(n_br) < p["SHOCK_PROB"]
    sign = np.where(rng.random(n_br) < p["SHOCK_DOWN"], -1.0, 1.0)
    for o, hs, sg in zip(onsets, has, sign):
        if hs:
            shock[o] += sg * p["SHOCK_SIZE"]
    h = np.empty(n_days); r = np.empty(n_days)
    h[0] = target_var[0]
    for i in range(n_days):
        if i > 0:
            react = a * (1.0 + lev) if r[i - 1] < 0 else a * (1.0 - lev)        # leverage effect, mean reaction a
            h[i] = (1 - a - b) * target_var[i] + react * r[i - 1] ** 2 + b * h[i - 1]
        r[i] = np.sqrt(h[i]) * (z[i] + shock[i] - p["BREAK_DRIFT"] * logm[i])   # sell-off drift while a break is active
    r = r + p["DRIFT"]
    # overnight gap (outside the trading-day range), then the trading day as a Brownian bridge
    g = p["OVERNIGHT_SHARE"]
    r_night = np.sqrt(g * h) * rng.standard_normal(n_days)
    r_day = r - r_night
    rn = p["RANGE_NOISE"]
    mult = np.exp(rn * rng.standard_normal(n_days) - 0.5 * rn ** 2)          # mean 1
    K = int(p["INTRADAY_STEPS"])
    inc = rng.standard_normal((n_days, K)) * np.sqrt((1.0 - g) * h * mult / K)[:, None]
    path = np.cumsum(inc, 1)
    path = path - (np.arange(1, K + 1)[None, :] / K) * (path[:, -1] - r_day)[:, None]
    hi = np.maximum(0.0, path.max(1)); lo = np.minimum(0.0, path.min(1))
    close = 100.0 * np.exp(np.cumsum(r))
    prev = np.r_[100.0, close[:-1]]
    opn = prev * np.exp(r_night)
    return pd.DataFrame(dict(Date=pd.bdate_range("2000-01-03", periods=n_days), Open=opn,
                             High=opn * np.exp(hi), Low=opn * np.exp(lo), Close=close))


def frame_ok(df, dfc):
    """False if a simulated series exploded (compounding breaks): non-finite inputs or
    absurd volatility."""
    r = df["Close"].pct_change().values[1:]
    return (len(dfc) > 0 and np.isfinite(dfc[FEATS].values).all() and np.isfinite(dfc["pvol"].values).all()
            and np.isfinite(r).all() and r.std() * np.sqrt(252) <= MAX_ANN_VOL)


# ============================================================================= #
# TASKS (the exact real-data format; used by Step 4)
# ============================================================================= #
def tasks_from_frame(dfc, rng, n_tasks, n_ctx, n_q):
    X = dfc[FEATS].values.astype(np.float64)
    y = dfc["Target_Vol_Spike"].values.astype(np.float32)
    lo, hi = n_ctx + GAP, len(dfc) - n_q
    out = []
    if hi <= lo:
        return out
    for ts in rng.integers(lo, hi, n_tasks):
        tre = ts - GAP; trs = tre - n_ctx
        Xc, Xq = X[trs:tre], X[ts:ts + n_q]
        mu = Xc.mean(0); sd = Xc.std(0); sd[sd == 0] = 1.0           # StandardScaler (population sd)
        Xc = np.clip((Xc - mu) / sd, -5.0, 5.0); Xq = np.clip((Xq - mu) / sd, -5.0, 5.0)
        tc = (np.arange(n_ctx) / float(max(1, n_ctx - 1))).astype(np.float32)
        out.append((np.column_stack([tc, Xc]).astype(np.float32), y[trs:tre].copy(),
                    np.column_stack([np.ones(n_q, np.float32), Xq]).astype(np.float32), y[ts:ts + n_q].copy()))
    return out


def task_batch(rng, bs, p, n_ctx=None, n_q=None):
    """bs tasks as (context, context answers, questions, question answers). One context
    length per batch (drawn 48-126 unless given)."""
    n_q = n_q or TASK["N_Q"]
    n_ctx = n_ctx or int(rng.integers(TASK["N_CTX_MIN"], TASK["N_CTX_MAX"] + 1))
    tasks = []
    while len(tasks) < bs:
        df = simulate_series(rng, TASK["N_DAYS"], p)
        dfc = pipeline_frame(df)
        if not frame_ok(df, dfc):
            continue                                   # an exploded series: draw another
        tasks += tasks_from_frame(dfc, rng, min(TASK["TASKS_PER_SERIES"], bs - len(tasks)), n_ctx, n_q)
    return tasks


def load_params(path=None):
    """The chosen calibrated settings (results\\pricepath_params.json); defaults if absent."""
    path = path or PARAMS_FILE
    if os.path.exists(path):
        with open(path) as f:
            return dict(DEFAULT_PARAMS, **json.load(f)["params"])
    return dict(DEFAULT_PARAMS)


# ============================================================================= #
# STATISTICS
# ============================================================================= #
def acf(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) <= lag + 2:
        return np.nan
    a, b = x[:-lag] - x[:-lag].mean(), x[lag:] - x[lag:].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def lev_corr(r):
    """Correlation of today's return with the mean |return| over the next 5 days."""
    r = np.asarray(r, float)
    fut = pd.Series(np.abs(r)).rolling(5).mean().shift(-5).values
    return corr(r, fut)


def blocks_from(n_rows, window, start):
    for i in range(start, n_rows, FORECAST_HORIZON):
        ts, te = i, min(i + FORECAST_HORIZON, n_rows)
        tre = ts - GAP; trs = max(0, tre - window)
        if tre > trs and te > ts:
            yield trs, tre, ts, te


def lr_auc(dfc):
    """Walk-forward logistic regression with time on the 8 inputs (126-day window)."""
    X = dfc[FEATS].values.astype(float); y = dfc["Target_Vol_Spike"].values.astype(int)
    ps, ys = [], []
    for bi, (trs, tre, ts, te) in enumerate(blocks_from(len(dfc), 126, LR_START)):
        if bi % LR_EVERY:
            continue
        ytr = y[trs:tre]
        if len(np.unique(ytr)) < 2:
            continue
        mu = X[trs:tre].mean(0); sd = X[trs:tre].std(0); sd[sd == 0] = 1.0
        t_ctx = (np.arange(tre - trs) / max(1, tre - trs - 1))
        A = np.column_stack([t_ctx, np.clip((X[trs:tre] - mu) / sd, -5, 5)])
        B = np.column_stack([np.ones(te - ts), np.clip((X[ts:te] - mu) / sd, -5, 5)])
        ps.append(LogisticRegression(max_iter=300).fit(A, ytr).predict_proba(B)[:, 1]); ys.append(y[ts:te])
    if not ps:
        return np.nan
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return float(roc_auc_score(ys, ps)) if len(np.unique(ys)) == 2 else np.nan


def series_stats(df, synthetic=False):
    dfc = pipeline_frame(df)
    if synthetic and not frame_ok(df, dfc):
        return None                                    # exploded series
    r = df["Close"].pct_change().values[1:]
    r = r[np.isfinite(r)]
    y = dfc["Target_Vol_Spike"].values
    return dict(ann_vol=float(r.std() * np.sqrt(252)),
                kurtosis=float(pd.Series(r).kurt()),
                acf_abs_1=acf(np.abs(r), 1), acf_abs_5=acf(np.abs(r), 5), acf_abs_20=acf(np.abs(r), 20),
                yes_rate=float(y.mean()), y_acf_1=acf(y, 1),
                onsets_126=float(dfc["onset"].sum() / len(dfc) * 126),
                flag_share=float(dfc["near_transition"].mean()),
                lr_auc=lr_auc(dfc),
                lev_corr=lev_corr(r),
                dist_y=corr(dfc["Dist_SMA_20"].values, dfc["Target_Vol_Spike"].values))


def summarise(rows):
    df = pd.DataFrame(rows)
    return {k: dict(median=float(df[k].median()), sd=float(df[k].std())) for k in df.columns}


def real_stats(files, end=None):
    """Statistics over real series (data before `end` if given). Returns the summary and the
    stocks skipped as too short."""
    rows, short = [], []
    for f in files:
        df = load_prices(f)
        if end is not None:
            df = df[df["Date"] < pd.Timestamp(end)].reset_index(drop=True)
        if len(df) < MIN_DAYS:
            short.append(ticker(f))
            continue
        rows.append(series_stats(df))
    return dict(summary=summarise(rows), n=len(rows), short=short)


def synth_stats(p, n_series, seed):
    """Statistics over simulated series; if any series exploded, the combination is
    unrealistic and is returned as {"_exploded": count}."""
    rng = np.random.default_rng(seed)
    rows = [series_stats(simulate_series(rng, SIM["N_DAYS"], p), synthetic=True) for _ in range(n_series)]
    bad = sum(r is None for r in rows)
    return {"_exploded": bad} if bad else summarise(rows)


def distance(syn, real):
    if "_exploded" in syn:
        return float("inf")
    return float(sum(((syn[k]["median"] - real[k]["median"]) / max(real[k]["sd"], 1e-6)) ** 2 for k in KEYS))


class Cache:
    """Saved simulation results, so an interrupted run resumes where it stopped and no
    combination is simulated twice."""
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def get(self, p, n_series, seed):
        key = json.dumps([sorted((k, float(v)) for k, v in p.items()), n_series, seed])
        if key not in self.d:
            self.d[key] = synth_stats(p, n_series, seed)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.d, f)
            os.replace(tmp, self.path)
        return self.d[key]


# ============================================================================= #
# CALIBRATION
# ============================================================================= #
def calibrate(real, label, cache):
    p = dict(DEFAULT_PARAMS)
    allres = []

    def stage(k, title, combos):
        t0 = time.time()
        print(f"{label:<12}| stage {k}/4 {title}, {len(combos)} combinations ...", end="", flush=True)
        for q in combos:
            allres.append((distance(cache.get(q, SIM["N_SERIES"], SIM["SEED"]), real), q))
        allres.sort(key=lambda x: x[0])
        print(f" done ({(time.time() - t0) / 60:.0f} min)", flush=True)
        return allres[0][1]

    s1 = stage(1, "clustering and breaks",
               [dict(p, **STAGE1_BASE, ALPHA=a, BETA=b, BREAK_RATE=br, BREAK_SIZE=bs)
                for a, b in GRID["GARCH"] for br in GRID["BREAK_RATE"] for bs in GRID["BREAK_SIZE"]])
    s2 = stage(2, "tails, onset shocks, range noise",
               [dict(s1, DOF=nu, SHOCK_SIZE=sh, RANGE_NOISE=rn)
                for nu in GRID["DOF"] for sh in GRID["SHOCK_SIZE"] for rn in GRID["RANGE_NOISE"]])
    best = stage(3, "asymmetry",
                 [dict(s2, LEVERAGE=lv, SHOCK_DOWN=sd, BREAK_DRIFT=bd)
                  for lv in GRID["LEVERAGE"] for sd in GRID["SHOCK_DOWN"] for bd in GRID["BREAK_DRIFT"]])
    d = allres[0][0]
    if not np.isfinite(d):
        raise SystemExit(f"{label}: every combination exploded -- the grid needs changing")
    edges = []
    for key, name, grid in [("ALPHA", "alpha", [g[0] for g in GRID["GARCH"]]), ("BREAK_RATE", "breaks per year", GRID["BREAK_RATE"]),
                            ("BREAK_SIZE", "break size", GRID["BREAK_SIZE"]), ("DOF", "tail degrees of freedom", GRID["DOF"]),
                            ("SHOCK_SIZE", "onset shock size", GRID["SHOCK_SIZE"]), ("RANGE_NOISE", "range noise", GRID["RANGE_NOISE"]),
                            ("LEVERAGE", "leverage", GRID["LEVERAGE"]), ("SHOCK_DOWN", "share of downward shocks", GRID["SHOCK_DOWN"]),
                            ("BREAK_DRIFT", "sell-off drift", GRID["BREAK_DRIFT"])]:
        val = best[key]
        if key in NATURAL_LIMITS and val in NATURAL_LIMITS[key]:
            continue                                   # cannot go further: not an edge
        if val in (min(grid), max(grid)):
            edges.append(f"{name} = {val} ({'lowest' if val == min(grid) else 'highest'} value tried)")
    t0 = time.time()
    print(f"{label:<12}| stage 4/4 volatility level ...", end="", flush=True)
    s40 = cache.get(best, SIM["N_FINAL"], SIM["SEED"])
    best = dict(best, SIGMA_MED=best["SIGMA_MED"] * real["ann_vol"]["median"] / s40["ann_vol"]["median"])
    final = cache.get(best, SIM["N_FINAL"], SIM["SEED"])
    if "_exploded" in final:
        raise SystemExit(f"{label}: the final settings exploded on {final['_exploded']} series -- the grid needs changing")
    print(f" done ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"{label:<12}| settings: alpha {best['ALPHA']}, beta {best['BETA']}, breaks/yr {best['BREAK_RATE']}, "
          f"break size {best['BREAK_SIZE']}, tail dof {best['DOF']}, onset shock {best['SHOCK_SIZE']}, "
          f"range noise {best['RANGE_NOISE']}, leverage {best['LEVERAGE']}, downward shocks {best['SHOCK_DOWN']}, "
          f"sell-off drift {best['BREAK_DRIFT']}, volatility median {best['SIGMA_MED']:.3f}")
    if edges:
        print(f"{label:<12}| " + ("edge of the grid after the extension round (accepted, no further extension): "
                                   if EXTENSION_ROUND else "WARNING, edge of the grid: ") + "; ".join(edges)
              + ("" if EXTENSION_ROUND else " (one extension round is allowed: widen GRID in that direction and run again)"))
    return dict(params=best, synthetic=final, distance_calibration=d, edge_warnings=edges,
                grid=[dict(distance=r[0], **{k: r[1][k] for k in ("ALPHA", "BETA", "BREAK_RATE", "BREAK_SIZE", "DOF",
                                                                  "SHOCK_SIZE", "RANGE_NOISE", "LEVERAGE", "SHOCK_DOWN",
                                                                  "BREAK_DRIFT")}) for r in allres])


# ============================================================================= #
# TABLE
# ============================================================================= #
ROWS = [("ann_vol", "Annual volatility"), ("kurtosis", "Tail heaviness"),
        ("acf_abs_1", "Volatility clustering, 1 day"), ("acf_abs_5", "Volatility clustering, 5 days"),
        ("acf_abs_20", "Volatility clustering, 20 days"), ("lev_corr", "Leverage effect"),
        ("yes_rate", "Share of high-volatility days"), ("y_acf_1", "Answer persistence"),
        ("onsets_126", "Break onsets per 126 days"), ("flag_share", "Share of days flagged as a break"),
        ("dist_y", "Distance from 20-day average vs answer"), ("lr_auc", "Logistic regression AUC")]

LEGEND = [
    "Annual volatility: standard deviation of daily returns, annualised.",
    "Tail heaviness: excess kurtosis of daily returns (0 = normal distribution).",
    "Volatility clustering: correlation of the size of today's return with the size of the return 1, 5 or 20 days later.",
    "Leverage effect: correlation of today's return with the average size of returns over the next 5 days "
    "(negative = falls raise volatility).",
    "Share of high-volatility days: share of days whose answer is \"yes\" (next 5 days' volatility above its 126-day median).",
    "Answer persistence: correlation of each day's answer with the next day's.",
    "Break onsets per 126 days: how often the causal break flag switches on.",
    "Share of days flagged as a break: share of days inside a flagged break.",
    "Distance from 20-day average vs answer: correlation of the price's distance from its 20-day average with the answer.",
    "Logistic regression AUC: how predictable the answer is from the 8 inputs.",
    "Distance to test stocks: overall gap from the test stocks across the calibrated statistics (lower = closer);",
    "  for the calibration stocks, how close real stocks other than the test stocks get.",
]


def table(calib, test, sim):
    w0, w1, w2 = 41, 14, 7
    lines = ["TABLE 4. The price-path prior against real stocks (medians over series)", "",
             f"{'':<{w0}}{'Real stocks':<{w1 + w2}}Price-path",
             f"{'Statistic':<{w0}}{'calibration':<{w1}}{'test':<{w2}}prior"]
    for k, name in ROWS:
        lines.append(f"{name:<{w0}}{calib[k]['median']:<{w1}.3f}{test[k]['median']:<{w2}.3f}{sim[k]['median']:.3f}")
    lines.append(f"{'Distance to test stocks':<{w0}}{distance(calib, test):<{w1}.2f}{'-':<{w2}}{distance(sim, test):.2f}")
    return lines + [""] + LEGEND


# ============================================================================= #
# MAIN
# ============================================================================= #
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for st in self.s: st.write(x)
    def flush(self):
        for st in self.s: st.flush()


def _main():
    os.makedirs(RESULTS, exist_ok=True)
    real_file = os.path.join(RESULTS, "real_stats.json")
    if MODE == "analyse":
        print("step3_prior_calibration | mode analyse")
        missing = [f for f in (real_file, PARAMS_FILE) if not os.path.exists(f)]
        if missing:
            raise SystemExit("not found (run MODE = \"run\" first): " + ", ".join(missing))
        real = json.load(open(real_file))
        print("\n" + "\n".join(table(real["calibration"]["summary"], real["test"]["summary"],
                                     json.load(open(PARAMS_FILE))["synthetic"])))
        return

    found = check_folders()
    print(f"step3_prior_calibration | mode {MODE} | calibration stocks: {len(found[CALIB_DIR])} | "
          f"test stocks: {len(found[TEST_DIR])} | no overlap")
    if MODE == "run" and os.path.exists(real_file):
        real = json.load(open(real_file))
        print("real statistics: loaded from results\\real_stats.json")
    else:
        t0 = time.time()
        real = dict(calibration=real_stats(price_files(CALIB_DIR)), test=real_stats(price_files(TEST_DIR)))
        sh = real["calibration"]["short"] + real["test"]["short"]
        print(f"real statistics: calibration stocks ({real['calibration']['n']} series), test stocks "
              f"({real['test']['n']} series)" + (f"; too short, left out: {', '.join(sh)}" if sh else "")
              + f" ({time.time() - t0:.0f}s)")
        if MODE == "run":
            with open(real_file, "w") as f:
                json.dump(real, f)
    calib, test = real["calibration"]["summary"], real["test"]["summary"]

    if MODE == "check":
        rng = np.random.default_rng(0)
        c, yc, q, yq = task_batch(rng, 16, DEFAULT_PARAMS)[0]
        print(f"tasks: context {c.shape[0]} rows x {c.shape[1]} columns (time + 8 inputs), {q.shape[0]} questions; "
              f"inputs clipped to [{c[:, 1:].min():.1f}, {c[:, 1:].max():.1f}]")
        t0 = time.time()
        s = synth_stats(DEFAULT_PARAMS, SIM["N_CHECK"], SIM["SEED"])
        per = (time.time() - t0) / SIM["N_CHECK"]
        n_series = (len(GRID["GARCH"]) * len(GRID["BREAK_RATE"]) * len(GRID["BREAK_SIZE"])
                    + len(GRID["DOF"]) * len(GRID["SHOCK_SIZE"]) * len(GRID["RANGE_NOISE"])
                    + len(GRID["LEVERAGE"]) * len(GRID["SHOCK_DOWN"]) * len(GRID["BREAK_DRIFT"])) * SIM["N_SERIES"] \
            + 2 * SIM["N_FINAL"]
        print(f"simulation: {per:.2f}s per simulated stock -> full run about {per * n_series / 60:.0f} min")
        if "_exploded" in s:
            raise SystemExit("the default settings exploded -- something is wrong with the simulator")
        print("\n(check mode: default settings, not calibrated, 6 simulated stocks -- the numbers mean nothing yet)\n")
        print("\n".join(table(calib, test, s)))
        return

    grid_now = json.loads(json.dumps(GRID))
    if os.path.exists(PARAMS_FILE) and json.load(open(PARAMS_FILE)).get("grid_settings") == grid_now:
        res = json.load(open(PARAMS_FILE))
        print(f"{LABEL:<12}| already calibrated with this grid (results\\pricepath_params.json)")
    else:
        if os.path.exists(PARAMS_FILE):
            print(f"{LABEL:<12}| the grid has changed: recalibrating (saved simulations are reused)")
        res = calibrate(calib, LABEL, Cache(os.path.join(RESULTS, "synth_cache.json")))
        with open(PARAMS_FILE, "w") as f:
            json.dump(dict(res, calibration_data="Data\\stocks_calibration, full history", grid_settings=grid_now,
                           extension_round=EXTENSION_ROUND), f, indent=1)
        print(f"{LABEL:<12}| saved: results\\pricepath_params.json (read by Step 4)")
    print("\n" + "\n".join(table(calib, test, res["synthetic"])))


def main():
    with open(OUTFILE, "w", encoding="utf-8") as out:
        sys.stdout = _Tee(sys.__stdout__, out)
        try:
            _main()
        finally:
            sys.stdout = sys.__stdout__
    print(f"\nresults written to {OUTFILE}")


if __name__ == "__main__":
    main()