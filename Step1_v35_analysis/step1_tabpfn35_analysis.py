"""
Step 1: TabPFN v2, TabPFN-3 and TabPFN-3.5 on the test stocks and currencies
============================================================================
TabPFN-3.5 (tabpfn package 9.x) is run here, with the time column, through its official
classifier. TabPFN v2 and TabPFN-3 were run in earlier versions of the project; their saved
predictions are read from previous_versions_results. All three are scored on the same days.

FOLDERS (the parent folder can have any name)
  <parent>\\Data\\stocks_test\\                         20 test stock price files
  <parent>\\Data\\currencies\\                          7 currency price files (original, unrepaired)
  <parent>\\Step1_v35_analysis\\previous_versions_results\\tabpfn_v2\\{stocks_test, currencies}\\
  <parent>\\Step1_v35_analysis\\previous_versions_results\\tabpfn_v3\\{stocks_test, currencies}\\
Written by this file:
  <parent>\\Step1_v35_analysis\\results\\{stocks_test, currencies}\\    TabPFN-3.5 predictions
  <parent>\\Step1_v35_analysis\\Step1_v35_analysis_results.txt         the output, including the tables

RUN MODES (edit MODE, press F5)
  "check"    1 stock + 1 currency: folder checks, join check, time estimate
  "run"      everything (resumes from the predictions already in results\\)
  "analyse"  tables only, from the saved predictions

ENVIRONMENT: tabpfn 9.x with PyTorch for CUDA 12.8, pandas, scipy, scikit-learn, openpyxl.
If the TabPFN-3.5 download needs a token, set it as a Windows environment variable
"""

import glob
import os
os.environ["TABPFN_TOKEN"] = "" ## Generate and insert TabPFN token here!!!!
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODE = "run"          # "check" | "run" | "analyse"

_HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(_HERE)
DATA = os.path.join(PARENT, "Data")
PREV = os.path.join(_HERE, "previous_versions_results")
RESULTS = os.path.join(_HERE, "results")
OUTFILE = os.path.join(_HERE, "Step1_v35_analysis_results.txt")
DATASETS = {"stocks_test": 20, "currencies": 7}          # folder name -> expected number of series

DEVICE = "cuda"
X_CLIP = 5.0
WINDOW = 126
MIN_N = 30
CAL_WINDOW, MIN_CAL, LABEL_DELAY = 250, 100, 6
MODELS = [("TabPFN v2", "p_base_t"), ("TabPFN-3", "p_v3_t"), ("TabPFN-3.5", "p_v35_t")]

# ---- thesis pipeline constants ----------------------------------------------- #
GAP = 5
LABEL_HORIZON = 5
BASELINE_DAYS = 126
FORECAST_HORIZON = 10
TRAIL_VOL_FAST = 10
TRAIL_VOL_SLOW = 60
BREAK_Z = 2.0
BREAK_PAD = 5
MIN_BREAK_OBS = 60
EPS = 1e-12
THESIS_WINDOW = 1260
FIRST_TEST_ROW = THESIS_WINDOW + 26

assert GAP >= LABEL_HORIZON, "gap below the label horizon would leak future labels into training"


# ============================================================================= #
# PIPELINE (unchanged from the earlier steps)
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


def load_prices(path):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def prepare_ticker(path):
    ticker = os.path.splitext(os.path.basename(path))[0]
    df = load_prices(path)
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
    dfc = df.dropna().reset_index(drop=True)
    feats = [f"Lag_{i}" for i in range(1, 6)] + ["Momentum_10", "Vol_20", "Dist_SMA_20"]
    if dfc["near_transition"].sum() < MIN_BREAK_OBS or len(dfc) < THESIS_WINDOW + 200:
        return None
    return ticker, dfc, feats


def walk_forward_blocks(n_rows, window):
    out = []
    for i in range(FIRST_TEST_ROW, n_rows, FORECAST_HORIZON):
        ts, te = i, min(i + FORECAST_HORIZON, n_rows)
        tre = ts - GAP
        trs = max(0, tre - window)
        assert (tre - 1) + LABEL_HORIZON < ts
        if tre > trs and te > ts:
            out.append((trs, tre, ts, te))
    return out


# ============================================================================= #
# FOLDERS
# ============================================================================= #
def names(folder, exts):
    if not os.path.isdir(folder):
        return None
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(exts))


def check_folders(datasets):
    """Stop before any computing unless every folder exists and holds the same series."""
    problems, plan = [], {}
    for name in datasets:
        n_exp = DATASETS[name]
        folders = {"prices": os.path.join(DATA, name),
                   "TabPFN v2": os.path.join(PREV, "tabpfn_v2", name),
                   "TabPFN-3": os.path.join(PREV, "tabpfn_v3", name)}
        sets = {}
        for label, f in folders.items():
            s = names(f, (".xlsx", ".xls", ".csv") if label == "prices" else (".csv",))
            if s is None:
                problems.append(f"{name}: folder not found: {f}")
            elif len(s) != n_exp:
                problems.append(f"{name}: {f} holds {len(s)} files, expected {n_exp}")
            sets[label] = s
        if all(sets.get(k) for k in folders):
            if not (set(sets["prices"]) == set(sets["TabPFN v2"]) == set(sets["TabPFN-3"])):
                problems.append(f"{name}: the three folders hold different series: "
                                + "; ".join(f"{k}: {', '.join(v)}" for k, v in sets.items()))
            fx = [t for t in sets["prices"] if t.endswith("=X")]
            if name == "stocks_test" and fx:
                problems.append(f"stocks_test holds currency files: {', '.join(fx)}")
            if name == "currencies" and len(fx) != len(sets["prices"]):
                problems.append("currencies holds non-currency files")
        plan[name] = dict(folders=folders, series=sets.get("prices") or [])
    calib = names(os.path.join(DATA, "stocks_calibration"), (".xlsx", ".xls", ".csv"))
    if calib and plan.get("stocks_test"):
        both = set(calib) & set(plan["stocks_test"]["series"])
        if both:
            problems.append(f"stocks_calibration and stocks_test share series: {', '.join(sorted(both))}")
    if problems:
        raise SystemExit("FOLDER PROBLEMS -- nothing was computed:\n  " + "\n  ".join(problems))
    return plan


def price_file(folder, ticker):
    for ext in ("xlsx", "csv", "xls"):
        p = os.path.join(folder, f"{ticker}.{ext}")
        if os.path.exists(p):
            return p
    return None


def load_folder(folder, cols=None):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, "*.csv")))], ignore_index=True)
    return df if cols is None else df[cols]


# ============================================================================= #
# TABPFN-3.5
# ============================================================================= #
def tabpfn_version():
    try:
        import tabpfn
        return getattr(tabpfn, "__version__", "unknown")
    except ImportError:
        return None


def make_model():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device=DEVICE)


def run_series(name, path, saved, model, timing):
    prep = prepare_ticker(path)
    if prep is None:
        print(f"  {os.path.basename(path)}: skipped (too few breaks or too short)"); return
    ticker, dfc, feats = prep
    os.makedirs(os.path.join(RESULTS, name), exist_ok=True)
    op = os.path.join(RESULTS, name, f"{ticker}.csv")
    if os.path.exists(op) and "p_v35_t" in pd.read_csv(op, nrows=1).columns:
        print(f"  {ticker}: already computed"); return
    X = dfc[feats].values.astype(np.float64)
    y = dfc["Target_Vol_Spike"].values.astype(int)
    blocks = walk_forward_blocks(len(dfc), WINDOW)
    rows, dates, targets, brk, probs = [], [], [], [], []
    t0 = time.time(); every = max(1, len(blocks) // 10)
    print(f"  {ticker}: {len(blocks)} blocks ...", flush=True)
    for bi, (trs, tre, ts, te) in enumerate(blocks):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[trs:tre]); Xte = sc.transform(X[ts:te])
        ytr = y[trs:tre]
        t_ctx = (np.arange(trs, tre) - trs) / float(max(1, tre - trs - 1))
        Xtr_t = np.column_stack([t_ctx, np.clip(Xtr, -X_CLIP, X_CLIP)])
        Xte_t = np.column_stack([np.ones(te - ts), np.clip(Xte, -X_CLIP, X_CLIP)])
        if len(np.unique(ytr)) < 2:
            p = np.full(te - ts, (ytr.sum() + 1.0) / (len(ytr) + 2.0))
        else:
            t1 = time.time()
            model.fit(Xtr_t, ytr); p = model.predict_proba(Xte_t)[:, 1]
            timing.append(time.time() - t1)
        rows.extend(range(ts, te)); dates.extend(dfc["Date"].iloc[ts:te].astype(str))
        targets.extend(y[ts:te]); brk.extend(dfc["near_transition"].values[ts:te]); probs.extend(p)
        if (bi + 1) % every == 0 or bi + 1 == len(blocks):
            el = time.time() - t0
            print(f"    {ticker} {bi + 1}/{len(blocks)} blocks  {el:.0f}s  ~{el / (bi + 1) * (len(blocks) - bi - 1):.0f}s left",
                  flush=True)
    out = pd.DataFrame(dict(ticker=ticker, row=rows, date=dates, target=targets, is_break=brk, p_v35_t=probs))
    check_join(out, saved, ticker)
    out.to_csv(op, index=False)


def check_join(new, saved, ticker):
    s = saved[saved["ticker"] == ticker].sort_values("row").reset_index(drop=True)
    n = new.sort_values("row").reset_index(drop=True)
    ok = (len(s) == len(n) and np.array_equal(s["row"].values, n["row"].values)
          and (s["date"].astype(str).str[:10].values == n["date"].astype(str).str[:10].values).all()
          and np.array_equal(s["target"].values.astype(int), n["target"].values.astype(int))
          and np.array_equal(s["is_break"].values.astype(bool), n["is_break"].values.astype(bool)))
    if not ok:
        raise RuntimeError(f"{ticker}: days, answers or break flags differ from the TabPFN v2 results -- "
                           f"the price data or pipeline is not identical, so the comparison is refused")
    print(f"    {ticker}: join check passed", flush=True)


# ============================================================================= #
# TABLES
# ============================================================================= #
def skill(y, p):
    acc = ((p >= 0.5) == (y > 0.5)).mean()
    return acc - max(y.mean(), 1 - y.mean())


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def temperatures(d, col):
    """Walk-forward temperature scaling: at each 10-day block, fit one temperature on the last
    250 days whose answers are already known (answers known 6 days later, at least 100)."""
    rows, blocks = d["row"].values, d["block"].values
    y, p = d["target"].values.astype(float), d[col].values.astype(float)
    out = []
    for b in np.unique(blocks):
        ts = rows[blocks == b].min()
        kn = np.where((rows <= ts - LABEL_DELAY) & (rows > ts - LABEL_DELAY - CAL_WINDOW))[0]
        if len(kn) < MIN_CAL or len(np.unique(y[kn])) < 2:
            continue
        z, yk = logit(p[kn]), y[kn]

        def nll(lt):
            q = np.clip(1 / (1 + np.exp(-z / np.exp(lt))), 1e-6, 1 - 1e-6)
            return -np.mean(yk * np.log(q) + (1 - yk) * np.log(1 - q))
        out.append(float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)))
    return out


def table(name, df, n_series):
    res = {}
    for label, col in MODELS:
        vals = {k: [] for k in ("auc_n", "auc_b", "sk_n", "sk_b")}
        temps = []
        for _, d in df.groupby("ticker"):
            d = d.sort_values("row").reset_index(drop=True)
            y, p, b = d["target"].values.astype(float), d[col].values.astype(float), d["is_break"].values.astype(bool)
            for cell, m in (("n", ~b), ("b", b)):
                if m.sum() >= MIN_N:
                    vals[f"auc_{cell}"].append(auc(y[m], p[m])); vals[f"sk_{cell}"].append(skill(y[m], p[m]))
            temps += temperatures(d, col)
        res[label] = {k: np.nanmean(v) for k, v in vals.items()}
        res[label]["temp"] = float(np.median(temps)) if temps else np.nan
    title = {"stocks_test": f"{n_series} test stocks", "currencies": f"{n_series} currencies"}[name]
    w = 14
    lines = [f"{title} ({len(df):,} test days)",
             f"{'Model':<{w}}{'AUC':<17}{'Skill (majority class invariant)':<35}{'Temperature'}",
             f"{'':<{w}}{'normal':<9}{'break':<8}{'normal':<10}{'break':<25}{'(overconfidence metric)'}"]
    for label, _ in MODELS:
        r = res[label]
        lines.append(f"{label:<{w}}{r['auc_n']:<9.3f}{r['auc_b']:<8.3f}{r['sk_n']:<+10.3f}{r['sk_b']:<+25.3f}{r['temp']:.2f}")
    return lines


LEGEND = ["AUC: ranking quality (0.5 = guessing).",
          "Skill (majority class invariant): accuracy above always giving the more common answer "
          "(0 = no better than that).",
          "Temperature (overconfidence metric): factor by which the predictions must be softened to match "
          "how often they are right (1 = none needed)."]


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
    global DEVICE
    ver = tabpfn_version()
    if MODE != "analyse":
        try:
            import torch
            if not torch.cuda.is_available():
                DEVICE = "cpu"
        except ImportError:
            DEVICE = "cpu"
    print(f"Step1_v35_analysis | tabpfn {ver} | mode {MODE} | device {DEVICE}")
    if MODE != "analyse":
        try:
            from tabpfn.constants import ModelVersion
            ModelVersion.V3_5
        except Exception:
            raise SystemExit("TabPFN-3.5 needs tabpfn >= 9.0 -- run this in the tabpfn 9.x environment")
    plan = check_folders(list(DATASETS))
    print("Data:  " + "   ".join(f"{os.path.join(DATA, n)} ({len(plan[n]['series'])} files)" for n in DATASETS))
    model = make_model() if MODE != "analyse" else None
    tables = []
    for name, spec in plan.items():
        f = spec["folders"]
        saved = load_folder(f["TabPFN v2"], ["ticker", "row", "date", "block", "target", "is_break", "p_base_t"])
        series = spec["series"][:1] if MODE == "check" else spec["series"]
        if MODE != "analyse":
            timing = []
            print(f"\n{name.upper()}: {len(series)} series")
            for tk in series:
                run_series(name, price_file(f["prices"], tk), saved, model, timing)
            if MODE == "check" and timing:
                n_blocks = saved.groupby("ticker")["block"].nunique().mean()
                print(f"  about {np.mean(timing):.3f}s per block -> full run for {name}: "
                      f"~{n_blocks * len(spec['series']) * np.mean(timing) / 3600:.1f} h")
        new_files = glob.glob(os.path.join(RESULTS, name, "*.csv"))
        if not new_files:
            continue
        new = load_folder(os.path.join(RESULTS, name), ["ticker", "row", "p_v35_t"])
        v3 = load_folder(f["TabPFN-3"], ["ticker", "row", "p_v3_t"])
        df = saved.merge(new, on=["ticker", "row"], how="inner").merge(v3, on=["ticker", "row"], how="inner")
        tables.append(table(name, df, df["ticker"].nunique()))
    if MODE == "check":
        print("\n(check mode: the tables below use the series computed so far)")
    for i, t in enumerate(tables, 1):
        print(f"\nTABLE {i}. " + t[0])
        print("\n".join(t[1:]))
    if tables:
        print("\n" + "\n".join(LEGEND))


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