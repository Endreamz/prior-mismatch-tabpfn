"""
Step 5 for TabPFN-3.5: TabPFN-3.5 fine-tuned on the price-path prior
====================================================================
The TabPFN v2 Step 4 (Step4_tabpfn_v2_finetune.py) repeated for TabPFN-3.5: the same prior,
the same training tasks (same seeds, same task stream), the same recipe as far as the
model allows, the same test days and the same tables.

FOUR ROWS IN TABLES 8 AND 9
  Logistic regression          the classical baseline (as in the v2 Step 4)
  TabPFN-3.5                   the Step 1 result (standard TabPFNClassifier), read from the
                               saved Step 1 predictions
  TabPFN-3.5, called directly  the same pretrained network applied to the pipeline's inputs
                               exactly as in training, without TabPFNClassifier's own
                               preprocessing and ensemble
  TabPFN-3.5 fine-tuned        the fine-tuned network, called directly; mean of 3 seeds
TABLE 10: fine-tuned minus called directly, with 95% intervals over series and seed ranges.

FINE-TUNE RECIPE (the v2 recipe; differences forced by the model are marked)
  start         pretrained TabPFN-3.5 (default classifier weights)
  optimiser     schedule-free AdamW if the schedulefree package is installed, otherwise AdamW
                with a 100-step warm-up; learning rate 1e-5; no weight decay
  training      4,000 steps, batch 32 tasks, 8 question rows per task, fp16 with loss
                scaling, gradient clipping 1.0; seeds 0, 1, 2; task stream seeded 20,000 +
                seed, so the training tasks are exactly those of the v2 fine-tune
  3.5 only      the regression head and target encoders, unused for a yes/no answer, are
                frozen (as in Prior Labs' own fine-tuning code); activation recomputation is
                on to fit in GPU memory (it changes memory use, not the result); the batch
                can be split into micro-batches (MICRO_BS) with gradients accumulated --
                the same 32 tasks per step
  monitoring    loss on 400 fixed held-out price-path tasks, before and after training

TESTING: the Step 1 pipeline and test days; every series must match the saved Step 1 results
exactly (rows, dates, answers, break flags), or the run stops.

FOLDERS (the parent folder can have any name)
  <parent>\\Data\\stocks_test\\, <parent>\\Data\\currencies\\        the original price files (as in Step 1)
  <parent>\\Step1_v35_analysis\\previous_versions_results\\tabpfn_v2\\{stocks_test, currencies}\\
                         saved Step 1 TabPFN v2 predictions (test days, answers, break flags)
  <parent>\\Step1_v35_analysis\\results\\{stocks_test, currencies}\\
                         saved Step 1 TabPFN-3.5 predictions (the TabPFN-3.5 row)
  <parent>\\step3_prior_calibration\\step3_prior_calibration.py and results\\pricepath_params.json
  <parent>\\step5_tabpfn_v35_finetune\\step5_tabpfn_v35_finetune.py   this file
Written by this file (inside step5_tabpfn_v35_finetune):
  checkpoints\\tabpfn-v3.5-pricepath-seed{0,1,2}.safetensors, results\\{stocks_test, currencies}\\,
  step5_tabpfn_v35_finetune_results.txt (check mode: checkpoints_check\\, results_check\\)

RUN MODES (edit MODE, press F5)
  "check"    a 50-step fine-tune of one seed, a reload test, 1 stock + 1 currency, the join
             check, peak GPU memory and a time estimate
  "run"      3 fine-tunes and the full test (resumes from saved checkpoints and predictions)
  "analyse"  the tables only, from the saved predictions
  If "check" runs out of GPU memory, set MICRO_BS to 16 or 8 and run "check" again.

FILES NEEDED
  scripts      this file; step3_prior_calibration.py (in the step3_prior_calibration folder)
  data         Data\\stocks_test (20 files), Data\\currencies (7 files), the saved Step 1 TabPFN v2
               and TabPFN-3.5 predictions, step3_prior_calibration\\results\\pricepath_params.json
  weights      TabPFN-3.5 default classifier weights (cached by the tabpfn package in Step 1)
  environment  the tabpfn 9.x environment used for TabPFN-3.5 in Step 1 (PyTorch with CUDA),
               pandas, scipy, scikit-learn, openpyxl; install schedulefree in it
               (pip install schedulefree) so the optimiser matches the v2 fine-tune
"""
import glob
import importlib.util
import os
import shutil
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
DATASETS = {"stocks_test": 20, "currencies": 7}          # folder name -> expected number of series
OUTFILE = os.path.join(_HERE, "step5_tabpfn_v35_finetune_results.txt")

FT = dict(SEEDS=[0, 1, 2], STEPS=4000, BS=32, N_Q=8, LR=1e-5, WARMUP=100, WEIGHT_DECAY=0.0,
          GRAD_CLIP=1.0, AMP=True, PROGRESS_EVERY=1000)
MICRO_BS = 32           # tasks per forward pass; 16 or 8 if the GPU runs out of memory (same 32 tasks per step)
CHECK = dict(SEEDS=[0], STEPS=50, PROGRESS_EVERY=10)
N_HELDOUT, HELDOUT_SEED = 400, 515_151

X_CLIP = 5.0
WINDOW = 126
MIN_N = 30
CAL_WINDOW, MIN_CAL, LABEL_DELAY = 250, 100, 6

# ---- thesis pipeline constants (as in Step 1) -------------------------------- #
GAP = 5
LABEL_HORIZON = 5
BASELINE_DAYS = 126
FORECAST_HORIZON = 10
TRAIL_VOL_FAST = 10
TRAIL_VOL_SLOW = 60
BREAK_Z = 2.0
BREAK_PAD = 5
MIN_BREAK_OBS = 60
THESIS_WINDOW = 1260
FIRST_TEST_ROW = THESIS_WINDOW + 26

assert GAP >= LABEL_HORIZON, "gap below the label horizon would leak future labels into training"

DEVICE = "cuda"


# ============================================================================= #
# PIPELINE (copied from Step 1)
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


def _read(path, **kw):
    return pd.read_excel(path, **kw) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, **kw)


def load_prices(path):
    """Reads a price file with Date, Open, High, Low, Close columns (also lower-case names,
    dates in the first unnamed column, and the yfinance three-header-row layout)."""
    df = _read(path)
    first = str(df.columns[0]).strip().lower()
    if first == "price" and len(df) > 2 and str(df.iloc[1, 0]).strip().lower() == "date":
        names_ = ["Date"] + [str(c).strip() for c in df.columns[1:]]
        df = _read(path, skiprows=3, header=None)
        df.columns = names_[:df.shape[1]]
    rename = {c: c.strip().capitalize() for c in df.columns if isinstance(c, str)
              and c.strip().lower() in ("date", "open", "high", "low", "close", "volume")}
    df = df.rename(columns=rename)
    if "Date" not in df.columns and str(df.columns[0]).startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "Date"})
    missing = [c for c in ("Date", "High", "Low", "Close") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: cannot read the price file -- missing column(s) {missing}; "
                         f"columns found: {list(df.columns)[:10]}. Use the original price files.")
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ("Open", "High", "Low", "Close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


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
def find_dir(name):
    """A folder directly inside the parent folder, matched without regard to capitals."""
    p = os.path.join(PARENT, name)
    if os.path.isdir(p):
        return p
    for d in os.listdir(PARENT):
        if d.lower() == name.lower() and os.path.isdir(os.path.join(PARENT, d)):
            return os.path.join(PARENT, d)
    return None


def names(folder, exts):
    if not folder or not os.path.isdir(folder):
        return None
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(exts))


def price_file(folder, ticker):
    for ext in ("xlsx", "csv", "xls"):
        p = os.path.join(folder, f"{ticker}.{ext}")
        if os.path.exists(p):
            return p
    return None


def check_folders():
    """Stop before any computing unless every folder and file is in place."""
    problems, plan = [], {}
    step1 = find_dir("Step1_v35_analysis")
    step3 = find_dir("step3_prior_calibration")
    if step1 is None:
        problems.append(f"folder not found: {os.path.join(PARENT, 'Step1_v35_analysis')}")
    if step3 is None:
        problems.append(f"folder not found: {os.path.join(PARENT, 'step3_prior_calibration')}")
    else:
        if not os.path.exists(os.path.join(step3, "step3_prior_calibration.py")):
            problems.append(f"step3_prior_calibration.py not found in {step3}")
        if not os.path.exists(os.path.join(step3, "results", "pricepath_params.json")):
            problems.append(f"{os.path.join(step3, 'results', 'pricepath_params.json')} not found -- run Step 3 first")
    for name, n_exp in DATASETS.items():
        prices = os.path.join(DATA, name)
        saved = os.path.join(step1, "previous_versions_results", "tabpfn_v2", name) if step1 else None
        v35 = os.path.join(step1, "results", name) if step1 else None
        sp, ss = names(prices, (".xlsx", ".xls", ".csv")), names(saved, (".csv",))
        s35 = names(v35, (".csv",))
        if step1 and s35 is None:
            problems.append(f"{name}: saved Step 1 TabPFN-3.5 predictions not found: {v35}")
        if ss and s35 and set(ss) != set(s35):
            problems.append(f"{name}: the saved Step 1 TabPFN v2 and TabPFN-3.5 predictions hold different series")
        if sp is None:
            problems.append(f"{name}: folder not found: {prices}")
        elif len(sp) != n_exp:
            problems.append(f"{name}: {prices} holds {len(sp)} files, expected {n_exp}")
        if step1 and ss is None:
            problems.append(f"{name}: saved Step 1 TabPFN v2 predictions not found: {saved}")
        if sp and ss and set(sp) != set(ss):
            problems.append(f"{name}: the price files and the saved Step 1 predictions hold different series")
        plan[name] = dict(prices=prices, saved=saved, v35=v35, series=sp or [])
    if problems:
        raise SystemExit("FOLDER PROBLEMS -- nothing was computed:\n  " + "\n  ".join(problems))
    return plan, step3


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_folder(folder, cols=None):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, "*.csv")))], ignore_index=True)
    return df if cols is None else df[cols]


# ============================================================================= #
# TABPFN-3.5: LOADING, CALLING, SAVING
# ============================================================================= #
def load_v35(path=None):
    """The raw TabPFN-3.5 network (default classifier weights, or a saved fine-tune)."""
    from tabpfn.model_loading import load_model_criterion_config
    models, _crit, _cfgs, _inf = load_model_criterion_config(
        model_path=path, check_bar_distribution_criterion=False, estimator_type="classifier",
        version="v3.5", download_if_not_exists=True)
    return models[0].float().to(DEVICE)


def base_checkpoint():
    """The default TabPFN-3.5 checkpoint as a dict (config etc.), to save fine-tunes in the
    same format."""
    from tabpfn.checkpoint import Checkpoint
    from tabpfn.model_loading import resolve_model_path
    paths, _d, _n, _w = resolve_model_path(None, "classifier", "v3.5")
    return {k: v for k, v in Checkpoint(paths[0]).load().items() if k != "state_dict"}


def freeze_unused(model):
    """Freeze what a yes/no answer never uses: the regression head and target encoders
    (as Prior Labs' fine-tuning code does)."""
    mods = [model.col_y_encoder["regression"], model.icl_y_encoder["regression"],
            model.heads.mlp_regression, model.heads.output_projection]
    n = 0
    for m in mods:
        for prm in m.parameters():
            prm.requires_grad = False
            n += prm.numel()
    return n


def sdpa_ctx():
    try:
        from tabpfn.finetuning._torch_compat import sdpa_kernel_context
        return sdpa_kernel_context()
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def perf_options():
    from tabpfn.architectures.interface import PerformanceOptions
    return PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)


def stack_tasks(torch, tasks):
    Xtr = torch.tensor(np.stack([t[0] for t in tasks], 1))
    ytr = torch.tensor(np.stack([t[1] for t in tasks], 1)).unsqueeze(-1)
    Xte = torch.tensor(np.stack([t[2] for t in tasks], 1))
    yte = torch.tensor(np.stack([t[3] for t in tasks], 1))
    return Xtr, ytr, Xte, yte


def forward(torch, model, Xtr, ytr, Xte, amp):
    """Context and question rows in one table (rows, tasks, columns); answers for the context
    rows; returns the yes/no logits of the question rows (questions, tasks, 2)."""
    x = torch.cat([Xtr, Xte], 0).to(DEVICE)
    y = ytr.to(DEVICE).squeeze(-1)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and DEVICE == "cuda")), sdpa_ctx():
        out = model(x, y, "multiclass", performance_options=perf_options())
    return out[:, :, :2].float()


def make_optimizer(torch, model):
    params = [prm for prm in model.parameters() if prm.requires_grad]
    try:
        from schedulefree import AdamWScheduleFree
        return AdamWScheduleFree(params, lr=FT["LR"], warmup_steps=FT["WARMUP"],
                                 weight_decay=FT["WEIGHT_DECAY"]), None, True
    except ImportError:
        opt = torch.optim.AdamW(params, lr=FT["LR"], weight_decay=FT["WEIGHT_DECAY"])
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / FT["WARMUP"])), False


def set_mode(model, opt, is_sf, train):
    model.train() if train else model.eval()
    if is_sf:
        opt.train() if train else opt.eval()


def make_scaler(torch, enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def heldout_tasks(PP, p):
    rng = np.random.default_rng(HELDOUT_SEED)
    return [PP.task_batch(rng, N_HELDOUT // 2, p, n_ctx=n) for n in (64, 126)]


def heldout_loss(torch, model, sets, amp):
    import torch.nn.functional as F
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for tasks in sets:
            for i in range(0, len(tasks), MICRO_BS):
                Xtr, ytr, Xte, yte = stack_tasks(torch, tasks[i:i + MICRO_BS])
                o = forward(torch, model, Xtr, ytr, Xte, amp).reshape(-1, 2)
                tot += F.cross_entropy(o, yte.to(DEVICE).reshape(-1).long(), reduction="sum").item()
                n += yte.numel()
    return tot / n


def finetune(torch, PP, p, seed, steps, every, ck, held, base_ck):
    import torch.nn.functional as F
    t0 = time.time()
    print(f"fine-tune, seed {seed}: {steps:,} steps ...", flush=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(20_000 + seed)
    model = load_v35(None)
    n_frozen = freeze_unused(model)
    if seed == FT["SEEDS"][0] or steps == CHECK["STEPS"]:
        n_all = sum(prm.numel() for prm in model.parameters())
        print(f"  TabPFN-3.5: {n_all / 1e6:.0f}M parameters, {n_frozen / 1e6:.1f}M frozen (regression parts)")
    opt, sched, is_sf = make_optimizer(torch, model)
    amp = FT["AMP"] and DEVICE == "cuda"
    scaler = make_scaler(torch, amp)
    set_mode(model, opt, is_sf, False)
    l0 = heldout_loss(torch, model, held, amp)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, skipped, t_loop = [], 0, time.time()
    n_micro = max(1, FT["BS"] // MICRO_BS)
    for step in range(1, steps + 1):
        set_mode(model, opt, is_sf, True)
        tasks = PP.task_batch(rng, FT["BS"], p, n_q=FT["N_Q"])          # the same 32 tasks as the v2 fine-tune
        bad, tot = False, 0.0
        for k in range(n_micro):
            Xtr, ytr, Xte, yte = stack_tasks(torch, tasks[k * MICRO_BS:(k + 1) * MICRO_BS])
            loss = F.cross_entropy(forward(torch, model, Xtr, ytr, Xte, amp).reshape(-1, 2),
                                   yte.to(DEVICE).reshape(-1).long()) / n_micro
            if not torch.isfinite(loss):
                bad = True; break
            with sdpa_ctx():
                scaler.scale(loss).backward()
            tot += loss.item()
        if bad:
            skipped += 1; opt.zero_grad(set_to_none=True); continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([prm for prm in model.parameters() if prm.requires_grad], FT["GRAD_CLIP"])
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if sched is not None:
            sched.step()
        losses.append(tot)
        if step % every == 0 and step < steps:
            el = time.time() - t_loop
            print(f"  seed {seed}: step {step:,}/{steps:,}, training loss {np.mean(losses[-100:]):.3f}, "
                  f"~{el / step * (steps - step) / 60:.0f} min left", flush=True)
    per_step = (time.time() - t_loop) / steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE == "cuda" else float("nan")
    set_mode(model, opt, is_sf, False)
    l1 = heldout_loss(torch, model, held, amp)
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    from tabpfn.checkpoint import save_as_safetensors          # the same format as the original weights
    save_as_safetensors(dict(base_ck, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), ck)
    print(f"fine-tune, seed {seed}: done ({(time.time() - t0) / 60:.0f} min) | held-out price-path tasks: "
          f"loss {l0:.3f} -> {l1:.3f} | peak GPU memory {peak:.1f} GB"
          + (f" | {skipped} steps skipped (non-finite loss)" if skipped else ""), flush=True)
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return per_step


# ============================================================================= #
# CALLING THE NETWORK DIRECTLY
# ============================================================================= #
class DirectPFN:
    """The network called directly, exactly as in training: context and question rows in one
    table, context answers; no TabPFNClassifier processing."""
    def __init__(self, torch, path):
        self.torch = torch
        self.m = load_v35(path).eval()

    def predict(self, X, y, Xq):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.vstack([X, Xq]).astype(np.float32), device=DEVICE)[:, None, :]
            ty = t.tensor(np.asarray(y, np.float32), device=DEVICE)[:, None]
            out = self.m(x, ty, "multiclass")
            return t.softmax(out[:, 0, :2].float(), -1)[:, 1].cpu().numpy()


class LogReg:
    """The classical baseline: logistic regression on the same context (scikit-learn's
    standard settings)."""
    def predict(self, X, y, Xq):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=300).fit(X, y).predict_proba(Xq)[:, 1]


def run_series(name, path, saved, models, res_dir, timing):
    prep = prepare_ticker(path)
    if prep is None:
        print(f"  {os.path.basename(path)}: skipped (too few breaks or too short)"); return
    ticker, dfc, feats = prep
    op = os.path.join(res_dir, name, f"{ticker}.csv")
    done = pd.read_csv(op) if os.path.exists(op) else None
    todo = [c for c in models if done is None or c not in done.columns]
    if not todo:
        print(f"  {ticker}: already computed"); return
    X = dfc[feats].values.astype(np.float64)
    y = dfc["Target_Vol_Spike"].values.astype(int)
    blocks = walk_forward_blocks(len(dfc), WINDOW)
    rows, dates, targets, brk = [], [], [], []
    probs = {c: [] for c in todo}
    t0 = time.time()
    print(f"  {ticker}: {len(blocks)} blocks x {len(todo)} model(s) ...", end="", flush=True)
    for trs, tre, ts, te in blocks:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[trs:tre]); Xte = sc.transform(X[ts:te])
        ytr = y[trs:tre]
        t_ctx = (np.arange(trs, tre) - trs) / float(max(1, tre - trs - 1))
        Xtr_t = np.column_stack([t_ctx, np.clip(Xtr, -X_CLIP, X_CLIP)])
        Xte_t = np.column_stack([np.ones(te - ts), np.clip(Xte, -X_CLIP, X_CLIP)])
        rows.extend(range(ts, te)); dates.extend(dfc["Date"].iloc[ts:te].astype(str))
        targets.extend(y[ts:te]); brk.extend(dfc["near_transition"].values[ts:te])
        single = len(np.unique(ytr)) < 2
        for c in todo:
            if single:
                p = np.full(te - ts, (ytr.sum() + 1.0) / (len(ytr) + 2.0))
            else:
                t1 = time.time()
                p = models[c].predict(Xtr_t, ytr, Xte_t)
                timing.append(time.time() - t1)
            probs[c].extend(p)
    out = pd.DataFrame(dict(ticker=ticker, row=rows, date=dates, target=targets, is_break=brk))
    for c in todo:
        out[c] = probs[c]
    check_join(out, saved, ticker)
    if done is not None:
        for c in todo:
            done[c] = out[c].values
        out = done
    os.makedirs(os.path.dirname(op), exist_ok=True)
    out.to_csv(op, index=False)
    print(f" done ({time.time() - t0:.0f}s), join check passed", flush=True)


def check_join(new, saved, ticker):
    s = saved[saved["ticker"] == ticker].sort_values("row").reset_index(drop=True)
    n = new.sort_values("row").reset_index(drop=True)
    ok = (len(s) == len(n) and np.array_equal(s["row"].values, n["row"].values)
          and (s["date"].astype(str).str[:10].values == n["date"].astype(str).str[:10].values).all()
          and np.array_equal(s["target"].values.astype(int), n["target"].values.astype(int))
          and np.array_equal(s["is_break"].values.astype(bool), n["is_break"].values.astype(bool)))
    if not ok:
        raise SystemExit(f"\n{ticker}: days, answers or break flags differ from the saved Step 1 TabPFN v2 results -- "
                         f"the price file is not the original one used for them, so the comparison is refused")


# ============================================================================= #
# TABLES (the Step 1 metrics)
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


ROWS = [("Logistic regression", ["p_lr"]),
        ("TabPFN-3.5", ["p_v35_t"]),
        ("TabPFN-3.5, called directly", ["p_direct"]),
        ("TabPFN-3.5 fine-tuned", [f"p_ft_s{s}" for s in FT["SEEDS"]])]


def table(name, df, n_series, number):
    w = 35
    title = {"stocks_test": f"TabPFN-3.5 fine-tuned on the price-path prior: {n_series} test stocks",
             "currencies": f"The same for the {n_series} currencies"}[name]
    lines = [f"TABLE {number}. {title} ({len(df):,} test days)", "",
             f"{'':<{w}}{'AUC':<17}{'Skill (majority class invariant)':<35}Temperature",
             f"{'Model':<{w}}{'normal':<9}{'break':<8}{'normal':<10}{'break':<25}(overconfidence metric)"]
    for label, cols in ROWS:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        vals = {k: [] for k in ("auc_n", "auc_b", "sk_n", "sk_b")}
        temps = []
        for _, d in df.groupby("ticker"):
            d = d.sort_values("row").reset_index(drop=True)
            y, b = d["target"].values.astype(float), d["is_break"].values.astype(bool)
            for cell, m in (("n", ~b), ("b", b)):
                if m.sum() >= MIN_N:
                    vals[f"auc_{cell}"].append(np.nanmean([auc(y[m], d[c].values[m]) for c in cols]))
                    vals[f"sk_{cell}"].append(np.mean([skill(y[m], d[c].values[m]) for c in cols]))
            for c in cols:
                temps += temperatures(d, c)
        r = {k: np.nanmean(v) for k, v in vals.items()}
        t = float(np.median(temps)) if temps else np.nan
        lines.append(f"{label:<{w}}{r['auc_n']:<9.3f}{r['auc_b']:<8.3f}{r['sk_n']:<+10.3f}{r['sk_b']:<+25.3f}{t:.2f}")
    return lines


LEGEND = [
    "Logistic regression: the classical baseline; the same inputs (with time), 126-day window, gap and test days.",
    "TabPFN-3.5: as in Step 1, through the standard TabPFNClassifier (its own preprocessing and ensemble).",
    "Called directly: the network applied to the pipeline's inputs exactly as in training, without that preprocessing.",
    "TabPFN-3.5 fine-tuned: pretrained TabPFN-3.5 trained further on price-path tasks; mean of 3 seeds; called directly.",
    "AUC: ranking quality (0.5 = guessing).",
    "Skill (majority class invariant): accuracy above always giving the more common answer (0 = no better than that).",
    "Temperature (overconfidence metric): factor by which the predictions must be softened to match how often they "
    "are right (1 = none needed).",
]


N_BOOT = 2000


def series_metrics(df, cols):
    """Per series: AUC and skill on normal and break days (averaged over cols), and the
    walk-forward temperatures (one list per col)."""
    out = {}
    for tk, d in df.groupby("ticker"):
        d = d.sort_values("row").reset_index(drop=True)
        y, b = d["target"].values.astype(float), d["is_break"].values.astype(bool)
        r = {}
        for cell, m in (("n", ~b), ("b", b)):
            ok = m.sum() >= MIN_N
            r[f"auc_{cell}"] = np.nanmean([auc(y[m], d[c].values[m]) for c in cols]) if ok else np.nan
            r[f"sk_{cell}"] = np.mean([skill(y[m], d[c].values[m]) for c in cols]) if ok else np.nan
        r["temps"] = [temperatures(d, c) for c in cols]
        out[tk] = r
    return out


def effect(df):
    """Fine-tuned minus called directly: mean, 95% interval over series, range over seeds."""
    ft = [c for c in ROWS[-1][1] if c in df.columns]
    base = series_metrics(df, ["p_direct"])
    both = series_metrics(df, ft)
    per_seed = [series_metrics(df, [c]) for c in ft]
    tks = sorted(base)
    rng = np.random.default_rng(0)
    res = {}
    for k in ("auc_n", "auc_b", "sk_n", "sk_b"):
        diff = np.array([both[t][k] - base[t][k] for t in tks])
        diff = diff[np.isfinite(diff)]
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
        seeds = [np.nanmean([ps[t][k] - base[t][k] for t in tks]) for ps in per_seed]
        res[k] = (diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), min(seeds), max(seeds))

    # temperature: the median over all walk-forward fits, as in Tables 5 and 6 (resampling series)
    arr = lambda metr, j=None: [np.array([x for jj, lst in enumerate(metr[t]["temps"]) if j is None or jj == j for x in lst])
                                for t in tks]
    A, B_ = arr(both), arr(base)
    pooled = lambda arrs, idx: np.median(np.concatenate([arrs[i] for i in idx])) if len(idx) else np.nan
    allidx = np.arange(len(tks))
    d0 = pooled(A, allidx) - pooled(B_, allidx)
    bs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(tks), len(tks))
        bs.append(pooled(A, idx) - pooled(B_, idx))
    seeds = [pooled(arr(both, j), allidx) - pooled(B_, allidx) for j in range(len(ft))]
    res["temp"] = (d0, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), min(seeds), max(seeds))
    return res, len(tks)


def table7(dfs):
    """dfs: {dataset name: joined predictions}."""
    eff = {name: effect(df) for name, df in dfs.items()}
    w0, w1, w2, w3 = 27, 9, 18, 18
    label = {"stocks_test": "test stocks", "currencies": "currencies"}
    head1 = f"{'':<{w0}}" + "".join(f"{f'{n} {label[name]}':<{w1 + w2 + w3 + 2}}" for name, (_, n) in eff.items())
    head2 = f"{'Difference in':<{w0}}" + "".join(f"{'mean':<{w1}}{'95% interval':<{w2}}{'seed range':<{w3 + 2}}" for _ in eff)
    lines = ["TABLE 10. Effect of the fine-tune: TabPFN-3.5 fine-tuned minus TabPFN-3.5 called directly", "", head1, head2]
    for k, name, f in [("auc_n", "AUC, normal days", "+.3f"), ("auc_b", "AUC, break days", "+.3f"),
                       ("sk_n", "Skill, normal days", "+.3f"), ("sk_b", "Skill, break days", "+.3f"),
                       ("temp", "Temperature", "+.2f")]:
        line = f"{name:<{w0}}"
        for _, (r, _) in eff.items():
            m, lo, hi, s0, s1 = r[k]
            line += (f"{format(m, f):<{w1}}{'[' + format(lo, f) + ', ' + format(hi, f) + ']':<{w2}}"
                     f"{format(s0, f) + ' to ' + format(s1, f):<{w3 + 2}}")
        lines.append(line.rstrip())
    return lines + ["", "95% interval: resampling series (each series counts once; 2,000 resamples).",
                    "Seed range: the smallest and largest difference among the three fine-tunes."]


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
    check = MODE == "check"
    res_dir = os.path.join(_HERE, "results_check" if check else "results")
    ck_dir = os.path.join(_HERE, "checkpoints_check" if check else "checkpoints")
    plan, step3 = check_folders()
    if check:                                   # a fresh check every time
        shutil.rmtree(res_dir, ignore_errors=True)
        shutil.rmtree(ck_dir, ignore_errors=True)
    params_file = os.path.join(step3, "results", "pricepath_params.json")
    seeds = CHECK["SEEDS"] if check else FT["SEEDS"]
    if MODE != "analyse":
        try:
            import tabpfn
            ver = getattr(tabpfn, "__version__", "unknown")
        except ImportError:
            ver = None
        try:
            from tabpfn.constants import ModelVersion
            ModelVersion.V3_5
        except Exception:
            raise SystemExit(f"TabPFN-3.5 needs tabpfn >= 9 (found {ver}): run this in the tabpfn 9.x environment of Step 1")
        import torch
        if not torch.cuda.is_available():
            DEVICE = "cpu"
        print(f"step5_tabpfn_v35_finetune | tabpfn {ver} | mode {MODE} | device {DEVICE} | micro-batch {MICRO_BS} | "
              f"prior: {params_file}")
        PP = load_module("step3_prior_calibration", os.path.join(step3, "step3_prior_calibration.py"))
        p = PP.load_params(params_file)
        try:
            import schedulefree  # noqa: F401
            opt_name = "schedule-free AdamW"
        except ImportError:
            opt_name = "AdamW with a 100-step warm-up (schedulefree not installed)"
        print(f"optimiser: {opt_name}")

        # ---- fine-tunes
        held, per_step, base_ck = None, None, None
        for s in seeds:
            ck = os.path.join(ck_dir, f"tabpfn-v3.5-pricepath-seed{s}.safetensors")
            if os.path.exists(ck) and not check:
                print(f"fine-tune, seed {s}: already trained ({os.path.relpath(ck, _HERE)})"); continue
            if held is None:
                held = heldout_tasks(PP, p)
                base_ck = base_checkpoint()
            per_step = finetune(torch, PP, p, s, CHECK["STEPS"] if check else FT["STEPS"],
                                CHECK["PROGRESS_EVERY"] if check else FT["PROGRESS_EVERY"], ck, held, base_ck)

        # ---- the networks, called directly (loading a checkpoint is also the reload test)
        models = {"p_lr": LogReg(), "p_direct": DirectPFN(torch, None)}
        for s in seeds:
            models[f"p_ft_s{s}"] = DirectPFN(torch, os.path.join(ck_dir, f"tabpfn-v3.5-pricepath-seed{s}.safetensors"))
        if check:
            print("checkpoint reload: OK")

        # ---- testing
        timing = []
        for name, spec in plan.items():
            saved = load_folder(spec["saved"], ["ticker", "row", "date", "block", "target", "is_break", "p_base_t"])
            series = spec["series"][:1] if check else spec["series"]
            print(f"{name}: {len(series)} series, logistic regression + TabPFN-3.5 called directly + {len(seeds)} fine-tune(s)")
            for tk in series:
                run_series(name, price_file(spec["prices"], tk), saved, models, res_dir, timing)
        if check and timing:
            n_blocks = sum(load_folder(spec["saved"], ["ticker", "block"]).groupby("ticker")["block"].nunique().sum()
                           for spec in plan.values())
            per_block = np.mean(timing) * (2 + len(FT["SEEDS"]))       # logistic regression, direct, 3 fine-tunes
            print(f"time estimate for \"run\": fine-tunes ~{per_step * FT['STEPS'] * len(FT['SEEDS']) / 60:.0f} min "
                  f"+ testing ~{n_blocks * per_block / 60:.0f} min")
    else:
        print("step5_tabpfn_v35_finetune | mode analyse")

    tables, dfs = [], {}
    for i, (name, spec) in enumerate(plan.items()):
        if not glob.glob(os.path.join(res_dir, name, "*.csv")):
            continue
        saved = load_folder(spec["saved"], ["ticker", "row", "block", "target", "is_break", "p_base_t"])
        new = load_folder(os.path.join(res_dir, name))
        keep = ["ticker", "row"] + [c for c in new.columns if c.startswith("p_")]
        df = saved.merge(new[keep], on=["ticker", "row"], how="inner")
        v35 = load_folder(spec["v35"], ["ticker", "row", "p_v35_t"])
        df = df.merge(v35, on=["ticker", "row"], how="inner")
        tables.append(table(name, df, df["ticker"].nunique(), 8 + i))
        if "p_direct" in df.columns and any(c in df.columns for c in ROWS[-1][1]):
            dfs[name] = df
    if check:
        print("\n(check mode: 50 training steps, one stock and one currency -- the numbers mean nothing yet)")
    for t in tables:
        print("\n" + "\n".join(t))
    if tables:
        print("\n" + "\n".join(LEGEND))
    if dfs:
        print("\n" + "\n".join(table7(dfs)))


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
