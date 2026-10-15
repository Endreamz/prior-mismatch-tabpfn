"""
Step 2: why the overconfidence of Step 1 happens -- a noise cross-test
======================================================================
Small prior-fitted networks (the same kind of model as TabPFN, trained from scratch) are
trained on two priors that differ in one thing only, the answer noise:
  low            the answers follow the inputs closely
  market-level   as noisy as the test stocks (a logistic regression with time reaches the
                 stocks' AUC of 0.617)
Each model is then tested on both worlds: its own and the other one. A test world is the
same generator as the prior with that world's noise level, so the two test worlds share
the same tasks and differ only in the noise. One number per cell: the temperature.

THE PRIOR (a TabPFN-style random causal model, as in the earlier synthetic experiments)
  each task: a random directed acyclic graph of 16 nodes with random nonlinear functions
  and noise; 8 nodes are the inputs, the last node is the answer (above its median = yes);
  a break moves a sparse set (30%) of edge weights at a time in [0.75, 0.90], keeping each
  node's weight norm fixed so difficulty does not change; context of 64 rows with times in
  [0, 1], the query at time 1. Answer noise: before thresholding, the standardised answer
  gets Gaussian noise with standard deviation SIGMA (low: 0; market-level: calibrated at
  the start of each run and saved in results\\sigma.json).

RUN MODES (edit MODE, press F5)
  "check"   1 seed, short training of one prior: plumbing and a time estimate (a few minutes)
  "run"     5 working seeds x 2 priors (resumes from results\\ if interrupted)

CONVERGENCE CHECK: a trained model counts only if it beats always giving the more common
answer on its own training world (skill above 0). A run that fails is replaced by the next
seed, so each prior always has 5 working seeds (at most MAX_EXTRA replacements).
  "analyse" the table only, from the saved results
Output: step_2_mechanism_explanation_results.txt next to this file; per-model results in
results\\. Needs no market data.

ENVIRONMENT: any environment with PyTorch (GPU recommended), numpy, scipy, scikit-learn.
"""
import json
import os
import sys
import time

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODE = "run"          # "check" | "run" | "analyse"

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")
OUTFILE = os.path.join(_HERE, "step_2_mechanism_explanation_results.txt")

CFG = dict(
    D=8, K=16, N_CTX=64,
    EDGE_P=0.35, DRIFT_FRAC=0.30, NODE_NOISE=0.3, WEIGHT_SCALE=2.0, N_REF=256, X_CLIP=5.0,
    DELTA=2.0,                       # break size (edge-weight change)
    TSTAR_TRAIN=(0.75, 0.90),        # break time
    TARGET_AUC=0.617,                # logistic regression with time on the 20 test stocks
    SIGMA_GRID=[0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    MODEL_D=128, HEADS=4, LAYERS=3, LR=1e-3, GRAD_CLIP=1.0, BS=64,
)
RUNS = dict(check=dict(SEEDS=[0], STEPS=300, N_EVAL=300, N_CALIB=150),
            run=dict(SEEDS=[0, 1, 2, 3, 4], STEPS=8000, N_EVAL=1500, N_CALIB=600))
MIN_SKILL = 1e-4        # convergence check: skill above 0 on its own world (the margin absorbs rounding)
MAX_EXTRA = 5           # at most this many replacement seeds per prior
PRIORS = ["low", "market-level"]          # training prior = answer noise; all tasks have jump breaks
SHAPE = "jumps"

ACTS = [np.tanh, lambda z: np.maximum(z, 0.0), lambda z: z, np.sin]


# ============================================================================= #
# THE PRIOR
# ============================================================================= #
def _act(V, codes):
    """Apply each task's activation (0 tanh, 1 relu, 2 identity, 3 sin) to its column."""
    c = codes[:, None]
    return np.where(c == 0, np.tanh(V), np.where(c == 1, np.maximum(V, 0.0), np.where(c == 2, V, np.sin(V))))


def build(shape, delta, sigma, rng, n, tstar=None):
    """n tasks at once. Returns contexts [n, N_CTX, 1+D+1] (time, inputs, answer), queries
    [n, 1+D] (time 1, inputs) and query answers [n].
    Each task: a random causal graph (16 nodes); 8 nodes are the inputs, the last node the
    answer; a break moves a sparse set of edge weights, keeping each node's weight norm.
    Paired: every random draw is made in the same order whatever `delta` is, so the same rng
    seed gives the same tasks with the break switched off (delta 0) or on."""
    D, K, N = CFG["D"], CFG["K"], CFG["N_CTX"]
    acts = rng.integers(0, len(ACTS), size=(n, K))
    mask = np.zeros((n, K, K), bool)
    for i in range(1, K):
        par = rng.random((n, i)) < CFG["EDGE_P"]
        none = ~par.any(1)
        par[none, rng.integers(0, i, size=n)[none]] = True
        mask[:, :i, i] = par
    n_tp = max(2, int(0.5 * (K - 1)))
    tgt = np.argsort(rng.random((n, K - 1)), 1)[:, :n_tp]
    mask[:, :, K - 1] = False
    np.put_along_axis(mask[:, :K - 1, K - 1], tgt, True, axis=1)
    W = rng.standard_normal((n, K, K)).astype(np.float32) * mask
    W = CFG["WEIGHT_SCALE"] * W / np.sqrt(mask.sum(1).clip(min=1))[:, None, :]
    wnorm = np.linalg.norm(W, axis=1)                                          # [n, K] per child node
    n_drift = np.maximum(1, np.round(CFG["DRIFT_FRAC"] * mask.sum((1, 2)))).astype(int)
    score = np.where(mask, rng.random((n, K, K)), np.inf).reshape(n, -1)
    rank = np.argsort(np.argsort(score, 1), 1).reshape(n, K, K)
    S = rank < n_drift[:, None, None]
    dirs = rng.uniform(-1.0, 1.0, size=(n, K, K)).astype(np.float32)
    feat = np.argsort(rng.random((n, K - 1)), 1)[:, :D]
    ts = rng.uniform(*CFG["TSTAR_TRAIN"], size=n) if tstar is None else np.full(n, float(tstar))
    Wp = W + np.where(S, dirs * delta, 0.0)
    Wp = Wp / (np.linalg.norm(Wp, axis=1, keepdims=True) + 1e-8) * wnorm[:, None, :]
    has_par = mask.any(1)                                                      # [n, K]

    def forward(sched):                                                        # sched: [n, m] in [0, 1], or None = no break
        m = CFG["N_REF"] if sched is None else sched.shape[1]
        V = np.zeros((n, m, K), np.float32)
        eps = rng.standard_normal((n, m, K)).astype(np.float32) * CFG["NODE_NOISE"]
        root = rng.standard_normal((n, m, K)).astype(np.float32)
        for i in range(K):
            pre = root[:, :, i]
            if i > 0:
                if sched is None:                                              # weights at s = 0 are W itself
                    lin = np.einsum("nmi,ni->nm", V[:, :, :i], W[:, :i, i])
                else:
                    Wt = (1.0 - sched)[..., None] * W[:, None, :i, i] + sched[..., None] * Wp[:, None, :i, i]
                    Wt = Wt / (np.linalg.norm(Wt, axis=2, keepdims=True) + 1e-8) * wnorm[:, None, i:i + 1]
                    lin = np.einsum("nmi,nmi->nm", V[:, :, :i], Wt)
                pre = np.where(has_par[:, i:i + 1], lin + eps[:, :, i], pre)
            V[:, :, i] = _act(pre, acts[:, i])
        return V

    ref = forward(None)[:, :, -1]
    mu, sd = np.median(ref, 1), ref.std(1) + 1e-8
    tc = rng.uniform(0.0, 1.0, (n, N)).astype(np.float32)
    t = np.concatenate([tc, np.ones((n, 1), np.float32)], 1)
    sched = t if shape == "smooth" else (t >= ts[:, None]).astype(np.float32)
    V = forward(sched.astype(np.float32))
    noise = rng.standard_normal((n, N + 1)).astype(np.float32) * sigma
    y = (((V[:, :, -1] - mu[:, None]) / sd[:, None] + noise) > 0).astype(np.float32)
    X = np.take_along_axis(V, feat[:, None, :], axis=2)
    mx, vx = X[:, :N].mean(1, keepdims=True), X[:, :N].std(1, keepdims=True) + 1e-6
    X = np.clip((X - mx) / vx, -CFG["X_CLIP"], CFG["X_CLIP"]).astype(np.float32)
    C = np.concatenate([tc[..., None], X[:, :N], y[:, :N, None]], 2).astype(np.float32)
    Q = np.concatenate([np.ones((n, 1), np.float32), X[:, N]], 1).astype(np.float32)
    return C, Q, y[:, N]


def calibrate_sigma(n_tasks):
    """The answer noise at which a per-task logistic regression with time reaches the test
    stocks' AUC (normal tasks, no break)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    res = []
    for sg in CFG["SIGMA_GRID"]:
        C, Q, Y = build("smooth", 0.0, sg, np.random.default_rng(777), n_tasks)
        ps, ys = [], []
        for c, q, yq in zip(C, Q, Y):
            if c[:, -1].min() == c[:, -1].max():
                continue
            ps.append(LogisticRegression(max_iter=300).fit(c[:, :-1], c[:, -1]).predict_proba(q[None, :])[0, 1])
            ys.append(yq)
        res.append((sg, roc_auc_score(ys, ps)))
    sg, a = min(res, key=lambda r: abs(r[1] - CFG["TARGET_AUC"]))
    print("Answer noise calibration (logistic regression with time, normal tasks): "
          + ", ".join(f"{s_:g} -> AUC {a_:.3f}" for s_, a_ in res))
    print(f"  market-level noise = {sg:g} (AUC {a:.3f}; test stocks {CFG['TARGET_AUC']:.3f}); low noise = 0.0")
    return sg


# ============================================================================= #
# MODEL
# ============================================================================= #
def make_model(torch, nn):
    class TimePFN(nn.Module):
        def __init__(self):
            super().__init__()
            d = CFG["MODEL_D"]
            self.ce = nn.Linear(CFG["D"] + 2, d)
            self.qe = nn.Linear(CFG["D"] + 1, d)
            enc = nn.TransformerEncoderLayer(d, CFG["HEADS"], dim_feedforward=2 * d, batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(enc, CFG["LAYERS"])
            self.h = nn.Linear(d, 1)

        def forward(self, ctx, q):
            z = self.enc(torch.cat([self.qe(q).unsqueeze(1), self.ce(ctx)], 1))
            return self.h(z[:, 0, :]).squeeze(-1)
    return TimePFN()


def train(torch, nn, dev, noise, sigma, seed, steps):
    torch.manual_seed(seed)
    rng = np.random.default_rng(10_000 * (1 + PRIORS.index(noise)) + seed)
    m = make_model(torch, nn).to(dev)
    opt = torch.optim.Adam(m.parameters(), CFG["LR"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.1 * CFG["LR"])
    lf = nn.BCEWithLogitsLoss()
    m.train()
    for _ in range(steps):
        C, Q, Y = build(SHAPE, CFG["DELTA"], sigma, rng, CFG["BS"])
        loss = lf(m(torch.tensor(C, device=dev), torch.tensor(Q, device=dev)), torch.tensor(Y, device=dev))
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), CFG["GRAD_CLIP"])
        opt.step(); sched.step()
    m.eval()
    return m


def predict(torch, dev, m, C, Q):
    out = []
    with torch.no_grad():
        for i in range(0, len(C), 500):
            out.append(m(torch.tensor(C[i:i + 500], device=dev), torch.tensor(Q[i:i + 500], device=dev)).cpu().numpy())
    return np.concatenate(out)


# ============================================================================= #
# METRICS
# ============================================================================= #
def metrics(z, y):
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import roc_auc_score
    p = 1.0 / (1.0 + np.exp(-z))
    acc = ((p >= 0.5) == (y > 0.5)).mean()

    def nll(lt):
        q = np.clip(1.0 / (1.0 + np.exp(-z / np.exp(lt))), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    temp = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    return dict(auc=float(roc_auc_score(y, p)), skill=float(acc - max(y.mean(), 1 - y.mean())), temp=temp)


def table(res, n_seeds, n_eval):
    w = 30
    lines = ["Small PFNs trained on one prior and tested on another",
             f"({n_seeds} seed{'s' if n_seeds > 1 else ''}, {n_eval:,} test tasks per cell)",
             "",
             f"{'':<{w}}Temperature (overconfidence metric)",
             f"{'':<{w}}tested on",
             f"{'Trained on':<{w}}{'low noise':<14}market-level noise"]
    for noise in PRIORS:
        rs = [r for r in res if r["noise"] == noise]
        if not rs:
            continue
        g = lambda world: float(np.mean([r[world]["temp"] for r in rs]))
        lines.append(f"{noise + ' noise':<{w}}{g('low'):<14.2f}{g('market-level'):.2f}")
    return lines


LEGEND = ["Temperature (overconfidence metric): factor by which the predictions must be softened to",
          "match how often they are right (1 = none needed; below 1 = underconfident)."]


# ============================================================================= #
# MAIN
# ============================================================================= #
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for st in self.s: st.write(x)
    def flush(self):
        for st in self.s: st.flush()


def converged(r):
    """The convergence check: skill above 0 on the model's own training world."""
    return r[r["noise"]]["skill"] > MIN_SKILL


def _main():
    run = RUNS["check" if MODE == "check" else "run"]
    need = len(run["SEEDS"])
    os.makedirs(RESULTS, exist_ok=True)
    tag = "check_" if MODE == "check" else ""
    fname = lambda noise, seed: os.path.join(RESULTS, f"{tag}cross_{noise}_seed{seed}.json")
    res = []
    if MODE != "analyse":
        import torch
        import torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"step_2_mechanism_explanation | mode {MODE} | device {dev} | {need} working seed(s) | "
              f"{len(PRIORS)} priors x {run['STEPS']:,} training steps")
        sig_file = os.path.join(RESULTS, f"{tag}sigma.json")
        if os.path.exists(sig_file):
            sigma = json.load(open(sig_file))["sigma"]
            print(f"Answer noise from the saved calibration: market-level {sigma:g}, low 0.0")
        else:
            sigma = calibrate_sigma(run["N_CALIB"])
            json.dump(dict(sigma=sigma), open(sig_file, "w"))
        level = {"low": 0.0, "market-level": sigma}
        tests = {}

        def test_sets(seed):
            # both test worlds from the same rng seed: the same tasks, only the answer noise differs
            if seed not in tests:
                tests[seed] = {world: build(SHAPE, CFG["DELTA"], s_, np.random.default_rng(1000 + seed),
                                            run["N_EVAL"]) for world, s_ in level.items()}
            return tests[seed]

        for noise in PRIORS:
            n_ok, seed = 0, 0
            while n_ok < need:
                if seed >= need + MAX_EXTRA:
                    raise SystemExit(f"{noise} noise: only {n_ok} of {need} runs passed the convergence check "
                                     f"after {seed} seeds -- training is unstable, stopping")
                f, per = fname(noise, seed), None
                if os.path.exists(f):
                    r = json.load(open(f))
                    print(f"  seed {seed}: trained on {noise} noise ... already done", end="")
                else:
                    t0 = time.time()
                    print(f"  seed {seed}: trained on {noise} noise ...", end="", flush=True)
                    m = train(torch, nn, dev, noise, level[noise], seed, run["STEPS"])
                    r = dict(noise=noise, seed=seed)
                    for world, (C, Q, Y) in test_sets(seed).items():
                        r[world] = metrics(predict(torch, dev, m, C, Q), Y)
                    json.dump(r, open(f, "w"))
                    per = (time.time() - t0) / run["STEPS"]
                    print(f" done ({time.time() - t0:.0f}s)", end="")
                if converged(r):
                    n_ok += 1
                    print("", flush=True)
                else:
                    print(f" -- did not learn (skill {r[noise]['skill']:+.3f} on its own world)"
                          + ("" if MODE == "check" else f"; seed {seed + 1} added"), flush=True)
                if MODE == "check":
                    if per is not None:
                        print(f"    about {per:.3f}s per training step -> full run: "
                              f"~{per * RUNS['run']['STEPS'] * len(PRIORS) * len(RUNS['run']['SEEDS']) / 3600:.1f} h")
                    break
                seed += 1
            if MODE == "check":
                break
    else:
        print("step_2_mechanism_explanation | mode analyse")
    for noise in PRIORS:
        ok, seed = [], 0
        while len(ok) < need and seed < need + MAX_EXTRA:
            if os.path.exists(fname(noise, seed)):
                r = json.load(open(fname(noise, seed)))
                if converged(r) or MODE == "check":
                    ok.append(r)
            seed += 1
        res += ok
    if res:
        n_seeds = min(sum(r["noise"] == noise for r in res) for noise in PRIORS if any(r["noise"] == noise for r in res))
        lines = table(res, n_seeds, run["N_EVAL"])
        if MODE == "check":
            print("\n(check mode: one prior, short training -- the numbers mean nothing yet)")
        print("\nTABLE 3. " + lines[0])
        print("\n".join(lines[1:]))
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