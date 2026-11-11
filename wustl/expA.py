import os
# ===== 9c-3: hardening with matched training budgets: central-10, central-40, local-10, federated (self-contained) =====
import numpy as np, pandas as pd, torch, torch.nn as nn, time
import sys; torch.set_num_threads(4)
from sklearn.metrics import roc_auc_score
P = os.environ.get("WUSTL_CSV", "data/wustl-ehms-2020.csv")
OUT = "results/wustl"; os.makedirs(OUT, exist_ok=True)
dev = "cpu"
df = pd.read_csv(P); df.columns = [c.strip() for c in df.columns]
y = df["Label"].astype(int).values; cat = df["Attack Category"].astype(str).str.lower().values; N=len(y)
ids = ["Dir","Flgs","SrcAddr","DstAddr","Sport","Dport","SrcMac","DstMac","Packet_num"]
cols = [c for c in df.columns if c not in ids+["Label","Attack Category"]]
X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
edges = np.linspace(0, N, 21).astype(int); test_blocks = [3, 7, 11, 15, 19]
te = np.concatenate([np.arange(edges[b], edges[b+1]) for b in test_blocks])
banned = np.zeros(N, bool)
for b in test_blocks: banned[max(0, edges[b]-100):min(N, edges[b+1]+100)] = True
tr = np.where(~banned)[0]; trn0 = tr[y[tr] == 0]
mu, sd = X[trn0].mean(0), X[trn0].std(0) + 1e-6
print("test", len(te), int(y[te].sum()), "train", len(tr))
Z = ((X - mu) / sd).astype(np.float32); Zt = torch.tensor(Z, device=dev)
tr = np.sort(tr); nval = len(tr) // 5
val, fit = tr[-nval:], tr[:-nval]; fit = fit[fit < val.min() - 100]; trn = fit[y[fit] == 0]
lo = torch.tensor(np.percentile(Z[trn], 0.5, axis=0), device=dev); hi = torch.tensor(np.percentile(Z[trn], 99.5, axis=0), device=dev)
PW = float((1 - y[fit].mean()) / y[fit].mean())
alt_te = te[cat[te] == "data alteration"]; spo_te = te[cat[te] == "spoofing"]; nor_te = te[y[te] == 0]
x0 = Zt[alt_te]; xsp = Zt[spo_te]; xnor = Zt[nor_te]
EPS = [0.25, 0.5, 1.0]; N_CLIENTS = 4

def make_model():
    return nn.Sequential(nn.Linear(34,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1)).to(dev)
def local_train(m, A, t, epochs):
    opt = torch.optim.Adam(m.parameters(), 1e-3); pw = torch.tensor(PW, device=dev); m.train()
    for ep in range(epochs):
        perm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 256):
            idx = perm[i:i+256]; opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(A[idx]).squeeze(1), t[idx], pos_weight=pw).backward(); opt.step()
VAL_X = None
def val_loss(m):
    m.eval()
    with torch.no_grad():
        l = float(nn.functional.binary_cross_entropy_with_logits(m(VAL_X).squeeze(1), VAL_Y, pos_weight=torch.tensor(PW)))
    m.train(); return l
CHOSEN = []
def train_central(Xc, yc, seed, epochs):
    torch.manual_seed(seed); m = make_model()
    A = torch.tensor(Xc, device=dev); t = torch.tensor(yc, dtype=torch.float32, device=dev)
    if epochs != "ES":
        local_train(m, A, t, epochs); m.eval(); return m
    best, best_state, bad, best_ep = 1e9, None, 0, 0
    for ep in range(1, MAXEP + 1):
        local_train(m, A, t, 1); l = val_loss(m)
        if l < best - 1e-6: best, best_state, bad, best_ep = l, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(best_state); m.eval(); CHOSEN.append(("central", seed, best_ep)); return m
def fedavg(clients, seed, rounds=40, E=1, es=False):
    torch.manual_seed(seed); g = make_model(); best, best_state, bad, best_r = 1e9, None, 0, 0
    data = [(torch.tensor(c["X"], device=dev), torch.tensor(c["y"], dtype=torch.float32, device=dev)) for c in clients]
    w = np.array([len(c["y"]) for c in clients], float); w /= w.sum()
    for r in range(rounds):
        states = []
        for A, t in data:
            m = make_model(); m.load_state_dict(g.state_dict()); local_train(m, A, t, E); states.append(m.state_dict())
        g.load_state_dict({k: sum(float(w[i]) * states[i][k] for i in range(len(states))) for k in states[0]})
        if es:
            l = val_loss(g)
            if l < best - 1e-6: best, best_state, bad, best_r = l, {k: v.clone() for k, v in g.state_dict().items()}, 0, r + 1
            else:
                bad += 1
                if bad >= PATIENCE: break
    if es: g.load_state_dict(best_state); CHOSEN.append(("fed", seed, best_r))
    g.eval(); return g
fn = lambda m: (lambda T: torch.sigmoid(m(T).squeeze(-1)).detach())

def attack_es(f, x0_, eps, seed, gens=25, pop=16, want_x=False):
    g = torch.Generator(device=dev).manual_seed(seed); S, D = x0_.shape
    lo_s = torch.minimum(lo, x0_); hi_s = torch.maximum(hi, x0_); best_x, best_s = x0_.clone(), f(x0_)
    for _ in range(gens):
        cand = best_x[:, None, :] + torch.randn(S, pop, D, device=dev, generator=g) * (eps / 2)
        cand = x0_[:, None, :] + (cand - x0_[:, None, :]).clamp(-eps, eps)
        cand = torch.minimum(torch.maximum(cand, lo_s[:, None, :]), hi_s[:, None, :])
        sc = f(cand.reshape(-1, D)).reshape(S, pop); mn, ai = sc.min(1); upd = mn < best_s
        best_s = torch.where(upd, mn, best_s); best_x = torch.where(upd[:, None], cand[torch.arange(S), ai], best_x)
    return (best_s, best_x) if want_x else best_s
def attack_pgd(m, x0_, eps, seed, steps=50, restarts=2):
    g = torch.Generator(device=dev).manual_seed(seed)
    lo_s = torch.minimum(lo, x0_); hi_s = torch.maximum(hi, x0_); alpha = eps / 10
    best_l = m(x0_).squeeze(-1).detach()
    for r in range(restarts):
        x = x0_ + (torch.rand(x0_.shape, device=dev, generator=g) * 2 - 1) * eps * float(r > 0)
        x = torch.minimum(torch.maximum(x, lo_s), hi_s)
        for _ in range(steps):
            x = x.detach().requires_grad_(True); m(x).squeeze(-1).sum().backward()
            with torch.no_grad():
                x = x - alpha * x.grad.sign(); x = x0_ + (x - x0_).clamp(-eps, eps); x = torch.minimum(torch.maximum(x, lo_s), hi_s)
                best_l = torch.minimum(best_l, m(x).squeeze(-1))
    return torch.sigmoid(best_l)

q95 = lambda a: float(np.percentile(a.cpu().numpy(), 95))
def evaluate(models, seed):
    outs = []
    for m in models:
        f = fn(m); thr = q95(f(xnor))                                  # test-matched 5% FPR (oracle diagnostic)
        r = dict(AUC=roc_auc_score(y[te], f(Zt[te]).cpu().numpy()), rec_alt=float((f(x0) > thr).float().mean()),
                 rec_spoof=float((f(xsp) > thr).float().mean()))
        det0 = f(x0) > thr; n0 = max(int(det0.sum()), 1)
        for eps in (0.5, 1.0):
            r[f"ES@{eps}"]  = float((det0 & (attack_es(f, x0, eps, 1000 + seed) <= thr)).sum() / n0)
            r[f"PGD@{eps}"] = float((det0 & (attack_pgd(m, x0, eps, 7000 + seed) <= thr)).sum() / n0)
        outs.append(r)
    return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}

def run_setting(mode, epochs, seed):
    parts = [fit] if mode == "central" else np.array_split(fit, N_CLIENTS)
    clients = [dict(X=Z[p].copy(), y=y[p].copy(), alt=Zt[p[cat[p] == "data alteration"]]) for p in parts]
    train = (lambda cl: [fedavg(cl, seed, es=(epochs=="ES"))]) if mode == "fed" else (lambda cl: [train_central(c["X"], c["y"], seed, epochs) for c in cl])
    models = train(clients); res = {"round 0": evaluate(models, seed)}
    for rnd in (1, 2):
        for ci, c in enumerate(clients):
            if len(c["alt"]) == 0: continue
            m = models[0] if mode == "fed" else models[ci]; f = fn(m)
            advs = [attack_es(f, c["alt"], e, 200 + seed*10 + rnd*100 + ci*1000 + k, want_x=True)[1].cpu().numpy() for k, e in enumerate(EPS)]
            c["X"] = np.vstack([c["X"]] + advs); c["y"] = np.r_[c["y"], np.ones(sum(len(a) for a in advs), dtype=c["y"].dtype)]
        models = train(clients); res[f"round {rnd}"] = evaluate(models, seed)
    return res

VAL_X = Zt[val]; VAL_Y = torch.tensor(y[val], dtype=torch.float32)
MAXEP = 40; PATIENCE = 5
SETTINGS = [("central, early stopping", "central", "ES"), ("central, 10 epochs", "central", 10), ("central, 40 epochs", "central", 40),
            ("local-only, early stopping", "local", "ES"), ("federated, early stopping", "fed", "ES"), ("federated (FedAvg, E=1, R=40)", "fed", 1)]
NSEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rows = []; t0 = time.time()
for seed in range(NSEEDS):
    for name, mode, ep in SETTINGS:
        for rname, r in run_setting(mode, ep, seed).items(): rows.append(dict(seed=seed, setting=name, round=rname, **r))
    print("seed", seed, "done", round(time.time() - t0), "s", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/expA_early_stopping.csv", index=False); pd.DataFrame(CHOSEN, columns=["mode","seed","best"]).to_csv(f"{OUT}/expA_chosen_epochs.csv", index=False)
g = R.groupby(["setting", "round"], sort=False)
fmt = lambda c: g[c].mean().round(3).astype(str) + " ± " + g[c].std().round(3).astype(str)
print(pd.DataFrame(CHOSEN, columns=["mode","seed","best"]).groupby("mode")["best"].describe()); print("\nHARDENING WITH MATCHED TRAINING BUDGETS (6 seeds, mean ± SD; evasion at test-matched 5% FPR)")
print(pd.DataFrame({c: fmt(c) for c in ["AUC", "rec_alt", "rec_spoof", "ES@0.5", "PGD@0.5", "ES@1.0", "PGD@1.0"]}).to_string())