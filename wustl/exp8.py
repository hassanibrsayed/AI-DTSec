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
torch.use_deterministic_algorithms(True)
tr = np.sort(tr); nval = len(tr) // 5
val, fit = tr[-nval:], tr[:-nval]; fit = fit[fit < val.min() - 100]; trn = fit[y[fit] == 0]
lo = torch.tensor(np.percentile(Z[trn], 0.5, axis=0), device=dev); hi = torch.tensor(np.percentile(Z[trn], 99.5, axis=0), device=dev)
PW = float((1 - y[fit].mean()) / y[fit].mean())
alt_te = te[cat[te] == "data alteration"]; spo_te = te[cat[te] == "spoofing"]; nor_te = te[y[te] == 0]
x0 = Zt[alt_te]; xsp = Zt[spo_te]; xval_n = Zt[val[y[val] == 0]]; xnor = Zt[nor_te]
N_CLIENTS = 4

def make_model():
    return nn.Sequential(nn.Linear(34,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1)).to(dev)
def local_train(m, A, t, epochs):
    opt = torch.optim.Adam(m.parameters(), 1e-3); pw = torch.tensor(PW, device=dev); m.train()
    for ep in range(epochs):
        perm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 256):
            idx = perm[i:i+256]; opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(A[idx]).squeeze(1), t[idx], pos_weight=pw).backward(); opt.step()
def train_central(Xc, yc, seed, epochs):
    torch.manual_seed(seed); m = make_model()
    local_train(m, torch.tensor(Xc, device=dev), torch.tensor(yc, dtype=torch.float32, device=dev), epochs); m.eval(); return m
def fedavg(parts, seed, rounds, E):
    torch.manual_seed(seed); g = make_model()
    data = [(torch.tensor(Z[p], device=dev), torch.tensor(y[p], dtype=torch.float32, device=dev)) for p in parts]
    w = np.array([len(p) for p in parts], float); w /= w.sum()
    for r in range(rounds):
        states = []
        for A, t in data:
            m = make_model(); m.load_state_dict(g.state_dict()); local_train(m, A, t, E); states.append(m.state_dict())
        g.load_state_dict({k: sum(float(w[i]) * states[i][k] for i in range(len(states))) for k in states[0]})
    g.eval(); return g
fn = lambda m: (lambda T: torch.sigmoid(m(T).squeeze(-1)).detach())

def attack_es(f, x0_, eps, seed, gens=25, pop=16):
    g = torch.Generator(device=dev).manual_seed(seed); S, D = x0_.shape
    lo_s = torch.minimum(lo, x0_); hi_s = torch.maximum(hi, x0_); best_x, best_s = x0_.clone(), f(x0_)
    for _ in range(gens):
        cand = best_x[:, None, :] + torch.randn(S, pop, D, device=dev, generator=g) * (eps / 2)
        cand = x0_[:, None, :] + (cand - x0_[:, None, :]).clamp(-eps, eps)
        cand = torch.minimum(torch.maximum(cand, lo_s[:, None, :]), hi_s[:, None, :])
        sc = f(cand.reshape(-1, D)).reshape(S, pop); mn, ai = sc.min(1); upd = mn < best_s
        best_s = torch.where(upd, mn, best_s); best_x = torch.where(upd[:, None], cand[torch.arange(S), ai], best_x)
    return best_s
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
def evaluate(m, seed):
    f = fn(m); thr = q95(f(xnor))                                     # test-matched 5% FPR (oracle diagnostic)
    r = dict(AUC=roc_auc_score(y[te], f(Zt[te]).cpu().numpy()), rec_spoof=float((f(xsp) > thr).float().mean()),
             logit_abs=float(m(x0).abs().mean()))
    det0 = f(x0) > thr; n0 = max(int(det0.sum()), 1)
    for eps in (0.5, 1.0):
        r[f"ES@{eps}"]  = float((det0 & (attack_es(f, x0, eps, 1000 + seed) <= thr)).sum() / n0)
        r[f"PGD@{eps}"] = float((det0 & (attack_pgd(m, x0, eps, 7000 + seed) <= thr)).sum() / n0)
    return r

chrono = lambda seed: np.array_split(fit, N_CLIENTS)
iid    = lambda seed: np.array_split(np.random.default_rng(seed).permutation(fit), N_CLIENTS)
SETTINGS = {
 "central 40 epochs":          lambda s: train_central(Z[fit], y[fit], s, 40),
 "central 20 epochs":          lambda s: train_central(Z[fit], y[fit], s, 20),
 "central 10 epochs":          lambda s: train_central(Z[fit], y[fit], s, 10),
 "central 5 epochs":           lambda s: train_central(Z[fit], y[fit], s, 5),
 "FedAvg chrono, E=1, R=40":   lambda s: fedavg(chrono(s), s, 40, 1),
 "FedAvg iid,    E=1, R=40":   lambda s: fedavg(iid(s), s, 40, 1),
 "FedAvg chrono, E=5, R=8":    lambda s: fedavg(chrono(s), s, 8, 5),
 "FedAvg iid,    E=5, R=8":    lambda s: fedavg(iid(s), s, 8, 5),
}
rows = []; t0 = time.time()
for seed in range(6):
    for name, mk in SETTINGS.items():
        rows.append(dict(seed=seed, setting=name, **evaluate(mk(seed), seed)))
    print("seed", seed, "done", round(time.time() - t0), "s", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/expB8_controls.csv", index=False)
g = R.groupby("setting", sort=False)
fmt = lambda c: g[c].mean().round(3).astype(str) + " ± " + g[c].std().round(3).astype(str)
print("\nCLEAN MODELS, 6 seeds (mean ± SD); evasion at test-matched 5% FPR")
print(pd.DataFrame({c: fmt(c) for c in ["AUC", "rec_spoof", "logit_abs", "ES@0.5", "PGD@0.5", "ES@1.0", "PGD@1.0"]}).to_string())