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
from sklearn.metrics import roc_auc_score
tr = np.sort(tr); nval = len(tr) // 5
val, fit = tr[-nval:], tr[:-nval]; fit = fit[fit < val.min() - 100]; trn = fit[y[fit] == 0]
lo = torch.tensor(np.percentile(Z[trn], 0.5, axis=0), device=dev); hi = torch.tensor(np.percentile(Z[trn], 99.5, axis=0), device=dev)
PW = float((1 - y[fit].mean()) / y[fit].mean())
alt_te = te[cat[te] == "data alteration"]; spo_te = te[cat[te] == "spoofing"]; nor_te = te[y[te] == 0]; alt_fit = fit[cat[fit] == "data alteration"]
x0 = Zt[alt_te]

def train_clf(Xt, yt, seed, epochs=40):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(34,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1)).to(dev)
    A = torch.tensor(Xt, device=dev); t = torch.tensor(yt, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3); pw = torch.tensor(PW, device=dev)
    for ep in range(epochs):
        perm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 256):
            idx = perm[i:i+256]; opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(A[idx]).squeeze(1), t[idx], pos_weight=pw).backward(); opt.step()
    m.eval()
    return m, (lambda T: torch.sigmoid(m(T).squeeze(-1)).detach())

def attack_es(fn, x0, eps, seed, gens=25, pop=16, sigma_div=2.0):
    g = torch.Generator(device=dev).manual_seed(seed); S, D = x0.shape
    lo_s = torch.minimum(lo, x0); hi_s = torch.maximum(hi, x0); best_x, best_s = x0.clone(), fn(x0)
    for _ in range(gens):
        cand = best_x[:, None, :] + torch.randn(S, pop, D, device=dev, generator=g) * (eps / sigma_div)
        cand = x0[:, None, :] + (cand - x0[:, None, :]).clamp(-eps, eps)
        cand = torch.minimum(torch.maximum(cand, lo_s[:, None, :]), hi_s[:, None, :])
        sc = fn(cand.reshape(-1, D)).reshape(S, pop); mn, ai = sc.min(1); upd = mn < best_s
        best_s = torch.where(upd, mn, best_s); best_x = torch.where(upd[:, None], cand[torch.arange(S), ai], best_x)
    return best_s, best_x

def attack_random(fn, x0, eps, seed, gens=25, pop=16):
    g = torch.Generator(device=dev).manual_seed(seed); S, D = x0.shape
    lo_s = torch.minimum(lo, x0); hi_s = torch.maximum(hi, x0); best_x, best_s = x0.clone(), fn(x0)
    for _ in range(gens):
        cand = x0[:, None, :] + (torch.rand(S, pop, D, device=dev, generator=g) * 2 - 1) * eps
        cand = torch.minimum(torch.maximum(cand, lo_s[:, None, :]), hi_s[:, None, :])
        sc = fn(cand.reshape(-1, D)).reshape(S, pop); mn, ai = sc.min(1); upd = mn < best_s
        best_s = torch.where(upd, mn, best_s); best_x = torch.where(upd[:, None], cand[torch.arange(S), ai], best_x)
    return best_s, best_x

def attack_pgd(model, x0, eps, seed, steps=50, restarts=2):
    g = torch.Generator(device=dev).manual_seed(seed)
    lo_s = torch.minimum(lo, x0); hi_s = torch.maximum(hi, x0); alpha = eps / 10
    best_x = x0.clone(); best_l = model(x0).squeeze(-1).detach()
    for r in range(restarts):
        x = x0 + (torch.rand(x0.shape, device=dev, generator=g) * 2 - 1) * eps * float(r > 0)
        x = torch.minimum(torch.maximum(x, lo_s), hi_s)
        for _ in range(steps):
            x = x.detach().requires_grad_(True); model(x).squeeze(-1).sum().backward()
            with torch.no_grad():
                x = x - alpha * x.grad.sign(); x = x0 + (x - x0).clamp(-eps, eps); x = torch.minimum(torch.maximum(x, lo_s), hi_s)
                l = model(x).squeeze(-1); upd = l < best_l; best_l = torch.where(upd, l, best_l); best_x = torch.where(upd[:, None], x, best_x)
    return torch.sigmoid(best_l), best_x

q95 = lambda a: float(np.percentile(a.cpu().numpy(), 95))
EPS = [0.25, 0.5, 1.0]; rows = []; ops = []; t0 = time.time()
for seed in range(3):
    m0, f0 = train_clf(Z[fit], y[fit], seed)
    adv = np.concatenate([attack_es(f0, Zt[alt_fit], e, 200 + seed*10 + k)[1].cpu().numpy() for k, e in enumerate(EPS)])
    m1, f1 = train_clf(np.vstack([Z[fit], adv]), np.r_[y[fit], np.ones(len(adv), dtype=y.dtype)], seed)
    adv2 = np.concatenate([attack_es(f1, Zt[alt_fit], e, 300 + seed*10 + k)[1].cpu().numpy() for k, e in enumerate(EPS)])
    m2, f2 = train_clf(np.vstack([Z[fit], adv, adv2]), np.r_[y[fit], np.ones(len(adv)+len(adv2), dtype=y.dtype)], seed)
    sur_m, _ = train_clf(Z[fit], y[fit], seed + 100)
    for rname, (m, f) in {"round 0 (clean)": (m0, f0), "round 1": (m1, f1), "round 2": (m2, f2)}.items():
        thr_val  = q95(f(Zt[val[y[val] == 0]]))          # validation-calibrated (as in 8d / 9b)
        thr_test = q95(f(Zt[nor_te]))                    # oracle: exactly 5% FPR on the test normals
        for tname, thr in (("validation-calibrated", thr_val), ("test-matched 5% FPR", thr_test)):
            ops.append(dict(seed=seed, round=rname, threshold=tname, AUC=roc_auc_score(y[te], f(Zt[te]).cpu().numpy()),
                            recall_spoof=float((f(Zt[spo_te]) > thr).float().mean()),
                            test_fpr=float((f(Zt[nor_te]) > thr).float().mean()),
                            recall_alt=float((f(x0) > thr).float().mean())))
        for eps in EPS:
            res = {"ES (training attacker)": attack_es(f, x0, eps, 1000 + seed)[0],
                   "Random search": attack_random(f, x0, eps, 2000 + seed)[0],
                   "ES 5x budget, sigma=eps/3": attack_es(f, x0, eps, 5000 + seed, gens=125, sigma_div=3.0)[0],
                   "PGD white-box": attack_pgd(m, x0, eps, 7000 + seed)[0],
                   "PGD transfer (surrogate)": f(attack_pgd(sur_m, x0, eps, 8000 + seed)[1])}
            for tname, thr in (("validation-calibrated", thr_val), ("test-matched 5% FPR", thr_test)):
                det0 = f(x0) > thr; n0 = max(int(det0.sum()), 1)
                for name, s1 in res.items():
                    rows.append(dict(seed=seed, round=rname, eps=eps, attacker=name, threshold=tname,
                                     evasion=float((det0 & (s1 <= thr)).sum() / n0)))
    print("seed", seed, "done", round(time.time() - t0), "s", flush=True)

R = pd.DataFrame(rows); R.to_csv(f"{OUT}/expC_attackers.csv", index=False)
O = pd.DataFrame(ops); O.to_csv(f"{OUT}/expC_ops.csv", index=False)
print("\nOPERATING POINTS (mean over seeds)")
print(O.groupby(["threshold", "round"])[["test_fpr", "recall_alt"]].agg(["mean", "std"]).round(3).to_string())
print("\nEVASION (mean, std over seeds)")
print(R.groupby(["attacker", "eps", "threshold", "round"]).evasion.agg(["mean", "std"]).round(3)
        .unstack(["threshold", "round"]).to_string())