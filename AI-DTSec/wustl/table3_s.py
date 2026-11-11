import os
# ===== 8b: red-team evasion on the twin (data-alteration attacks), sequential vectorized =====
import numpy as np, pandas as pd, torch, torch.nn as nn, json, time
torch.set_num_threads(4)
P = os.environ.get("WUSTL_CSV", "data/wustl-ehms-2020.csv")
import sys; SEED=int(sys.argv[1]); OUT = f"results/wustl/base{SEED}"; os.makedirs(OUT, exist_ok=True)
dev = "cpu"
b = np.load(f"{OUT}/baseline.npz", allow_pickle=True)
mu, sd, thr, te, tr, cols = b["mu"], b["sd"], float(b["thr"]), b["te"], b["tr"], list(b["cols"])
df = pd.read_csv(P); df.columns = [c.strip() for c in df.columns]
y = df["Label"].astype(int).values; cat = df["Attack Category"].astype(str).str.lower().values
X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
Z = ((X - mu) / sd).astype(np.float32)
ae = nn.Sequential(nn.Linear(34,24), nn.ReLU(), nn.Linear(24,12), nn.ReLU(), nn.Linear(12,6),
                   nn.ReLU(), nn.Linear(6,12), nn.ReLU(), nn.Linear(12,24), nn.ReLU(), nn.Linear(24,34)).to(dev)
ae.load_state_dict(torch.load(f"{OUT}/ae_state.pt", map_location=dev)); ae.eval()
@torch.no_grad()
def score(t): return ((ae(t) - t)**2).mean(-1)

trn = tr[y[tr] == 0]
lo = torch.tensor(np.percentile(Z[trn], 0.5, axis=0), device=dev); hi = torch.tensor(np.percentile(Z[trn], 99.5, axis=0), device=dev)
bio = ["Temp","SpO2","Pulse_Rate","SYS","DIA","Heart_rate","Resp_Rate","ST"]
groups = {"biometric (8)": np.array([c in bio for c in cols]),
          "network (26)": np.array([c not in bio for c in cols]),
          "all (34)": np.ones(len(cols), bool)}

S_idx = te[cat[te] == "data alteration"]
x0 = torch.tensor(Z[S_idx], device=dev); s0 = score(x0)
det0 = s0 > thr
print(f"data-alteration test samples: {len(S_idx)} | detected before attack: {int(det0.sum())} ({det0.float().mean():.3f})")
lo_s = torch.minimum(lo, x0); hi_s = torch.maximum(hi, x0)       # stay in normal range, or original value if already outside

def attack(mask, eps, method, budget_gen=25, pop=16, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    m = torch.tensor(mask, device=dev, dtype=torch.float32)
    S, D = x0.shape
    best_x, best_s = x0.clone(), s0.clone()
    for gen in range(budget_gen):
        if method == "random":
            d = (torch.rand(S, pop, D, device=dev, generator=g) * 2 - 1) * eps
            cand = x0[:, None, :] + d * m
        else:                                                    # guided (1+lambda)-ES around the current best
            d = torch.randn(S, pop, D, device=dev, generator=g) * (eps / 2)
            cand = best_x[:, None, :] + d * m
            cand = x0[:, None, :] + (cand - x0[:, None, :]).clamp(-eps, eps) * m
        cand = torch.minimum(torch.maximum(cand, lo_s[:, None, :]), hi_s[:, None, :])
        sc = score(cand.reshape(-1, D)).reshape(S, pop)
        mn, ai = sc.min(1)
        upd = mn < best_s
        best_s = torch.where(upd, mn, best_s)
        best_x = torch.where(upd[:, None], cand[torch.arange(S), ai], best_x)
    return best_s

rows = []
for gname, mask in groups.items():
    for eps in [0.1, 0.25, 0.5, 1.0, 2.0]:
        for method in ["random", "guided"]:
          for aseed in (0,):
            t0 = time.time(); s1 = attack(mask, eps, method, seed=aseed); dt = time.time() - t0
            evaded = (det0 & (s1 <= thr)).sum().item() / max(int(det0.sum()), 1)
            rows.append(dict(group=gname, eps=eps, method=method, seed=aseed, detected_after=(s1 > thr).float().mean().item(),
                             evasion_rate=evaded, seconds=round(dt, 2)))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/table3_3seeds.csv", index=False)
print("\nEVASION RATE (share of originally detected alterations that fall below the threshold)")
print(res.pivot_table(index=["group", "eps"], columns="method", values="evasion_rate", aggfunc=["mean","std"]).round(3).to_string())
print("\nRECALL AFTER ATTACK (all alteration samples; before attack = %.3f)" % det0.float().mean().item())
print(res.pivot_table(index=["group", "eps"], columns="method", values="detected_after", aggfunc=["mean","std"]).round(3).to_string())