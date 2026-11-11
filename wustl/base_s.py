# ===== 8a: Paper 2 detector baseline (self-contained) =====
import os, numpy as np, pandas as pd, torch, torch.nn as nn, json
torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
from sklearn.metrics import roc_auc_score
P = os.environ.get("WUSTL_CSV", "data/wustl-ehms-2020.csv")
import sys; SEED=int(sys.argv[1]); OUT = f"results/wustl/base{SEED}"; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(P); df.columns = [c.strip() for c in df.columns]
y = df["Label"].astype(int).values; N = len(y)
cat = df["Attack Category"].astype(str).str.lower().values
ids = ["Dir","Flgs","SrcAddr","DstAddr","Sport","Dport","SrcMac","DstMac","Packet_num"]
cols = [c for c in df.columns if c not in ids+["Label","Attack Category"]]
X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
print("features:", len(cols), "| attack categories:", pd.Series(cat).value_counts().to_dict())

# leakage-aware split: 20 contiguous blocks, test = blocks 3,7,11,15,19, purge 100 samples around test blocks
edges = np.linspace(0, N, 21).astype(int); test_blocks = [3, 7, 11, 15, 19]
te = np.concatenate([np.arange(edges[b], edges[b+1]) for b in test_blocks])
banned = np.zeros(N, bool)
for b in test_blocks: banned[max(0, edges[b]-100):min(N, edges[b+1]+100)] = True
tr = np.where(~banned)[0]
trn = tr[y[tr] == 0]                     # detector sees normal training samples only
print(f"train samples {len(tr)} (normal used {len(trn)}) | test {len(te)} (attacks {int(y[te].sum())})")

mu, sd = X[trn].mean(0), X[trn].std(0) + 1e-6
Z = (X - mu) / sd
torch.manual_seed(SEED); dev = "cpu"
ae = nn.Sequential(nn.Linear(34,24), nn.ReLU(), nn.Linear(24,12), nn.ReLU(), nn.Linear(12,6),
                   nn.ReLU(), nn.Linear(6,12), nn.ReLU(), nn.Linear(12,24), nn.ReLU(), nn.Linear(24,34)).to(dev)
A = torch.tensor(Z[trn], device=dev); opt = torch.optim.Adam(ae.parameters(), 1e-3)
for ep in range(80):
    perm = torch.randperm(len(A), device=dev)
    for i in range(0, len(A), 128):
        idx = perm[i:i+128]; opt.zero_grad(); nn.functional.mse_loss(ae(A[idx]), A[idx]).backward(); opt.step()
def score(Zm):
    with torch.no_grad():
        t = torch.tensor(Zm, dtype=torch.float32, device=dev); return ((ae(t) - t)**2).mean(1).cpu().numpy()
thr = float(np.percentile(score(Z[trn]), 95))        # 95th percentile of normal training error
s = score(Z[te]); yt = y[te]; ct = cat[te]
print(f"\nthreshold (95th pct, normal train) = {thr:.4f}")
print(f"hold-out AUC-ROC = {roc_auc_score(yt, s):.3f} | false-positive rate on normal = {(s[yt==0] > thr).mean():.3f}")
for c in sorted(set(ct)):
    if c in ("normal", "nan"): continue
    m = ct == c
    print(f"  {c:18s} n={m.sum():4d} | detected (recall) = {(s[m] > thr).mean():.3f} | AUC vs normal = {roc_auc_score(np.r_[np.zeros((yt==0).sum()), np.ones(m.sum())], np.r_[s[yt==0], s[m]]):.3f}")
# save artifacts for 8b
torch.save(ae.state_dict(), f"{OUT}/ae_state.pt")
np.savez(f"{OUT}/baseline.npz", mu=mu, sd=sd, thr=thr, te=te, tr=tr, cols=np.array(cols))
json.dump({"thr": thr}, open(f"{OUT}/baseline.json", "w"))
print("saved to", OUT)