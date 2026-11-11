# Builds rt_data.npz and rt_ae.pt for the parallel red-team benchmark from the seed-0 baseline
# (run "python wustl/base_s.py 0" first). Usage, from the repository root: python parallel/make_rt_data.py
import os, shutil, numpy as np, pandas as pd
P = os.environ.get("WUSTL_CSV", "data/wustl-ehms-2020.csv"); B = "results/wustl/base0"; HERE = os.path.dirname(os.path.abspath(__file__))
b = np.load(f"{B}/baseline.npz", allow_pickle=True)
mu, sd, thr, te, tr, cols = b["mu"], b["sd"], float(b["thr"]), b["te"], b["tr"], list(b["cols"])
df = pd.read_csv(P); df.columns = [c.strip() for c in df.columns]
y = df["Label"].astype(int).values; cat = df["Attack Category"].astype(str).str.lower().values
X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
Z = ((X - mu) / sd).astype(np.float32)
trn = tr[y[tr] == 0]; lo = np.percentile(Z[trn], 0.5, axis=0).astype(np.float32); hi = np.percentile(Z[trn], 99.5, axis=0).astype(np.float32)
x0 = Z[te[(y[te] == 1) & (cat[te] == "data alteration")]]
bio = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]
mb = np.array([c in bio for c in cols], np.float32)
np.savez(os.path.join(HERE, "rt_data.npz"), x0=x0, lo=lo, hi=hi, thr=np.float32(thr), mask_biometric=mb, mask_network=1 - mb, mask_all=np.ones_like(mb))
shutil.copy(f"{B}/ae_state.pt", os.path.join(HERE, "rt_ae.pt")); print("x0", x0.shape, "thr", thr)
