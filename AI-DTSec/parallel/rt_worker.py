import os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); G = {}
def build():
    return nn.Sequential(nn.Linear(34,24), nn.ReLU(), nn.Linear(24,12), nn.ReLU(), nn.Linear(12,6), nn.ReLU(),
                         nn.Linear(6,12), nn.ReLU(), nn.Linear(12,24), nn.ReLU(), nn.Linear(24,34))
def init():
    torch.set_num_threads(1)
    d = np.load(os.path.join(HERE, "rt_data.npz")); G["x0"] = torch.tensor(d["x0"]); G["lo"] = torch.tensor(d["lo"]); G["hi"] = torch.tensor(d["hi"])
    G["thr"] = float(d["thr"]); G["masks"] = {k: torch.tensor(d["mask_" + k], dtype=torch.float32) for k in ("biometric", "network", "all")}
    m = build(); m.load_state_dict(torch.load(os.path.join(HERE, "rt_ae.pt"), map_location="cpu")); m.eval(); G["ae"] = m
def run(task):
    group, eps, method, seed, gens = task
    m = G["ae"]; x0 = G["x0"]; lo, hi = G["lo"], G["hi"]; mk = G["masks"][group]
    gen = torch.Generator().manual_seed(seed); S, D = x0.shape
    def score(t):
        with torch.no_grad(): return ((m(t) - t) ** 2).mean(-1)
    lo_s = torch.minimum(lo, x0); hi_s = torch.maximum(hi, x0); best_x, best_s = x0.clone(), score(x0); s0 = best_s.clone()
    for _ in range(gens):
        if method == "random":
            cand = x0[:, None, :] + (torch.rand(S, 16, D, generator=gen) * 2 - 1) * eps * mk
        else:
            cand = best_x[:, None, :] + torch.randn(S, 16, D, generator=gen) * (eps / 2) * mk
            cand = x0[:, None, :] + (cand - x0[:, None, :]).clamp(-eps, eps) * mk
        cand = torch.minimum(torch.maximum(cand, lo_s[:, None, :]), hi_s[:, None, :])
        sc = score(cand.reshape(-1, D)).reshape(S, 16); mn, ai = sc.min(1); upd = mn < best_s
        best_s = torch.where(upd, mn, best_s); best_x = torch.where(upd[:, None], cand[torch.arange(S), ai], best_x)
    det0 = s0 > G["thr"]
    return task, float((det0 & (best_s <= G["thr"])).sum() / max(int(det0.sum()), 1))
def burn(n):
    s = 0
    for i in range(n): s += i * i % 7
    return s
