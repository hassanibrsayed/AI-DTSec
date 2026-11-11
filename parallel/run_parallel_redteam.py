"""Parallel red-team grid benchmark for AI-DTSec (Paper 2, Section 5.6).
Run:  python run_parallel_redteam.py            (full: 90 scenarios x 200 generations, 3 trials)
      python run_parallel_redteam.py --quick    (short test)
Needs: Python 3.9+, numpy, pandas, torch (CPU is enough). Keep rt_worker.py, rt_data.npz, rt_ae.pt in the same folder.
Close other heavy programs and keep the laptop on mains power while it runs."""
import os, sys, time, platform, statistics, multiprocessing as mp
import pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import rt_worker as W
def main():
    quick = "--quick" in sys.argv; gens = 20 if quick else 200; trials = 1 if quick else 3
    groups = ["biometric", "network", "all"]; eps_list = [0.1, 0.25, 0.5, 1.0, 2.0]
    tasks = [(g, e, m, s, gens) for g in groups for e in eps_list for m in ["random", "guided"] for s in range(3)]
    ncpu = os.cpu_count(); phys = None
    try:
        import psutil; phys = psutil.cpu_count(logical=False); ncpu = len(psutil.Process().cpu_affinity())
    except Exception: pass
    tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), None)
    print(f"machine: {platform.processor() or platform.machine()} | logical cores {ncpu} | physical cores {phys} | torch {torch.__version__}")
    print(f"scenarios {len(tasks)} x {gens} generations, {trials} trial(s) per setting", flush=True)
    ctx = mp.get_context("spawn")
    # 1) does the machine scale at all? (pure-Python CPU burn)
    N = 4_000_000; t0 = time.time(); [W.burn(N) for _ in range(8)]; tb = time.time() - t0
    burn_rows = [dict(workers=1, seconds=tb, speedup=1.0)]
    for w in [2, 4, 8]:
        if w > ncpu: continue
        with ctx.Pool(w) as p:
            p.map(W.burn, [1000] * w); t0 = time.time(); p.map(W.burn, [N] * 8, chunksize=1); dt = time.time() - t0
        burn_rows.append(dict(workers=w, seconds=dt, speedup=tb / dt)); print(f"CPU check, {w} processes: speedup {tb/dt:.2f}", flush=True)
    # 2) red-team grid: sequential baseline (one process, one thread)
    W.init(); torch.set_num_threads(1); W.run(tasks[0])
    seq_times = []
    for _ in range(trials):
        t0 = time.time(); seq = [W.run(t) for t in tasks]; seq_times.append(time.time() - t0)
    t_seq = statistics.median(seq_times); print(f"sequential: {t_seq:.1f} s (trials: {', '.join(f'{t:.1f}' for t in seq_times)})", flush=True)
    rows = [dict(workers=0, seconds=t_seq, speedup=1.0, efficiency=1.0, identical=True, startup_s=0.0)]
    for w in [1, 2, 4, 8, 12, 16, 24]:
        if w > ncpu: continue
        times = []; same = True
        t0 = time.time(); pool = ctx.Pool(w, initializer=W.init); pool.map(W.run, tasks[:w], chunksize=1); startup = time.time() - t0
        for _ in range(trials):
            t0 = time.time(); par = pool.map(W.run, tasks, chunksize=1); times.append(time.time() - t0)
            same = same and all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(seq, par))
        pool.close(); pool.join(); dt = statistics.median(times)
        rows.append(dict(workers=w, seconds=dt, speedup=t_seq / dt, efficiency=t_seq / dt / w, identical=same, startup_s=startup))
        print(f"{w:2d} workers: {dt:.1f} s | speedup {t_seq/dt:.2f} | efficiency {t_seq/dt/w:.2f} | identical results: {same} | pool start-up {startup:.1f} s", flush=True)
    # sequential again at the end, to detect drift (thermal throttling, core scheduling)
    W.init(); torch.set_num_threads(1); t0 = time.time(); [W.run(t) for t in tasks]; t_end = time.time() - t0
    print(f"sequential again at the end: {t_end:.1f} s", flush=True)
    rows.append(dict(workers=-1, seconds=t_end, speedup=t_seq / t_end, efficiency=float("nan"), identical=True, startup_s=0.0))
    out = pd.DataFrame(rows); tag = tag or ("quick" if quick else "full")
    out.to_csv(os.path.join(HERE, f"parallel_redteam_{tag}.csv"), index=False)
    pd.DataFrame(burn_rows).to_csv(os.path.join(HERE, f"cpu_check_{tag}.csv"), index=False)
    pd.DataFrame([dict(task=f"{t[0]}|{t[1]}|{t[2]}|{t[3]}", evasion=e) for t, e in seq]).to_csv(os.path.join(HERE, f"grid_results_{tag}.csv"), index=False)
    print("\n" + out.round(3).to_string(index=False)); print(f"\nSaved results in {HERE}")
if __name__ == "__main__":
    mp.freeze_support(); main()
