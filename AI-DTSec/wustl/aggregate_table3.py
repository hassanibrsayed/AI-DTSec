# Combines the per-seed red-team results (results/wustl/base{0..4}/table3_3seeds.csv) into results/wustl/table3_5detseeds.csv
import pandas as pd
R = pd.concat([pd.read_csv(f"results/wustl/base{s}/table3_3seeds.csv").assign(det=s) for s in range(5)], ignore_index=True)
R.to_csv("results/wustl/table3_5detseeds.csv", index=False); print(R.groupby(["group", "eps", "method"]).evasion_rate.agg(["mean", "std"]).round(3))
