# AI-DTSec

Code and results for the paper

> **AI-DTSec: Adversarial Red-Teaming, Detector Hardening and Parallel Red-Team Campaigns in Healthcare Digital Twins**
> Nabila A. Khodeir, Ghada F. Elkabbany, Samah O. Kamel, Hassan I. Ahmed

The repository reproduces the experiments on WUSTL-EHMS-2020 (Sections 4-5.6 of the paper) and the replication on CICIoMT2024 (Section 5.7): the label-free autoencoder baseline, the gradient-free red-team agent, adversarial hardening at a matched false-positive rate, the attackers never used in training, the training-budget and federated controls, early stopping, the parallel red-team campaign grid, and the CICIoMT2024 hardening procedures (fixed adversarial samples, PGD samples and on-the-fly adversarial training).

## Setup

```bash
pip install -r requirements.txt
```

Everything runs on CPU. The scripts set `torch.use_deterministic_algorithms(True)`, so a rerun with the same seeds gives the same numbers.

Download the datasets as described in [`data/README.md`](data/README.md). **Run every command from the repository root.** Results are written to `results/` and figures to `figures/`.

## What produces what

| Paper | Command(s) | Output |
|---|---|---|
| Table 2, Fig. 3 (autoencoder baseline, 5 seeds) | `for s in 0 1 2 3 4; do python wustl/base_s.py $s; done` then `python wustl/table2_seeds.py` | `results/wustl/base{s}/`, `table2_5seeds.csv` |
| Table 3, Fig. 4 (red-team evasion of the autoencoder) | `for s in 0 1 2 3 4; do python wustl/table3_s.py $s; done` then `python wustl/aggregate_table3.py` | `table3_5detseeds.csv` |
| Figs. 3-4 | `python wustl/fig34.py` | `figures/fig3.png`, `fig4.png` |
| Tables 4-7 (hardening, independent attackers, test-matched threshold) | `python wustl/expC.py` | `expC_attackers.csv`, `expC_ops.csv` |
| Table 8 (training budget, federated controls, mean absolute logit) | `python wustl/exp8.py` | `expB8_controls.csv` |
| Table 9 (hardening by training budget and mode, early stopping) | `python wustl/expA.py` and `python wustl/expA_local10.py` | `expA_early_stopping.csv`, `expA_chosen_epochs.csv`, `expA_local10.csv` |
| Figs. 5-7 | `python wustl/figs.py` | `figures/fig5.png`-`fig7.png` |
| Table 10, Fig. 8 (parallel red-team grid) | `python parallel/make_rt_data.py` (needs `base_s.py 0`), then `python parallel/run_parallel_redteam.py`, then `python parallel/fig_par.py` | `parallel/parallel_redteam_full.csv`, `figures/fig_par.png` |
| Tables 13-14 (CICIoMT2024 autoencoder and red team) | `python ciciomt/cic_exp.py A` and `python ciciomt/cic_v2.py A` | `results/ciciomt/cic_baseline.csv`, `cic_redteam.csv`, `v2/A_redteam.csv` |
| Table 15, Fig. 11 (CICIoMT2024 hardening) | `python ciciomt/cic_v2.py ES 0`, `PGD 0`, `MADRY 0`, `ES 1`, `PGD 1`, `MADRY 1`, then `python ciciomt/cic_v2_tables.py` and `python ciciomt/fig_cic2.py` | `results/ciciomt/v2/` |
| 40-epoch check in Section 5.7 | `python ciciomt/cic_exp.py B_40` (and `B_ES`) | `results/ciciomt/cic_hardening_*.csv` |
| Figs. 1-2 (architecture and hardening loop) | `python diagrams/fig12_bw.py` | `figures/fig1_arch_bw.png`, `fig2_loop_bw.png` |

In `cic_v2.py`, the second argument is the threat model: `0` = all 43 features may change, `1` = the 23 protocol and TCP-flag features are frozen.

## Parallel benchmarks

`parallel/run_parallel_redteam.py` is self-contained once `rt_data.npz` and `rt_ae.pt` have been built with `make_rt_data.py`. It runs the 90-scenario grid (`full`; `python parallel/run_parallel_redteam.py --quick` for a short test) sequentially and with 1, 2, 4, 8, ... workers, checks that every parallel run returns the same results as the sequential run, and runs a CPU-bound check of how much parallelism the machine provides. `fig_par.py` plots `results/parallel/parallel_redteam_pcores.csv`, the run reported in the paper; copy your own `parallel/parallel_redteam_full.csv` there to plot it instead. Table 10 was measured on a laptop with an Intel Core i9-13980HX, with the process restricted to the 16 logical CPUs of the 8 performance cores (on Windows: `start "" /wait /high /affinity FFFF python parallel/run_parallel_redteam.py`).

The shared-memory scoring benchmark and the training-workload benchmark (Tables 11-12, Figs. 9-10) were run in separate Google Colab sessions and are not part of this repository.

`notebooks/AI-DTSec_colab.ipynb` is the original exploratory Colab notebook (sections 8a-9c-3), kept with its outputs for reference; it reads the data from Google Drive (`/content/drive/MyDrive/AI-DTSec/...`), so adjust the paths before running it. It includes the red-team grid on a 2-core Colab runtime (1.07x). The scripts above are the cleaned, deterministic versions used for the reported numbers.

## Results

`results/` contains the CSV files behind every table, so the tables and figures can be checked or redrawn without rerunning the experiments. The `results/parallel/*_2core_container.csv` files are the run of the grid on a 2-core cloud container (2.06x) mentioned in Section 5.6.

## Data

The datasets are not redistributed. WUSTL-EHMS-2020 is available from Washington University in St. Louis, and CICIoMT2024 from the Canadian Institute for Cybersecurity (see `data/README.md`).
