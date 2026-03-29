# Efficient Logistic Regression with Mixture of Sigmoids ß

This repository is reduced to the code and data needed to reproduce the paper "Efficient Logistic Regression with Mixture of Sigmoids" figures based on:

- average prequential log-loss
- the AIOLI/EW `B`-sweep

## Requirements

Use a Python 3 environment with the packages listed in `requirements.txt`.

## Kept scripts

- `a9a_experiment_new.py`: runs the main online comparison and saves raw results.
- `b_sweep_aioli_ew.py`: runs the `B`-sweep and saves raw results.
- `replot_tuned_only_avg_logloss.py`: builds the final 3-panel average log-loss figure.
- `replot_b_sweep_all3.py`: builds the final 3-panel `B`-sweep figure.
- `libsvm_datasets.py`: dataset loading and caching utilities.

## Main figure workflow

Run the per-dataset experiments:

```bash
python3 a9a_experiment_new.py --dataset a9a --split train --max-rounds 2000 --repeats 5 --unregularized-ogd-ons
python3 a9a_experiment_new.py --dataset w8a --split train --max-rounds 2000 --repeats 5 --unregularized-ogd-ons
python3 a9a_experiment_new.py --dataset ijcnn1 --split train --max-rounds 2000 --repeats 5 --unregularized-ogd-ons
```

Build the final combined figure:

```bash
python3 replot_tuned_only_avg_logloss.py
```

## B-sweep workflow

Run the per-dataset sweeps:

```bash
python3 b_sweep_aioli_ew.py --dataset a9a --split train --max-rounds 1000 --repeats 1
python3 b_sweep_aioli_ew.py --dataset w8a --split train --max-rounds 1000 --repeats 1
python3 b_sweep_aioli_ew.py --dataset ijcnn1 --split train --max-rounds 1000 --repeats 1
```

Build the final combined figure:

```bash
python3 replot_b_sweep_all3.py
```

## Outputs

Raw result bundles are written to `figs_online_mala_comparable/` as `.npz` and `.json`.
The two replot scripts write the final paper-ready figures (`.pdf` and `.pgf`) into the same directory.
