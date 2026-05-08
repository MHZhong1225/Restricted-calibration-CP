# SGCP

This repository contains a minimal, runnable implementation of SGCP. The release
keeps only the proposed method and the code needed to run it on the supported
`syn`, `mimic`, `nursery`, and `bach` datasets.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use conda:

```bash
conda activate cp
pip install -r requirements.txt
```

## Run

For a quick smoke test:

```bash
conda activate cp
python run.py --grid-json json/sgcp_syn.json --outdir results_syn
python run.py --grid-json json/sgcp_mimic.json --mimic-preprocessed-path path/to/mimic_iv_processed.csv --outdir results_mimic
python run.py --grid-json json/sgcp_nursery.json --nursery-csv-path path/to/nursery.csv --outdir results_nursery
python run.py --grid-json json/sgcp_bach.json --bach-data-dir dataset/bach --outdir results_bach
```

The paper hyperparameter presets are stored in:

```text
json/sgcp_bach.json
json/sgcp_mimic.json
json/sgcp_nursery.json
json/sgcp_syn.json
```

Run a preset with:

```bash
python run.py --grid-json json/sgcp_syn.json --outdir results_syn
```

For real datasets, provide the paths when they are not at the default locations:

```bash
python run.py --grid-json json/sgcp_mimic.json --mimic-preprocessed-path path/to/mimic-iv_processed.csv
python run.py --grid-json json/sgcp_nursery.json --nursery-csv-path path/to/nursery.csv
python run.py --grid-json json/sgcp_bach.json --bach-data-dir path/to/bach_imagefolder
```

Default dataset paths are:

```text
dataset/mimic/mimic-iv_processed.csv
dataset/nursery/nursery.csv
dataset/bach
```

BACH expects the fixed class names `Benign`, `InSitu`, `Invasive`, and `Normal`.
Both layouts produced by `dataset/prepare_bach.py` are supported:

```text
flat:  dataset/bach/Benign/*.png
split: dataset/bach/train/Benign/*.png
```

To prepare BACH from the original class folders:

```bash
python dataset/prepare_bach.py --source-dir path/to/original_bach --target-dir dataset/datasets/bach --layout flat
```

The script prints SGCP metrics and appends them to `summary.csv` inside the chosen output directory.
