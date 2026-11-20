# CIPHER

**Consensus-informed propagation for multi-source graph-knowledge fusion in
autism-related interaction reasoning.**

CIPHER integrates source-specific protein interaction graphs with frozen ASD
priors and pathway annotations. The implementation contains source-specific
Chebyshev encoding, Consensus Spectral Source Transport (CSST), ASD-conditioned
structured propagation, and a source-factor relational decoder.

This repository contains the complete model, the two fixed processed benchmark
datasets, a single-fold training command, checkpoint-based prediction, and small
software tests. Pretrained checkpoints are not distributed.

## Installation

Use Python 3.8-3.11. The reference environment uses Python 3.8, PyTorch 2.3.0,
PyTorch Geometric 2.6.1, and CUDA 11.8. Install a PyTorch build appropriate for
your hardware before installing the package when using a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Pinned dependencies are also listed in `requirements.txt`. Run the following
commands from the repository root.

## Data

The supplied `data/strict/autdb` and `data/strict/sfari` directories contain all
processed inputs consumed by the model, including frozen annotations and the
five validation folds and independent test sets. Use the supplied splits for
reproducibility. BioGRID contributes background graph evidence to both datasets.

```bash
python -m cipher.check_data --dataset-dir data/strict/autdb
python -m cipher.check_data --dataset-dir data/strict/sfari
sha256sum -c data/SHA256SUMS
```

These commands check fixed-split consistency and verify that supervised pairs
are excluded from the propagation graph. See [Data documentation](docs/DATA.md)
for schemas, identifiers, annotation provenance, and third-party data terms.

## Minimal Software Test

```bash
OMP_NUM_THREADS=2 python -m pytest
```

The tests use a synthetic graph, run a short train/save/load/predict pipeline,
verify undirected pair symmetry and contribution reconstruction, and check finite
gradients. All tests run on the CPU.

## Train

```bash
python -m cipher.train \
  --dataset-dir data/strict/autdb --fold 1 \
  --config configs/cipher.yaml --output-dir outputs/autdb_fold1 --device cuda
```

For SFARI, use `data/strict/sfari` and a separate output directory. `--fold` selects
an existing fold from 1 to 5; it never repartitions the data. `--device cpu` is
supported, but full-graph training on more than one million edges is intended
for a GPU. Resource needs depend on graph size and representation width.

The default configuration uses field width 224, token width 56, Chebyshev order
4, two local layers, three operator steps, and 70 maximum epochs. Training uses
validation AUC for early stopping and selects an F1 threshold on that fold's
validation set. Independent-test labels are not used for model selection.
The command writes `cipher.pt` and `validation_metrics.json` to its output
directory. The checkpoint records the configuration, seed, fold, and selected
threshold.

## Predict

An input CSV contains `left,right` node indices from the same dataset. The
provided independent-test CSV can be used directly:

```bash
python -m cipher.predict \
  --dataset-dir data/strict/autdb \
  --checkpoint outputs/autdb_fold1/cipher.pt \
  --pairs data/strict/autdb/split_test.csv \
  --output outputs/autdb_fold1/test_predictions.csv \
  --representations outputs/autdb_fold1/test_representations.npz --device cuda
```

For protein symbols, provide `left,right` symbol columns and add
`--pair-format symbols`. Symbols must match the dataset exactly. Predictions use
the fixed training graph; pairs are not added to, or removed from, it at
inference. The CLI checks that the checkpoint and dataset names agree.

The prediction CSV preserves input columns and adds `probability`, `logit`, and
`prediction`, using the validation-selected threshold. Optional NPZ output
contains pairs, probabilities, logits, the final pre-classifier representation,
source-factor contributions, and labels if supplied. Contributions have shape
`[number_of_pairs, 4, 7]`. Probabilities are model scores under the supplied
training distribution, not experimentally verified disease associations.

## Organization

```text
configs/cipher.yaml            Default configuration
src/cipher/models/
  source_encoder.py           Source-specific spectral encoding
  csst.py                     Consensus spectral source transport
  structured_operator.py      ASD-conditioned propagation
  relational_decoder.py       Tokenization and symmetric pair scoring
  cipher.py                   Complete model
src/cipher/data.py             Fixed-graph/fold loader
src/cipher/losses.py           Classification, field, and source objectives
src/cipher/train.py            Single-fold training
src/cipher/predict.py          Cached-graph pair inference
src/cipher/check_data.py       Data-integrity checks
data/strict/                  AutDB and SFARI processed inputs
tests/                        Minimal synthetic software tests
docs/                         Method notation and data documentation
```

See [Method-to-code correspondence](docs/METHOD.md) for equations, symbols,
tensor shapes, and objective terms. Software is covered by `LICENSE`; biological
data retain the terms of their original providers.
