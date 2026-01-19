# MIR: Model Inferred Replica

**MIR (Model Inferred Replica)** is a component of the **MIGATE** framework (Model Inferred Replica for Adversarial Generalized Evasion), focusing on the model extraction phase for malware classifiers and antivirus systems.

MIR provides a comprehensive framework for extracting surrogate models from black-box malware classifiers and antivirus systems using active learning techniques. The extracted models can achieve high agreement with target models while maintaining low false positive rates.

## Features

- **Multiple Target Support**: Extract models from EMBER-based classifiers, SOREL-20M models (FCNN, LightGBM), and commercial antivirus systems (AV1-AV4)
- **Active Learning Strategies**: Entropy-based sampling, random sampling, medoids clustering, MC-Dropout uncertainty, K-center greedy, and ensemble-based methods
- **Surrogate Model Types**: Deep Neural Networks (DNN, dualDNN), LightGBM, and Support Vector Machines (SVM)
- **Feature Space Independence**: Automatic feature alignment for scenarios where target and attacker use different feature spaces
- **Black-Box Compliance**: Oracle interface hides all implementation details, requiring only model name and raw features

## Installation

### Prerequisites

- Python 3.8+
- pip
- Git

### Quick Setup

```bash
# Clone repository
git clone git@github.com:hytong05/model-stealing.git
cd model-stealing

# Run setup script
./setup_venv.sh

# Activate virtual environment
source venv/bin/activate
```

The setup script will:
1. Check Python version
2. Create virtual environment in `venv/`
3. Install dependencies from `requirements.txt`
4. Install `ember` package from GitHub

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/endgameinc/ember.git
```

## Quick Start

### Basic Model Extraction

```bash
python scripts/attacks/model_extraction.py \
  --data_dir /path/to/data \
  --dataset ember \
  --target_model sorel-FCNN \
  --type LGB \
  --method medoids \
  --budget 2500 \
  --num_queries 10 \
  --log_dir /path/to/logs \
  --seed 42
```

### Oracle Query Interface

```python
from src.targets.oracle_client import create_oracle_from_name
import numpy as np

# Initialize oracle (only need model name)
oracle = create_oracle_from_name("LEE")  # or "CEE", "CSE", "LSE"

# Query with raw features
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)  # Returns 0 (Benign) or 1 (Malware)
```

## Usage

### Model Extraction Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--data_dir` | Directory containing dataset | Path |
| `--dataset` | Dataset type | `ember`, `sorel`, `AV` |
| `--target_model` | Target model to extract | `ember`, `sorel-FCNN`, `sorel-LGB`, `AV1-AV4` |
| `--type` | Surrogate model architecture | `DNN`, `dualDNN`, `LGB`, `SVM` |
| `--method` | Active learning strategy | `entropy`, `random`, `medoids`, `mc_dropout`, `k-center`, `ensemble` |
| `--budget` | Total query budget | Integer |
| `--num_queries` | Number of query rounds | Integer |
| `--num_epochs` | Training epochs per round | Integer |
| `--log_dir` | Output directory | Path |
| `--fpr` | False positive rate threshold | Float |
| `--seed` | Random seed | Integer |

### Example: Extract from SOREL-FCNN

```bash
python scripts/attacks/model_extraction.py \
  --data_dir /data/sorel-data \
  --dataset sorel \
  --target_model sorel-FCNN \
  --type LGB \
  --method medoids \
  --budget 2500 \
  --num_queries 10 \
  --num_epochs 1 \
  --log_dir /tmp/logs/ \
  --fpr 0.006 \
  --seed 42
```

### Supported Target Models

| Model Name | Type | File | Normalization Stats |
|-----------|------|------|---------------------|
| CEE | Keras | `CEE.h5` | `CEE_normalization_stats.npz` |
| LEE | LightGBM | `LEE.lgb` | `LEE_normalization_stats.npz` |
| CSE | Keras | `CSE.h5` | (optional) |
| LSE | LightGBM | `LSE.lgb` | `LSE_normalization_stats.npz` |

**Requirements:**
- Models must be placed in `artifacts/targets/`
- Normalization stats naming: `{model_name}_normalization_stats.npz`
- LightGBM models require normalization stats
- Keras models: normalization stats optional

### Evaluation

Evaluate extracted surrogate model:

```bash
python scripts/attacks/evaluate_surrogate_similarity.py \
  --surrogate_path /path/to/surrogate_model \
  --target_path /path/to/target_model \
  --test_data /path/to/test_data \
  --output_dir /path/to/output
```

**Metrics:**
- **Agreement**: Agreement rate between surrogate and target (primary metric)
- **Accuracy**: Correct prediction rate with ground truth
- **AUC-ROC**: Area under ROC curve
- **Precision, Recall, F1-score**: Classification performance
- **Confusion matrices**: Detailed classification breakdown

## Project Structure

```
model-stealing/
├── src/                    # Core library
│   ├── attackers/         # Attack implementations
│   ├── targets/           # Target model interfaces
│   ├── sampling/          # Active learning strategies
│   ├── models/            # Surrogate architectures
│   ├── datasets/          # Dataset loaders
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
│   ├── attacks/           # Extraction pipelines
│   ├── oracle/            # Oracle interfaces
│   ├── data/              # Data preprocessing
│   ├── inference/         # Model inference
│   └── examples/          # Example scripts
├── config/                # Configuration files
├── artifacts/             # Model artifacts (not tracked)
│   └── targets/           # Target model files
├── data/                  # Datasets (not tracked)
├── output/                # Experiment outputs (not tracked)
├── logs/                  # Log files (not tracked)
├── requirements.txt       # Dependencies
└── setup_venv.sh         # Setup script
```

## Key Concepts

### Feature Space Independence

MIR handles scenarios where target and attacker models use different feature spaces:

- **Antivirus Systems**: Uses raw binary files as interface. AV processes with internal features, surrogate learns on attacker-defined features (e.g., EMBER)
- **ML Models**: Automatic feature alignment. Excess features trimmed if attacker's space is larger than target's

### Black-Box Compliance

The oracle interface is designed for black-box attacks:

- Attacker only needs model name and raw features
- All implementation details hidden (model type, normalization, preprocessing)
- Only `predict()` and `predict_proba()` exposed
- Automatic model type detection and preprocessing

## Troubleshooting

- **Import Errors**: Ensure virtual environment is activated and you're in project root
- **TensorFlow Issues**: `pip install tensorflow --upgrade`
- **PyTorch Issues**: See [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- **Ember Package**: If installation fails:
  ```bash
  pip install tqdm lief
  pip install git+https://github.com/endgameinc/ember.git
  ```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This repository is part of the MIGATE framework. For issues, questions, or contributions, please refer to the main MIGATE repository or open an issue in this repository.
