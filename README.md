# MIR: Model Inferred Replica

**MIR (Model Inferred Replica)** is a component of the **MIGATE** framework (Model Inferred Replica for Adversarial Generalized Evasion), focusing on the model extraction phase for malware classifiers and antivirus systems.

## Overview

MIR provides a comprehensive framework for extracting surrogate models from black-box malware classifiers and antivirus systems using active learning techniques. The extracted models can achieve high agreement with target models while maintaining low false positive rates.

### Key Features

- **Multiple Target Support**: Extract models from various targets including:
  - EMBER-based classifiers
  - SOREL-20M models (FCNN, LightGBM)
  - Commercial antivirus systems (AV1-AV4)
  
- **Active Learning Strategies**: Multiple sampling methods for efficient query selection:
  - Entropy-based sampling
  - Random sampling
  - Medoids clustering
  - MC-Dropout uncertainty
  - K-center greedy
  - Ensemble-based methods

- **Surrogate Model Types**: Support for various surrogate architectures:
  - Deep Neural Networks (DNN, dualDNN)
  - LightGBM
  - Support Vector Machines (SVM)

- **Feature Space Independence**: Handles scenarios where target and attacker use different feature spaces through automatic feature alignment

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (usually included with Python)
- Git (for installing ember package)

### Quick Setup (Recommended)

Run the automated setup script:

```bash
./setup_venv.sh
```

This script will:
1. Check Python version
2. Create a virtual environment in `venv/`
3. Install all dependencies from `requirements.txt`
4. Install the `ember` package from GitHub (not available on PyPI)

### Manual Setup

**Step 1: Create Virtual Environment**
```bash
python3 -m venv venv
```

**Step 2: Activate Virtual Environment**

On Linux/Mac:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install ember package from GitHub (not available on PyPI)
pip install git+https://github.com/endgameinc/ember.git
```

### Troubleshooting

- **Import Errors**: Ensure virtual environment is activated and you're in the project root directory
- **TensorFlow Issues**: Try `pip install tensorflow --upgrade`
- **PyTorch Issues**: May require separate installation. See: https://pytorch.org/get-started/locally/
- **Ember Package**: If installation fails, try:
  ```bash
  pip install tqdm lief
  pip install git+https://github.com/endgameinc/ember.git
  ```

## Project Structure

```
model-stealing/
├── src/                           # Core library
│   ├── attackers/                 # Attack implementations
│   ├── targets/                    # Target model interfaces
│   ├── sampling/                  # Active learning strategies
│   ├── models/                    # Surrogate model architectures
│   ├── datasets/                  # Dataset loaders
│   └── utils/                     # Utility functions
├── scripts/                       # Executable utilities
│   ├── attacks/                   # Extraction pipelines
│   │   ├── model_extraction.py    # Main extraction script
│   │   ├── extract_final_model.py
│   │   └── evaluate_surrogate_similarity.py
│   ├── oracle/                    # Black-box query interfaces
│   ├── data/                      # Data preprocessing
│   ├── inference/                 # Model inference utilities
│   └── examples/                  # Example usage scripts
├── config/                        # Configuration files
├── data/                          # Datasets (not tracked)
├── artifacts/                     # Model artifacts (not tracked)
│   └── targets/                   # Target model files
├── output/                        # Experiment outputs (not tracked)
├── storage/                       # Training checkpoints (not tracked)
├── logs/                          # Log files (not tracked)
├── notebooks/                     # Jupyter notebooks
├── examples/                      # Example scripts
├── requirements.txt               # Python dependencies
└── setup_venv.sh                  # Virtual environment setup script
```

## Usage

### Basic Model Extraction

The main extraction script is `scripts/attacks/model_extraction.py`:

```bash
python scripts/attacks/model_extraction.py \
  --data_dir /path/to/data \
  --dataset {ember,sorel,AV} \
  --target_model {ember,sorel-FCNN,sorel-LGB,AV1,AV2,AV3,AV4} \
  --type {DNN,dualDNN,LGB,SVM} \
  --method {entropy,random,medoids,mc_dropout,k-center,ensemble} \
  --budget 2500 \
  --num_queries 10 \
  --num_epochs 1 \
  --log_dir /path/to/logs \
  --fpr 0.006 \
  --seed 42
```

### Parameters

- `--data_dir`: Directory containing the dataset
- `--dataset`: Dataset type (`ember`, `sorel`, or `AV`)
- `--target_model`: Target model to extract from
- `--type`: Surrogate model architecture
- `--method`: Active learning sampling strategy
- `--budget`: Total query budget
- `--num_queries`: Number of query rounds
- `--num_epochs`: Training epochs per round
- `--log_dir`: Output directory for logs and models
- `--fpr`: False positive rate threshold for metrics
- `--seed`: Random seed for reproducibility

### Example: Extracting from SOREL-FCNN

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

## Oracle Query Interface

The framework provides a simplified oracle interface. The attacker only needs the model name and raw features - the oracle automatically handles all preprocessing.

### Simple Usage

```python
from src.targets.oracle_client import create_oracle_from_name

# Initialize oracle - only need the model name!
oracle = create_oracle_from_name("LEE")  # or "CEE", "CSE", "LSE"

# Query with raw features
import numpy as np
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)  # Returns 0 or 1

print(f"Prediction: {prediction[0]}")  # 0 = Benign, 1 = Malware
```

### Supported Models

| Model Name | Type | File | Normalization Stats |
|-----------|------|------|---------------------|
| CEE | Keras | `CEE.h5` | `CEE_normalization_stats.npz` |
| LEE | LightGBM | `LEE.lgb` | `LEE_normalization_stats.npz` |
| CSE | Keras | `CSE.h5` | (optional) |
| LSE | LightGBM | `LSE.lgb` | `LSE_normalization_stats.npz` |

### Oracle Requirements

- Models must be placed in `artifacts/targets/`
- Normalization stats (if available) must have the same name as the model + `_normalization_stats.npz`
- For LightGBM models, normalization stats are **required**
- For Keras models, normalization stats are **optional** (will be used if available)

### Black-Box Compliance

The oracle interface is black-box compliant:
- Attacker only needs model name and raw features
- All implementation details are hidden
- Model type, normalization, and preprocessing are automatic
- Only `predict()` and `predict_proba()` are exposed

## Feature Space Independence

MIR handles scenarios where target and attacker models use different feature spaces:

- **For Antivirus Systems**: Uses raw binary files as the common interface. The AV processes files with its internal features, while the surrogate learns on attacker-defined features (e.g., EMBER features).

- **For ML Models**: Automatically performs feature alignment. If the attacker's feature space is larger than the target's, excess features are trimmed. The framework ensures interface compliance while maintaining feature space independence.

## Evaluation

After extraction, evaluate surrogate model similarity to the target:

```bash
python scripts/attacks/evaluate_surrogate_similarity.py \
  --surrogate_path /path/to/surrogate_model \
  --target_path /path/to/target_model \
  --test_data /path/to/test_data \
  --output_dir /path/to/output
```

### Metrics

- **Accuracy**: Correct prediction rate with ground truth
- **Agreement**: Agreement rate between surrogate and target predictions (most important metric)
- **AUC-ROC**: Area under ROC curve
- **Precision, Recall, F1-score**: Classification performance metrics
- **Confusion matrices**: Detailed classification breakdown

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This repository is part of the MIGATE framework. For issues, questions, or contributions, please refer to the main MIGATE repository or open an issue in this repository.
