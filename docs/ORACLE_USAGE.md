# Oracle Query Interface Guide

## Overview

The MIR framework provides a simplified oracle interface for querying target models. The attacker only needs the model name and raw features - the oracle automatically handles all preprocessing and model-specific details.

## Simple Usage

### Basic Example

```python
from src.targets.oracle_client import create_oracle_from_name

# Initialize oracle - only need the model name!
oracle = create_oracle_from_name("LEE")  # or "CEE", "CSE", "LSE"

# Query with raw features
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)  # Returns 0 or 1

print(f"Prediction: {prediction[0]}")  # 0 = Benign, 1 = Malware
```

### Automatic Processing

The oracle automatically handles:
- ✅ Finding model files (`.h5` or `.lgb`)
- ✅ Detecting model type (Keras or LightGBM)
- ✅ Finding normalization statistics
- ✅ Normalizing features
- ✅ Aligning feature dimensions
- ✅ Returning binary predictions

**The attacker does NOT need to know:**
- ❌ Model type (h5 or lgb)
- ❌ Model file path
- ❌ Normalization statistics
- ❌ Preprocessing steps

## Supported Models

| Model Name | Type | File | Normalization Stats |
|-----------|------|------|---------------------|
| CEE | Keras | `CEE.h5` | `CEE_normalization_stats.npz` |
| LEE | LightGBM | `LEE.lgb` | `LEE_normalization_stats.npz` |
| CSE | Keras | `CSE.h5` | (optional) |
| LSE | LightGBM | `LSE.lgb` | `LSE_normalization_stats.npz` |

## Complete Example

```python
import numpy as np
from src.targets.oracle_client import create_oracle_from_name

# 1. Initialize oracle
oracle = create_oracle_from_name("LEE")

# 2. Query a single sample
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)
print(f"Prediction: {prediction[0]}")

# 3. Query a batch
batch = np.random.randn(10, 2381).astype(np.float32)
predictions = oracle.predict(batch)
print(f"Predictions: {predictions}")

# 4. Query with probabilities (if supported)
if oracle.supports_probabilities():
    probs = oracle.predict_proba(batch)
    print(f"Probabilities: {probs}")
```

## Usage in Scripts

### Testing Oracle Queries

```bash
# Only need model name - no model-type or model-path required
python config/test_oracle_query.py \
    --parquet-path data/test_ember_2018_v2_features_label_other.parquet \
    --model-name LEE \
    --max-samples 5000
```

### In Attack Scripts

```python
from src.targets.oracle_client import create_oracle_from_name

# In attack script
oracle = create_oracle_from_name("LEE")

# Query samples
for sample in samples:
    label = oracle.predict(sample)
    # Use label to train surrogate model
```

## Benefits

1. **Simplicity**: Only need the model name
2. **Automatic**: Oracle automatically detects and handles everything
3. **Flexible**: Works with both Keras and LightGBM models
4. **Clean**: No need to know preprocessing details

## Important Notes

- Models must be placed in `artifacts/targets/`
- Normalization stats (if available) must have the same name as the model + `_normalization_stats.npz`
- For LightGBM models, normalization stats are **required**
- For Keras models, normalization stats are **optional** (will be used if available)

## Black-Box Compliance

The oracle interface is designed to be black-box compliant:
- Attacker only needs model name and raw features
- All implementation details are hidden
- Model type, normalization, and preprocessing are automatic
- Only `predict()` and `predict_proba()` are exposed

For more details on black-box compliance, see [BLACKBOX_COMPLIANCE.md](BLACKBOX_COMPLIANCE.md).

## See Also

- `examples/ultra_simple_oracle.py` - Ultra simple example
- `examples/simple_oracle_query.py` - More complete example
- `config/test_oracle_query.py` - Oracle testing script
- [BLACKBOX_COMPLIANCE.md](BLACKBOX_COMPLIANCE.md) - Black-box attack guidelines
