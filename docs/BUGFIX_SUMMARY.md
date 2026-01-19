# Bug Fix Summary

## Critical Issue: Oracle Query with Raw Data

### Problem Description
- **Issue**: Oracle (target model) was queried with **raw data** (unscaled), but target model was trained with **scaled data**.
- **Consequence**: Oracle gave completely wrong predictions (0% agreement with ground truth when querying with raw data, but 90% agreement when querying with scaled data).
- **Root Cause**: Target model (`final_model.h5`) was trained with normalized data (RobustScaler + clip), but code queried oracle with raw data.

### Solution
1. **Scale data BEFORE querying oracle**:
   - Create scaler and fit on seed+val+pool
   - Scale all data (eval, seed, val, pool) before querying oracle
   - Query oracle with scaled data

2. **Updated code in `extract_final_model.py`**:
   - Move data scaling before oracle query
   - Query oracle with `X_eval_s`, `X_seed_s`, `X_val_s`, `X_pool_s` (scaled)
   - Train surrogate with scaled data (already correct)

### Code Changes
```python
# BEFORE (WRONG):
y_eval = oracle(X_eval)  # Query with raw data
y_seed = oracle(X_seed)  # Query with raw data
y_val = oracle(X_val)    # Query with raw data
scaler = RobustScaler()
scaler.fit(...)
X_eval_s = _clip_scale(scaler, X_eval)  # Scale after query

# AFTER (CORRECT):
scaler = RobustScaler()
scaler.fit(...)
X_eval_s = _clip_scale(scaler, X_eval)  # Scale BEFORE
y_eval = oracle(X_eval_s)  # Query with scaled data
y_seed = oracle(X_seed_s)  # Query with scaled data
y_val = oracle(X_val_s)    # Query with scaled data
```

## Secondary Issue: KerasAttacker Hardcoded Input Shape

### Problem Description
- **Issue**: `KerasAttacker` hardcoded `input_shape=(2381,)`, but dataset may have different number of features.
- **Consequence**: If dataset has different number of features than 2381, model won't match the data.

### Solution
1. **Added parameter `input_shape` to `KerasAttacker.__init__()`**:
   - Default is `(2381,)` to maintain backward compatibility
   - Allows passing `input_shape` from outside

2. **Updated `extract_final_model.py`**:
   - Pass `input_shape=(feature_dim,)` to `KerasAttacker`
   - Ensure model matches actual number of features in dataset

### Code Changes
```python
# BEFORE:
class KerasAttacker(AbstractAttacker):
    def __init__(self, early_stopping=30, seed=42, mc=False):
        self.model = create_dnn(seed=seed, input_shape=(2381,), mc=mc)

# AFTER:
class KerasAttacker(AbstractAttacker):
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381,)):
        self.model = create_dnn(seed=seed, input_shape=input_shape, mc=mc)

# Usage:
attacker = KerasAttacker(early_stopping=10, seed=seed, input_shape=(feature_dim,))
```

## Expected Results

After fixing the above issues:
1. **Correct oracle predictions**: Oracle will give correct predictions because it's queried with scaled data (same as during training).
2. **Increased agreement**: Agreement between surrogate and target will increase significantly (from ~0% to ~90%+).
3. **Increased accuracy**: Surrogate model will learn correct patterns from target model.
4. **Model matches data**: KerasAttacker will automatically match actual number of features in dataset.

## Modified Files

1. **`scripts/attacks/extract_final_model.py`**:
   - Scale data before querying oracle
   - Query oracle with scaled data
   - Pass `input_shape` to `KerasAttacker`

2. **`src/attackers/__init__.py`**:
   - Added parameter `input_shape` to `KerasAttacker.__init__()`

## Verification

To verify the issues are fixed:
1. Run `scripts/attacks/extract_final_model.py` with small dataset
2. Check oracle predictions distribution (should have both 0 and 1, not all 1)
3. Check agreement between surrogate and target (should be > 80%)
4. Check surrogate model accuracy (should increase with number of queries)
