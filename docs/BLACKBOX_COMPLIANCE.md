# Black-Box Attack Compliance

## Summary

This document describes the black-box compliance improvements in the MIR framework to ensure proper model extraction attacks where the attacker has minimal knowledge about the target model.

## Black-Box Attack Requirements

### Attacker CAN Know:
1. **Model name** (or API endpoint)
2. **Raw features** (can query)
3. **Predictions** (0 or 1, or probabilities if API allows)

### Attacker CANNOT Know:
1. ❌ Model type (Keras vs LightGBM)
2. ❌ Normalization statistics
3. ❌ Model architecture
4. ❌ Model parameters/weights
5. ❌ Target model training data
6. ❌ Feature importance
7. ❌ Internal model workings

### Oracle Client (Provider Side):
- ✅ Automatically detects model type
- ✅ Automatically loads normalization stats
- ✅ Automatically handles preprocessing
- ✅ Only exposes `predict()` and `predict_proba()`
- ✅ Hides all implementation details

## Implemented Improvements

### 1. BlackBoxOracleClient

**File:** `src/targets/oracle_client.py`

```python
class BlackBoxOracleClient(BaseOracleClient):
    """
    Black Box Oracle Client - Completely hides implementation details from attacker.
    
    Attacker only needs:
    - Model name
    - Raw features
    - Receives predictions
    """
    
    def __init__(self, model_name: str, ...):
        # Automatically detects everything, hidden from attacker
        self._oracle = create_oracle_from_name(...)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Only exposes predict(), hides implementation
        return self._oracle.predict(X)
```

### 2. Updated create_oracle_from_name

**File:** `src/targets/oracle_client.py`

- Returns `BlackBoxOracleClient` instead of `LocalOracleClient`
- Hides `model_type`, `normalization_stats_path` from attacker
- Automatically detects everything

### 3. Updated Attack Scripts

**File:** `scripts/attacks/extract_final_model.py`

- When using `model_name`, automatically uses `BlackBoxOracleClient`
- Attacker doesn't need to know `model_type` or `normalization_stats_path`
- Only needs model name and raw features

## Comparison

### Before (Violates Black-Box):

```python
# Attacker must know model_type and normalization_stats_path
oracle_client = LocalOracleClient(
    model_type="lgb",  # ❌ Attacker knows model type
    model_path=...,
    normalization_stats_path=...,  # ❌ Attacker knows normalization stats
    ...
)
```

### After (Black-Box Compliant):

```python
# Attacker only needs model name
oracle_client = create_oracle_from_name(
    model_name="LEE",  # ✅ Only need model name
    feature_dim=2381,
)
# Automatically detects model_type, loads normalization_stats, etc.
```

## Important Notes

### 1. Ground Truth Labels from Training Data
- ✅ **Valid**: Attacker uses ground truth labels from thief dataset
- 💡 Attacker controls thief dataset, may have labels for their own data
- 💡 This does not violate black-box assumption

### 2. get_required_feature_dim()
- ⚠️ **May be valid**: In real black-box attacks, attacker may know input size
- 💡 Through API documentation or trial-and-error
- 💡 But should not know model architecture or normalization

### 3. Logging/Debugging
- ⚠️ **For logging only**: Some information (model_type, model_path) is still logged
- 💡 In real black-box attacks, attacker should not see these logs
- 💡 Can disable logging or only log on provider side

## Conclusion

✅ **Improved** to ensure black-box compliance:
- Attacker only needs model name
- Oracle client automatically handles everything
- Implementation details are hidden

⚠️ **Still some points**:
- Logging may leak information (can be disabled)
- `get_required_feature_dim()` may leak information (can be hidden)

💡 **In practice**:
- Oracle client should run on separate server (provider side)
- Attacker can only query via API
- No access to code or logs

## See Also

- [ORACLE_USAGE.md](ORACLE_USAGE.md) - Oracle usage guide
- [BLACKBOX_ANALYSIS.md](BLACKBOX_ANALYSIS.md) - Detailed black-box analysis
