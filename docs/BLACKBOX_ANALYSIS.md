# Black-Box Compliance Analysis

## Summary

This document provides a detailed analysis of black-box compliance in the MIR framework, ensuring proper model extraction attacks where the attacker has minimal knowledge about the target model.

## Black-Box Attack Requirements

### ✅ Attacker CAN Know:
1. **Model name** (or API endpoint) - ✅ OK
2. **Raw features** (can query) - ✅ OK
3. **Predictions** (0 or 1, or probabilities) - ✅ OK

### ❌ Attacker CANNOT Know:
1. **Model type** (Keras vs LightGBM) - ✅ Hidden
2. **Normalization statistics** - ✅ Hidden
3. **Model architecture** - ✅ Hidden
4. **Model parameters/weights** - ✅ Hidden
5. **Target model training data** - ✅ OK (attacker has no access)
6. **Feature importance** - ✅ OK
7. **Internal model workings** - ✅ Hidden

## Detailed Checks

### 1. Oracle Client Interface

**Before (Violates):**
```python
# Attacker must know model_type and normalization_stats_path
oracle_client = LocalOracleClient(
    model_type="lgb",  # ❌ Attacker knows model type
    model_path=...,
    normalization_stats_path=...,  # ❌ Attacker knows normalization stats
)
```

**After (Black-Box Compliant):**
```python
# Attacker only needs model name
oracle_client = create_oracle_from_name(
    model_name="LEE",  # ✅ Only need model name
    feature_dim=2381,
)
# Automatically detects model_type, loads normalization_stats, etc.
```

### 2. BlackBoxOracleClient

**Attributes Attacker Can Access:**
- ✅ `model_name`: Model name (OK)
- ✅ `predict(X)`: Predict binary labels (OK)
- ✅ `predict_proba(X)`: Predict probabilities (OK)
- ✅ `supports_probabilities()`: Check probability support (OK)
- ✅ `get_required_feature_dim()`: Get required feature dimension (OK - may know via API docs)

**Attributes Attacker CANNOT Access:**
- ✅ `model_type`: Hidden
- ✅ `model_path`: Hidden
- ✅ `normalization_stats_path`: Hidden
- ⚠️ `_oracle`: Internal (in Python can still access, but in practice oracle runs on separate server)

### 3. Ground Truth Labels from Training Data

**✅ Valid:**
- Attacker uses ground truth labels from thief dataset
- Attacker controls thief dataset, may have labels for their own data
- This does not violate black-box assumption

### 4. Logging

**⚠️ Note:**
- Some information (model_type, model_path) is still logged in `extract_final_model.py`
- In real black-box attacks, attacker should not see these logs
- **Solution**: Logging should only be on provider side (server), not exposed to attacker

## Check Results

```
✅ model_type: Hidden
✅ model_path: Hidden
✅ normalization_stats_path: Hidden
✅ Oracle client only exposes predict() and predict_proba()
✅ Attacker only needs model name to create oracle client
```

## Implemented Improvements

### 1. Created BlackBoxOracleClient
- Wraps `LocalOracleClient` to hide implementation details
- Only exposes `predict()` and `predict_proba()`
- Automatically detects model type, loads normalization stats

### 2. Updated create_oracle_from_name
- Returns `BlackBoxOracleClient` (default `blackbox=True`)
- Automatically detects everything, hidden from attacker

### 3. Updated Attack Scripts
- When using `model_name`, automatically uses `BlackBoxOracleClient`
- Attacker doesn't need to know `model_type` or `normalization_stats_path`

## Important Considerations

### 1. Python Limitation
- In Python, there are no true private attributes
- Attacker can still access `_oracle` (but shouldn't)
- **In practice**: Oracle client runs on separate server, attacker has no access to code

### 2. Logging
- Logging may leak information
- **Solution**: Disable logging or only log on server side
- Attacker should not see logs about model_type, model_path, etc.

### 3. API Design
- In real black-box attacks, oracle should be an API endpoint
- Attacker can only query via HTTP/REST API
- No access to code, logs, or file system

## Conclusion

✅ **Improved** to ensure black-box compliance:
- Attacker only needs model name
- Oracle client automatically handles everything
- Implementation details are hidden

⚠️ **Still some points**:
- Logging may leak information (can be disabled)
- Python doesn't have true private (but in practice oracle runs on separate server)

💡 **In practice**:
- Oracle client should run on separate server (provider side)
- Attacker can only query via API
- No access to code or logs

## See Also

- [BLACKBOX_COMPLIANCE.md](BLACKBOX_COMPLIANCE.md) - Black-box compliance guide
- [ORACLE_USAGE.md](ORACLE_USAGE.md) - Oracle usage examples
