from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def build_model(
    model_name: str,
    params: Dict[str, Any],
    sample_count: int = 10
):
    """Instantiates classifier instance configured with hyperparameters."""
    model_name_lower = model_name.lower()

    if model_name_lower in ("random_forest", "rf"):
        rf_params = dict(params)
        return RandomForestClassifier(**rf_params)

    elif model_name_lower in ("lightgbm", "lgbm"):
        lgbm_params = dict(params)
        # Adapt leaf constraints for micro test samples while maintaining standard defaults
        if sample_count < 20 and "min_child_samples" not in lgbm_params:
            lgbm_params["min_child_samples"] = 1
        if "verbose" not in lgbm_params:
            lgbm_params["verbose"] = -1
        return LGBMClassifier(**lgbm_params)

    else:
        raise ValueError(f"Unsupported model identifier: '{model_name}'. Allowed: 'random_forest', 'lightgbm'.")
