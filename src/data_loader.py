import os
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(
    file_path: str,
    target_column: str = "label",
    binary_classification: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads CSV dataset and splits into feature matrix X and target vector y."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' missing from dataset columns.")

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])

    # Convert non-numeric values to numeric, filling NaNs
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if binary_classification:
        # 0 for Benign traffic, 1 for any DDoS/Attack variant
        y = (y_raw != "Benign").astype(int)
    else:
        y = y_raw.astype("category").cat.codes

    return X, y


def extract_feature_subsets(
    X: pd.DataFrame,
    top_10_features: List[str]
) -> Dict[str, pd.DataFrame]:
    """Returns dictionary of full and reduced (top-10) feature matrices."""
    available_top10 = [col for col in top_10_features if col in X.columns]
    if len(available_top10) < len(top_10_features):
        missing = set(top_10_features) - set(available_top10)
        raise ValueError(f"Specified top-10 features missing from dataset: {missing}")

    return {
        "full": X.copy(),
        "top_10": X[available_top10].copy()
    }


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits features and targets into training and evaluation sets."""
    stratify_target = y if stratify and len(y.unique()) > 1 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )
