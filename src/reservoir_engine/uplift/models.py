"""Machine learning models for production prediction.

Provides a simple, unified interface for training and evaluating
production prediction models.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd


def train_production_model(
    features_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    model_type: Literal["xgboost", "random_forest", "linear"] = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
    **model_kwargs: Any,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    """Train a production prediction model.

    Args:
        features_df: DataFrame with features and target
        target_column: Name of target column (e.g., "eur_12m")
        feature_columns: List of feature column names. If None, uses all
                        numeric columns except target.
        model_type: Type of model to train:
            - "xgboost": XGBoost regressor
            - "random_forest": Random Forest regressor
            - "linear": Linear regression
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        **model_kwargs: Additional arguments for the model

    Returns:
        Tuple of (trained_model, metrics_dict, predictions_df)

    Example:
        >>> model, metrics, preds = train_production_model(
        ...     features_df,
        ...     target_column="eur_12m",
        ...     model_type="xgboost"
        ... )
        >>> print(f"R²: {metrics['r2']:.3f}")
    """
    from sklearn.model_selection import train_test_split

    # Prepare features
    df = features_df.dropna(subset=[target_column]).copy()

    if feature_columns is None:
        feature_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_column
        ]

    # Drop rows with any NaN in features
    df = df.dropna(subset=feature_columns)

    X = df[feature_columns]
    y = df[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = _create_model(model_type, **model_kwargs)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = evaluate_model(y_train, y_pred_train, y_test, y_pred_test)

    # Create predictions DataFrame
    predictions = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred_test,
            "residual": y_test - y_pred_test,
        }
    )

    return model, metrics, predictions


def _create_model(model_type: str, **kwargs: Any) -> Any:
    """Create a model instance based on type."""
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost required. Install with: uv add xgboost")

        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        return XGBRegressor(**{**default_params, **kwargs})

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        }
        return RandomForestRegressor(**{**default_params, **kwargs})

    elif model_type == "linear":
        from sklearn.linear_model import Ridge

        return Ridge(**kwargs)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_model(
    y_train: pd.Series | np.ndarray,
    y_pred_train: np.ndarray,
    y_test: pd.Series | np.ndarray,
    y_pred_test: np.ndarray,
) -> dict[str, float]:
    """Calculate model evaluation metrics.

    Args:
        y_train: Training actual values
        y_pred_train: Training predictions
        y_test: Test actual values
        y_pred_test: Test predictions

    Returns:
        Dictionary with metrics:
        - r2_train, r2_test: R² scores
        - rmse_train, rmse_test: Root mean squared errors
        - mae_train, mae_test: Mean absolute errors
        - mape_test: Mean absolute percentage error (test only)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
    }

    # MAPE (avoid division by zero)
    y_test_array = np.array(y_test)
    mask = y_test_array != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_test_array[mask] - y_pred_test[mask]) / y_test_array[mask])) * 100
        metrics["mape_test"] = mape

    return metrics


def get_feature_importance(
    model: Any,
    feature_columns: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract feature importance from a trained model.

    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_columns: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance, sorted descending

    Example:
        >>> importance = get_feature_importance(model, feature_columns)
        >>> print(importance.head(10))
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")

    df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    )

    df = df.sort_values("importance", ascending=False)
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100

    return df.head(top_n)


def cross_validate_model(
    features_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    model_type: Literal["xgboost", "random_forest", "linear"] = "xgboost",
    n_folds: int = 5,
    **model_kwargs: Any,
) -> dict[str, float]:
    """Perform k-fold cross-validation.

    Args:
        features_df: DataFrame with features and target
        target_column: Name of target column
        feature_columns: List of feature column names
        model_type: Type of model to train
        n_folds: Number of cross-validation folds
        **model_kwargs: Additional arguments for the model

    Returns:
        Dictionary with mean and std of metrics across folds
    """
    from sklearn.model_selection import cross_val_score

    df = features_df.dropna(subset=[target_column]).copy()

    if feature_columns is None:
        feature_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_column
        ]

    df = df.dropna(subset=feature_columns)

    X = df[feature_columns]
    y = df[target_column]

    model = _create_model(model_type, **model_kwargs)

    # Calculate R² scores
    r2_scores = cross_val_score(model, X, y, cv=n_folds, scoring="r2")

    # Calculate negative MSE (sklearn convention)
    mse_scores = cross_val_score(model, X, y, cv=n_folds, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-mse_scores)

    return {
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
    }

