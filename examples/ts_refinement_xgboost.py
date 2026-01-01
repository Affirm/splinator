"""
TS-Refinement Early Stopping for XGBoost
=========================================

This example demonstrates the "Refine, Then Calibrate" training paradigm
from Berta et al. (2025) using splinator's TS-refinement metrics with XGBoost.

Key insight: Standard early stopping based on validation log-loss is suboptimal
because it forces a compromise between discrimination and calibration.
These are minimized at DIFFERENT points during training!

Strategy:
1. Use ts_refinement_loss as the early stopping criterion (train longer)
2. Apply TemperatureScaling post-hoc to fix calibration

This achieves better discrimination AND calibration than standard early stopping.

Dataset: Challenging synthetic data designed to stress-test calibration:
- 150,000 samples with complex nonlinear decision boundary
- 15% label noise (makes perfect calibration impossible)
- 50 features (20 informative, 30 noise/correlated)
- Creates conditions where models become overconfident quickly

This data clearly shows the calibration/refinement tradeoff: standard early
stopping stops too early (8 iterations) while TS-Refinement correctly trains
much longer (400+ iterations) for better discrimination.

References:
    Berta, M., Ciobanu, S., & Heusinger, M. (2025). Rethinking Early Stopping:
    Refine, Then Calibrate. arXiv preprint arXiv:2501.19195.
    https://arxiv.org/abs/2501.19195

Requirements:
    pip install xgboost splinator scikit-learn matplotlib
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# Check if xgboost is available
try:
    import xgboost as xgb
except ImportError:
    raise ImportError(
        "This example requires xgboost. Install with: pip install xgboost"
    )

from splinator import (
    ts_refinement_loss,
    ts_brier_refinement,  # Brier after TS (for fair comparison)
    calibration_loss,
    loss_decomposition,
    TemperatureScaling,
    # Brier-based decomposition (Berta et al. 2025)
    brier_decomposition,
)
from splinator.metric_wrappers import make_metric_wrapper


def load_adult_income_data():
    """Load the Adult Income (Census) dataset from OpenML.
    
    This is a classic benchmark for calibration with ~48k samples.
    Task: Predict whether income exceeds $50K/year.
    
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Preprocessed feature matrix.
    y : ndarray of shape (n_samples,)
        Binary target (1 = >$50K, 0 = <=$50K).
    """
    print("   Downloading Adult Income dataset from OpenML...")
    adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    
    X = adult.data
    y = adult.target
    
    # Convert target to binary
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print(f"   Dataset size: {len(y):,} samples")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution: {np.mean(y):.1%} positive")
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing: impute + one-hot encode
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_processed = X_processed.fillna(X_processed.median())
    
    return X_processed.values.astype(np.float32), y


def load_covertype_data():
    """Load the Cover Type dataset from OpenML.
    
    This is a larger dataset (~580k samples) converted to binary classification.
    Task: Predict forest cover type (class 2 vs rest).
    
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Binary target.
    """
    print("   Downloading Cover Type dataset from OpenML...")
    covertype = fetch_openml(name='covertype', version=3, as_frame=True, parser='auto')
    
    X = covertype.data.values.astype(np.float32)
    y_multi = covertype.target.astype(int).values
    
    # Convert to binary: class 2 (most common) vs rest
    y = (y_multi == 2).astype(int)
    
    print(f"   Dataset size: {len(y):,} samples")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution: {np.mean(y):.1%} positive")
    
    return X, y


def create_challenging_data(n_samples=200000, n_features=50, noise_rate=0.15, seed=42):
    """Create synthetic data designed to stress-test calibration vs refinement.
    
    This data has:
    1. Complex nonlinear decision boundary (XGBoost needs many iterations)
    2. Label noise (makes calibration harder, tests post-hoc correction)
    3. Class imbalance (30% positive)
    4. Redundant features (more room for overfitting)
    5. Different feature scales
    
    The key property: a model that trains longer will have better 
    discrimination but WORSE calibration, clearly showing the tradeoff.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features (some informative, some noise).
    noise_rate : float
        Fraction of labels to flip (makes calibration impossible to perfect).
    seed : int
        Random seed.
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    np.random.seed(seed)
    
    # Informative features
    n_informative = 20
    X_informative = np.random.randn(n_samples, n_informative)
    
    # Complex nonlinear decision boundary
    # Combines several interaction effects
    logit = (
        2.0 * X_informative[:, 0] * X_informative[:, 1]  # Interaction
        + 1.5 * np.sin(3 * X_informative[:, 2])           # Nonlinear
        + 1.0 * X_informative[:, 3]**2                    # Quadratic
        - 1.5 * X_informative[:, 4]                       # Linear
        + 0.8 * np.abs(X_informative[:, 5])               # Absolute
        + 0.6 * X_informative[:, 6] * X_informative[:, 7] * X_informative[:, 8]  # 3-way
        - 0.5 * np.cos(2 * X_informative[:, 9] + X_informative[:, 10])
        + 0.4 * (X_informative[:, 11] > 0).astype(float) * X_informative[:, 12]
        - 1.0  # Shift to get ~30% positive rate
    )
    
    # True probabilities
    true_probs = 1 / (1 + np.exp(-logit))
    
    # Generate labels from true probabilities
    y = (np.random.rand(n_samples) < true_probs).astype(int)
    
    # Add label noise (flip some labels) - this makes perfect calibration impossible
    noise_mask = np.random.rand(n_samples) < noise_rate
    y[noise_mask] = 1 - y[noise_mask]
    
    # Add noise features (redundant, some correlated with informative)
    n_noise = n_features - n_informative
    X_noise = np.random.randn(n_samples, n_noise)
    # Make some noise features correlated with informative ones
    for i in range(min(10, n_noise)):
        X_noise[:, i] = 0.7 * X_informative[:, i % n_informative] + 0.3 * X_noise[:, i]
    
    X = np.hstack([X_informative, X_noise]).astype(np.float32)
    
    # Different feature scales (stress test tree splits)
    scales = np.random.exponential(5, n_features)
    X *= scales
    
    print(f"   Synthetic challenging data:")
    print(f"   - Samples: {n_samples:,}")
    print(f"   - Features: {n_features} ({n_informative} informative, {n_noise} noise)")
    print(f"   - Label noise rate: {noise_rate:.0%}")
    print(f"   - Class distribution: {np.mean(y):.1%} positive")
    
    return X, y


def train_with_standard_early_stopping(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost with standard log-loss early stopping."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # More aggressive hyperparameters that tend to overfit
    # This creates conditions where calibration degrades faster than discrimination
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 10,
        'learning_rate': 0.3,  # Higher learning rate = faster overfitting
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 42,
    }
    
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,  # Shorter patience
        evals_result=evals_result,
        verbose_eval=False,
    )
    
    # Get predictions
    train_probs = model.predict(dtrain)
    val_probs = model.predict(dval)
    test_probs = model.predict(dtest)
    
    return {
        'model': model,
        'best_iteration': model.best_iteration,
        'train_probs': train_probs,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'evals_result': evals_result,
    }


def train_with_ts_refinement_early_stopping(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost with TS-refinement early stopping, then calibrate."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Create custom metric wrapper
    ts_metric = make_metric_wrapper(
        ts_refinement_loss,
        framework='xgboost',
        name='ts_refinement',
    )
    
    # Same aggressive hyperparameters
    params = {
        'objective': 'binary:logistic',
        'disable_default_eval_metric': True,  # Use only our custom metric
        'max_depth': 10,
        'learning_rate': 0.3,  # Higher learning rate = faster overfitting
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 42,
    }
    
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        custom_metric=ts_metric,
        early_stopping_rounds=10,  # Shorter patience
        evals_result=evals_result,
        verbose_eval=False,
    )
    
    # Get raw predictions
    train_probs_raw = model.predict(dtrain)
    val_probs_raw = model.predict(dval)
    test_probs_raw = model.predict(dtest)
    
    # Apply temperature scaling calibration
    ts = TemperatureScaling()
    ts.fit(val_probs_raw.reshape(-1, 1), y_val)
    
    train_probs = ts.predict(train_probs_raw.reshape(-1, 1))
    val_probs = ts.predict(val_probs_raw.reshape(-1, 1))
    test_probs = ts.predict(test_probs_raw.reshape(-1, 1))
    
    return {
        'model': model,
        'calibrator': ts,
        'best_iteration': model.best_iteration,
        'train_probs': train_probs,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'train_probs_raw': train_probs_raw,
        'val_probs_raw': val_probs_raw,
        'test_probs_raw': test_probs_raw,
        'evals_result': evals_result,
    }


def train_with_ts_brier_refinement_early_stopping(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost with TS-Brier refinement early stopping.
    
    Uses Brier score after temperature scaling - same recalibrator as TS-Refinement
    but with Brier scoring rule instead of log-loss.
    
    This allows direct comparison of log-loss vs Brier under the same recalibration.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Create custom metric wrapper for TS-Brier refinement
    ts_brier_metric = make_metric_wrapper(
        ts_brier_refinement,
        framework='xgboost',
        name='ts_brier_refinement',
    )
    
    # Same aggressive hyperparameters
    params = {
        'objective': 'binary:logistic',
        'disable_default_eval_metric': True,
        'max_depth': 10,
        'learning_rate': 0.3,
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 42,
    }
    
    start_time = time.time()
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        custom_metric=ts_brier_metric,
        early_stopping_rounds=10,
        evals_result=evals_result,
        verbose_eval=False,
    )
    training_time = time.time() - start_time
    
    # Get raw predictions
    train_probs_raw = model.predict(dtrain)
    val_probs_raw = model.predict(dval)
    test_probs_raw = model.predict(dtest)
    
    # Apply temperature scaling calibration
    ts = TemperatureScaling()
    ts.fit(val_probs_raw.reshape(-1, 1), y_val)
    
    train_probs = ts.predict(train_probs_raw.reshape(-1, 1))
    val_probs = ts.predict(val_probs_raw.reshape(-1, 1))
    test_probs = ts.predict(test_probs_raw.reshape(-1, 1))
    
    print(f"   Stopped at iteration: {model.best_iteration}")
    print(f"   Optimal temperature: {ts.temperature_:.3f}")
    print(f"   Training time: {training_time:.1f}s")
    
    return {
        'model': model,
        'calibrator': ts,
        'best_iteration': model.best_iteration,
        'train_probs': train_probs,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'train_probs_raw': train_probs_raw,
        'val_probs_raw': val_probs_raw,
        'test_probs_raw': test_probs_raw,
        'evals_result': evals_result,
        'training_time': training_time,
    }


def evaluate_model(y_true, y_pred, name):
    """Evaluate a model's predictions with both decompositions."""
    # Basic metrics
    metrics = {
        'AUC-ROC': roc_auc_score(y_true, y_pred),
        'Log-Loss': log_loss(y_true, y_pred),
        'Brier Score': brier_score_loss(y_true, y_pred),
    }
    
    # Log-loss decomposition (via Temperature Scaling)
    ts_decomp = loss_decomposition(y_true, y_pred)
    metrics['TS-Refinement'] = ts_decomp['refinement_loss']
    metrics['TS-Calibration'] = ts_decomp['calibration_loss']
    
    # Brier score decomposition (Berta et al. 2025 variational decomposition)
    # Refinement = Brier AFTER optimal recalibration (isotonic regression)
    # Calibration = Brier - Refinement (fixable by recalibration)
    brier_decomp = brier_decomposition(y_true, y_pred)
    metrics['Brier-Refinement'] = brier_decomp['refinement']  # Brier after recalibration
    metrics['Brier-Calibration'] = brier_decomp['calibration']  # Fixable portion
    metrics['Spread-Term'] = brier_decomp['spread_term']  # E[p(1-p)] for reference
    
    print(f"\n{name}:")
    print("-" * 50)
    print(f"  AUC-ROC:          {metrics['AUC-ROC']:.4f}")
    print(f"  Log-Loss:         {metrics['Log-Loss']:.4f}")
    print(f"  Brier Score:      {metrics['Brier Score']:.4f}")
    print(f"  --- Log-Loss Decomposition (TS-based) ---")
    print(f"  TS-Refinement:    {metrics['TS-Refinement']:.4f}")
    print(f"  TS-Calibration:   {metrics['TS-Calibration']:.4f}")
    print(f"  --- Brier Decomposition (Berta et al. 2025) ---")
    print(f"  Refinement:       {metrics['Brier-Refinement']:.4f} (Brier after recalibration)")
    print(f"  Calibration:      {metrics['Brier-Calibration']:.4f} (fixable portion)")
    print(f"  Spread E[p(1-p)]: {metrics['Spread-Term']:.4f} (raw, NOT refinement)")
    
    return metrics


def plot_comparison(standard_results, ts_results, ts_brier_results, y_val, y_test):
    """Plot comparison of three approaches."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    colors = {
        'std': '#1f77b4',      # blue
        'ts_ll': '#d62728',    # red
        'ts_b': '#ff7f0e',     # orange
    }
    
    # Plot 1: Training curves - iterations comparison
    ax1 = axes[0, 0]
    std_val_loss = standard_results['evals_result']['val']['logloss']
    ax1.plot(std_val_loss, label='Standard (log-loss)', color=colors['std'])
    ax1.axvline(x=standard_results['best_iteration'], color=colors['std'], 
                linestyle='--', alpha=0.5)
    
    ts_val_loss = ts_results['evals_result']['val']['ts_refinement']
    ax1.plot(ts_val_loss, label='TS-LogLoss-Ref', color=colors['ts_ll'])
    ax1.axvline(x=ts_results['best_iteration'], color=colors['ts_ll'], 
                linestyle='--', alpha=0.5)
    
    ts_brier_val_loss = ts_brier_results['evals_result']['val']['ts_brier_refinement']
    ax1.plot(ts_brier_val_loss, label='TS-Brier-Ref', color=colors['ts_b'])
    ax1.axvline(x=ts_brier_results['best_iteration'], color=colors['ts_b'], 
                linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Boosting Round')
    ax1.set_ylabel('Validation Metric')
    ax1.set_title(f'Early Stopping Comparison\n'
                  f'(Std: {standard_results["best_iteration"]}, '
                  f'TS-LL: {ts_results["best_iteration"]}, '
                  f'TS-B: {ts_brier_results["best_iteration"]})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Brier Decomposition (Berta et al. 2025)
    ax2 = axes[0, 1]
    
    std_brier = brier_decomposition(y_test, standard_results['test_probs'])
    ts_brier = brier_decomposition(y_test, ts_results['test_probs'])
    ts_b_brier = brier_decomposition(y_test, ts_brier_results['test_probs'])
    
    x = np.arange(3)
    width = 0.6
    
    # Brier = Refinement (after recal) + Calibration (fixable)
    refinement = [std_brier['refinement'], ts_brier['refinement'], 
                  ts_b_brier['refinement']]
    calibration = [std_brier['calibration'], ts_brier['calibration'], 
                   ts_b_brier['calibration']]
    
    bars1 = ax2.bar(x, refinement, width, label='Refinement', color='steelblue')
    bars2 = ax2.bar(x, calibration, width, bottom=refinement, 
                    label='Calibration (fixable)', color='coral', alpha=0.7)
    
    ax2.set_ylabel('Brier Score Component')
    ax2.set_title('Brier Decomposition (Berta et al. 2025)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Standard', 'TS-LL', 'TS-B'], fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add total Brier score annotations
    for i, (ref, cal) in enumerate(zip(refinement, calibration)):
        total = ref + cal
        ax2.annotate(f'{total:.4f}', xy=(i, total + 0.002),
                    ha='center', fontsize=8)
    
    # Plot 3: Calibration curves
    ax3 = axes[1, 0]
    
    from sklearn.calibration import calibration_curve
    
    # Standard model
    prob_true_std, prob_pred_std = calibration_curve(
        y_test, standard_results['test_probs'], n_bins=10, strategy='quantile'
    )
    ax3.plot(prob_pred_std, prob_true_std, 's-', label='Standard', color=colors['std'])
    
    # TS-LogLoss-Refinement
    prob_true_ts, prob_pred_ts = calibration_curve(
        y_test, ts_results['test_probs'], n_bins=10, strategy='quantile'
    )
    ax3.plot(prob_pred_ts, prob_true_ts, 'o-', label='TS-LL-Ref', color=colors['ts_ll'])
    
    # TS-Brier-Refinement
    prob_true_tsb, prob_pred_tsb = calibration_curve(
        y_test, ts_brier_results['test_probs'], n_bins=10, strategy='quantile'
    )
    ax3.plot(prob_pred_tsb, prob_true_tsb, 'd-', label='TS-Brier-Ref', color=colors['ts_b'])
    
    # Perfect calibration line
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.5)
    
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title('Calibration Curves (Test Set)')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary metrics comparison
    ax4 = axes[1, 1]
    
    metrics = ['AUC-ROC', 'Log-Loss', 'Brier']
    std_vals = [
        roc_auc_score(y_test, standard_results['test_probs']),
        log_loss(y_test, standard_results['test_probs']),
        brier_score_loss(y_test, standard_results['test_probs']),
    ]
    ts_ll_vals = [
        roc_auc_score(y_test, ts_results['test_probs']),
        log_loss(y_test, ts_results['test_probs']),
        brier_score_loss(y_test, ts_results['test_probs']),
    ]
    ts_b_vals = [
        roc_auc_score(y_test, ts_brier_results['test_probs']),
        log_loss(y_test, ts_brier_results['test_probs']),
        brier_score_loss(y_test, ts_brier_results['test_probs']),
    ]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax4.bar(x - width, std_vals, width, label='Standard', color=colors['std'])
    bars2 = ax4.bar(x, ts_ll_vals, width, label='TS-LL', color=colors['ts_ll'])
    bars3 = ax4.bar(x + width, ts_b_vals, width, label='TS-B', color=colors['ts_b'])
    
    ax4.set_ylabel('Score')
    ax4.set_title('Test Set Metrics (higher AUC, lower loss = better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=45)
    
    plt.tight_layout()
    plt.savefig('ts_refinement_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved to: ts_refinement_comparison.png")


def main():
    print("=" * 60)
    print("TS-Refinement Early Stopping for XGBoost")
    print("Challenging Synthetic Data (designed to stress-test calibration)")
    print("=" * 60)
    
    # Create challenging synthetic data
    # Key features:
    # - Complex nonlinear boundary (needs many iterations)
    # - 15% label noise (makes perfect calibration impossible)
    # - Class imbalance (~30% positive)
    # - 50 features (20 informative, 30 noise/correlated)
    print("\n1. Creating challenging synthetic data...")
    X, y = create_challenging_data(
        n_samples=150000,
        n_features=50,
        noise_rate=0.15,  # 15% label noise - stresses calibration
        seed=42
    )
    
    # Split: train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Train with standard early stopping
    print("\n2. Training with STANDARD early stopping (log-loss)...")
    standard_results = train_with_standard_early_stopping(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"   Stopped at iteration: {standard_results['best_iteration']}")
    
    # Train with TS-refinement early stopping (log-loss based, requires optimization)
    print("\n3. Training with TS-LOGLOSS-REFINEMENT early stopping...")
    t0 = time.time()
    ts_results = train_with_ts_refinement_early_stopping(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    ts_time = time.time() - t0
    print(f"   Stopped at iteration: {ts_results['best_iteration']}")
    print(f"   Optimal temperature: {ts_results['calibrator'].temperature_:.3f}")
    print(f"   Training time: {ts_time:.1f}s")
    
    # Train with TS-Brier-refinement early stopping (Brier after TS)
    print("\n4. Training with TS-BRIER-REFINEMENT early stopping...")
    ts_brier_results = train_with_ts_brier_refinement_early_stopping(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Evaluate all approaches
    print("\n5. Evaluating on TEST set...")
    
    std_metrics = evaluate_model(
        y_test, standard_results['test_probs'],
        "Standard Early Stopping (log-loss)"
    )
    
    ts_metrics = evaluate_model(
        y_test, ts_results['test_probs'],
        "TS-LogLoss-Refinement + Calibration"
    )
    
    ts_brier_metrics = evaluate_model(
        y_test, ts_brier_results['test_probs'],
        "TS-Brier-Refinement + Calibration"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nIterations trained:")
    print(f"  Standard:            {standard_results['best_iteration']}")
    print(f"  TS-LogLoss-Ref:      {ts_results['best_iteration']} "
          f"({ts_results['best_iteration'] - standard_results['best_iteration']:+d} more)")
    print(f"  TS-Brier-Ref:        {ts_brier_results['best_iteration']} "
          f"({ts_brier_results['best_iteration'] - standard_results['best_iteration']:+d} more)")
    
    print(f"\nTest set improvements vs Standard:")
    print(f"{'Metric':<15} {'TS-LL':>12} {'TS-Brier':>12}")
    print("-" * 45)
    
    for metric in ['AUC-ROC', 'Log-Loss', 'Brier Score']:
        ts_diff = ts_metrics[metric] - std_metrics[metric]
        ts_brier_diff = ts_brier_metrics[metric] - std_metrics[metric]
        print(f"{metric:<15} {ts_diff:>+12.4f} {ts_brier_diff:>+12.4f}")
    
    # Plot comparison
    print("\n6. Generating comparison plots...")
    plot_comparison(standard_results, ts_results, ts_brier_results, y_val, y_test)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

