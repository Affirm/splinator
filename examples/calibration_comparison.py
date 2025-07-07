"""
Comprehensive comparison of Splinator vs scikit-learn calibration methods
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from splinator.estimators import LinearSplineLogisticRegression, CDFSplineCalibrator
from splinator.metrics import expected_calibration_error, spiegelhalters_z_statistic
import time
import warnings
warnings.filterwarnings('ignore')

def evaluate_calibrator(y_true, y_pred_calibrated, method_name):
    """Evaluate calibration performance with multiple metrics"""
    results = {
        'method': method_name,
        'brier_score': brier_score_loss(y_true, y_pred_calibrated),
        'log_loss': log_loss(y_true, y_pred_calibrated),
        'ece': expected_calibration_error(y_true, y_pred_calibrated, n_bins=10),
        'spiegelhalter_z': abs(spiegelhalters_z_statistic(y_true, y_pred_calibrated))
    }
    return results

def run_comparison(n_samples=50000, n_features=20, n_knots=50):
    """Run comprehensive comparison"""
    print(f"\nRunning comparison with {n_samples} samples, {n_features} features, {n_knots} knots")
    
    # Generate data
    # Ensure n_informative + n_redundant < n_features
    n_informative = max(1, int(n_features * 0.6))  # 60% informative
    n_redundant = max(1, int(n_features * 0.2))    # 20% redundant
    
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_informative=n_informative, n_redundant=n_redundant, 
                              flip_y=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Train base classifier
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get uncalibrated predictions
    probas_cal = clf.predict_proba(X_cal)
    probas_test = clf.predict_proba(X_test)
    pred_cal = probas_cal[:, 1]
    pred_test = probas_test[:, 1]
    
    results = []
    
    # Uncalibrated baseline
    results.append(evaluate_calibrator(y_test, pred_test, 'Uncalibrated'))
    
    # 1. Sigmoid Calibration (Platt Scaling)
    start = time.time()
    lr = LogisticRegression()
    lr.fit(pred_cal.reshape(-1, 1), y_cal)
    lr_calibrated = lr.predict_proba(pred_test.reshape(-1, 1))[:, 1]
    sigmoid_time = time.time() - start
    result = evaluate_calibrator(y_test, lr_calibrated, 'Sigmoid (sklearn)')
    result['fit_time'] = sigmoid_time
    results.append(result)
    
    # 2. Isotonic Regression
    start = time.time()
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(pred_cal, y_cal)
    ir_calibrated = ir.predict(pred_test)
    isotonic_time = time.time() - start
    result = evaluate_calibrator(y_test, ir_calibrated, 'Isotonic (sklearn)')
    result['fit_time'] = isotonic_time
    results.append(result)
    
    # 3. CDF Spline Calibrator (Splinator)
    start = time.time()
    cdf_cal = CDFSplineCalibrator()  # num_knots=6 by default
    cdf_cal.fit(probas_cal, y_cal)
    cdf_calibrated_probas = cdf_cal.transform(probas_test)
    cdf_calibrated = cdf_calibrated_probas[:, 1]
    cdf_time = time.time() - start
    result = evaluate_calibrator(y_test, cdf_calibrated, 'CDFSplineCalibrator')
    result['fit_time'] = cdf_time
    results.append(result)
    
    # 4. Linear Spline Logistic Regression (Splinator)
    start = time.time()
    lslr = LinearSplineLogisticRegression(
        n_knots=n_knots,
        monotonicity="increasing",
        method='SLSQP',
        C=100,
        two_stage_fitting_initial_size=min(2000, len(y_cal))
    )
    lslr.fit(pred_cal.reshape(-1, 1), y_cal)
    lslr_calibrated = lslr.predict(pred_test.reshape(-1, 1))
    lslr_time = time.time() - start
    result = evaluate_calibrator(y_test, lslr_calibrated, f'LSLR (n_knots={n_knots})')
    result['fit_time'] = lslr_time
    results.append(result)
    
    # Additional LSLR configurations
    for n_k in [20, 100]:
        if n_k != n_knots:
            start = time.time()
            lslr2 = LinearSplineLogisticRegression(
                n_knots=n_k,
                monotonicity="increasing",
                method='SLSQP',
                C=100,
                two_stage_fitting_initial_size=min(2000, len(y_cal))
            )
            lslr2.fit(pred_cal.reshape(-1, 1), y_cal)
            lslr2_calibrated = lslr2.predict(pred_test.reshape(-1, 1))
            lslr2_time = time.time() - start
            result = evaluate_calibrator(y_test, lslr2_calibrated, f'LSLR (n_knots={n_k})')
            result['fit_time'] = lslr2_time
            results.append(result)
    
    return pd.DataFrame(results)

# Run comparisons with different dataset sizes
results_list = []

for n_samples in [10000, 50000]:
    for n_features in [10, 30]:
        df = run_comparison(n_samples=n_samples, n_features=n_features, n_knots=50)
        df['n_samples'] = n_samples
        df['n_features'] = n_features
        results_list.append(df)

# Combine all results
all_results = pd.concat(results_list, ignore_index=True)

# Display summary
print("\n" + "="*80)
print("CALIBRATION COMPARISON RESULTS")
print("="*80)
print("\nLower is better for: brier_score, log_loss, ece, spiegelhalter_z")
print("\nAverage performance across all experiments:")
summary = all_results.groupby('method')[['brier_score', 'log_loss', 'ece', 'spiegelhalter_z', 'fit_time']].mean()
print(summary.round(4))

print("\n" + "="*80)
print("RELATIVE PERFORMANCE (% improvement over Uncalibrated)")
print("="*80)

uncalibrated_scores = all_results[all_results['method'] == 'Uncalibrated'][['brier_score', 'log_loss', 'ece', 'spiegelhalter_z']].mean()
for method in ['Sigmoid (sklearn)', 'Isotonic (sklearn)', 'CDFSplineCalibrator', 'LSLR (n_knots=50)']:
    method_scores = all_results[all_results['method'] == method][['brier_score', 'log_loss', 'ece', 'spiegelhalter_z']].mean()
    improvement = (uncalibrated_scores - method_scores) / uncalibrated_scores * 100
    print(f"\n{method}:")
    for metric in improvement.index:
        print(f"  {metric}: {improvement[metric]:.1f}% improvement")

# Save detailed results
all_results.to_csv('calibration_comparison_results.csv', index=False)
print("\nDetailed results saved to 'calibration_comparison_results.csv'") 