"""
Example demonstrating the Bonferroni-Bootstrap Hybrid Validator.

This script shows how the hybrid validator combines:
- Bonferroni correction for specificity (prevents false positives)
- Bootstrap resampling for sensitivity (handles non-normal distributions)

The validator adaptively selects the appropriate method based on data
characteristics, providing both statistical rigor and robustness.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.hybrid_validator import HybridValidator


def example_1_bonferroni_validation():
    """Example 1: Validation with Bonferroni correction."""
    print("=" * 80)
    print("Example 1: Bonferroni Validation (Normal Data)")
    print("=" * 80)
    
    # Initialize validator
    validator = HybridValidator(alpha=0.05, n_bootstrap=1000)
    
    # Hypothesis: Variant belongs to 2D Ising class
    print("\n1. Hypothesis: Variant belongs to 2D Ising universality class")
    predicted_exponents = {
        'beta': 0.125,
        'nu': 1.0,
        'gamma': 1.75
    }
    
    # Measured values (close to predictions, normal distribution)
    measured_exponents = {
        'beta': 0.123,
        'nu': 0.988,
        'gamma': 1.761
    }
    
    measured_errors = {
        'beta': 0.003,
        'nu': 0.016,
        'gamma': 0.024
    }
    
    print("\n2. Predicted vs Measured:")
    for exp_name in predicted_exponents.keys():
        pred = predicted_exponents[exp_name]
        meas = measured_exponents[exp_name]
        err = measured_errors[exp_name]
        print(f"   {exp_name}: {pred:.3f} vs {meas:.3f} ± {err:.3f}")
    
    # Validate with Bonferroni
    result = validator.validate_hypothesis(
        hypothesis_id='2d_ising_test',
        predicted_exponents=predicted_exponents,
        measured_exponents=measured_exponents,
        measured_errors=measured_errors,
        force_method='bonferroni'
    )
    
    print(f"\n3. Bonferroni Validation Result:")
    print(f"   Validated: {result.validated}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Method: {result.method_used}")
    print(f"   Recommendation: {result.recommendation}")
    
    # Show details
    print(f"\n4. Bonferroni Details:")
    bonf = result.bonferroni_results
    print(f"   Original α: {bonf['alpha_original']:.3f}")
    print(f"   Adjusted α: {bonf['alpha_adjusted']:.3f} (for {bonf['n_comparisons']} comparisons)")
    
    for exp_name, exp_result in bonf['exponent_results'].items():
        print(f"\n   {exp_name}:")
        print(f"      Z-score: {exp_result['z_score']:.2f}")
        print(f"      P-value: {exp_result['p_value']:.4f}")
        print(f"      P-value (adjusted): {exp_result['p_value_adjusted']:.4f}")
        print(f"      Validated: {exp_result['validated']}")


def example_2_bootstrap_validation():
    """Example 2: Validation with bootstrap resampling."""
    print("\n" + "=" * 80)
    print("Example 2: Bootstrap Validation (Non-Normal Data)")
    print("=" * 80)
    
    # Initialize validator
    validator = HybridValidator(alpha=0.05, n_bootstrap=1000)
    
    # Hypothesis: Variant belongs to mean field class
    print("\n1. Hypothesis: Variant belongs to Mean Field universality class")
    predicted_exponents = {
        'beta': 0.5,
        'nu': 0.5
    }
    
    # Generate non-normal data (skewed distribution)
    np.random.seed(42)
    beta_data = np.random.gamma(shape=2, scale=0.25, size=50)  # Skewed
    nu_data = np.random.gamma(shape=2, scale=0.25, size=50)
    
    measured_data = {
        'beta': beta_data,
        'nu': nu_data
    }
    
    measured_exponents = {
        'beta': np.mean(beta_data),
        'nu': np.mean(nu_data)
    }
    
    measured_errors = {
        'beta': np.std(beta_data) / np.sqrt(len(beta_data)),
        'nu': np.std(nu_data) / np.sqrt(len(nu_data))
    }
    
    print("\n2. Predicted vs Measured:")
    for exp_name in predicted_exponents.keys():
        pred = predicted_exponents[exp_name]
        meas = measured_exponents[exp_name]
        err = measured_errors[exp_name]
        print(f"   {exp_name}: {pred:.3f} vs {meas:.3f} ± {err:.3f}")
    
    # Analyze data characteristics
    print("\n3. Data Characteristics:")
    characteristics = validator.analyze_data_characteristics(beta_data)
    print(f"   Sample size: {characteristics.n_samples}")
    print(f"   Normal: {characteristics.is_normal} (p={characteristics.normality_p_value:.4f})")
    print(f"   Outliers: {characteristics.has_outliers}")
    print(f"   Recommendation: {characteristics.recommendation}")
    
    # Validate with bootstrap
    result = validator.validate_hypothesis(
        hypothesis_id='mean_field_test',
        predicted_exponents=predicted_exponents,
        measured_exponents=measured_exponents,
        measured_errors=measured_errors,
        measured_data=measured_data,
        force_method='bootstrap'
    )
    
    print(f"\n4. Bootstrap Validation Result:")
    print(f"   Validated: {result.validated}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Method: {result.method_used}")
    print(f"   Recommendation: {result.recommendation}")
    
    # Show bootstrap details
    print(f"\n5. Bootstrap Details:")
    boot = result.bootstrap_results
    for exp_name, exp_result in boot['exponent_results'].items():
        print(f"\n   {exp_name}:")
        print(f"      Measured: {exp_result['measured_mean']:.3f} ± {exp_result['measured_std']:.3f}")
        print(f"      95% CI: [{exp_result['ci_lower']:.3f}, {exp_result['ci_upper']:.3f}]")
        print(f"      Predicted: {exp_result['predicted']:.3f}")
        print(f"      In CI: {exp_result['validated']}")


def example_3_adaptive_selection():
    """Example 3: Adaptive method selection based on data."""
    print("\n" + "=" * 80)
    print("Example 3: Adaptive Method Selection")
    print("=" * 80)
    
    # Initialize validator
    validator = HybridValidator(alpha=0.05, n_bootstrap=1000)
    
    print("\n1. Testing with different data characteristics...")
    
    # Case 1: Normal data, adequate sample size
    print("\n   Case 1: Normal data, n=50")
    np.random.seed(42)
    normal_data = np.random.normal(0.5, 0.05, 50)
    char1 = validator.analyze_data_characteristics(normal_data)
    print(f"      Normal: {char1.is_normal}, Outliers: {char1.has_outliers}")
    print(f"      → Recommendation: {char1.recommendation}")
    
    # Case 2: Non-normal data (skewed)
    print("\n   Case 2: Skewed data, n=50")
    skewed_data = np.random.gamma(2, 0.25, 50)
    char2 = validator.analyze_data_characteristics(skewed_data)
    print(f"      Normal: {char2.is_normal}, Outliers: {char2.has_outliers}")
    print(f"      → Recommendation: {char2.recommendation}")
    
    # Case 3: Data with outliers
    print("\n   Case 3: Normal data with outliers, n=50")
    outlier_data = np.random.normal(0.5, 0.05, 50)
    outlier_data[0] = 2.0  # Add outlier
    char3 = validator.analyze_data_characteristics(outlier_data)
    print(f"      Normal: {char3.is_normal}, Outliers: {char3.has_outliers}")
    print(f"      → Recommendation: {char3.recommendation}")
    
    # Case 4: Small sample size
    print("\n   Case 4: Normal data, n=20 (small)")
    small_data = np.random.normal(0.5, 0.05, 20)
    char4 = validator.analyze_data_characteristics(small_data)
    print(f"      Normal: {char4.is_normal}, Adequate size: {char4.sample_size_adequate}")
    print(f"      → Recommendation: {char4.recommendation}")
    
    print("\n2. Key Insight:")
    print("   Validator automatically selects appropriate method:")
    print("   - Bonferroni: Normal data, no outliers, adequate sample")
    print("   - Bootstrap: Non-normal, outliers, or to be conservative")
    print("   - Both: Small samples or when extra confidence needed")


def example_4_method_comparison():
    """Example 4: Compare Bonferroni and bootstrap side-by-side."""
    print("\n" + "=" * 80)
    print("Example 4: Method Comparison")
    print("=" * 80)
    
    # Initialize validator
    validator = HybridValidator(alpha=0.05, n_bootstrap=1000)
    
    # Create data with slight non-normality
    np.random.seed(42)
    beta_data = np.random.gamma(shape=5, scale=0.1, size=50)  # Slightly skewed
    nu_data = np.random.gamma(shape=5, scale=0.2, size=50)
    
    predicted_exponents = {
        'beta': 0.5,
        'nu': 1.0
    }
    
    measured_exponents = {
        'beta': np.mean(beta_data),
        'nu': np.mean(nu_data)
    }
    
    measured_errors = {
        'beta': np.std(beta_data) / np.sqrt(len(beta_data)),
        'nu': np.std(nu_data) / np.sqrt(len(nu_data))
    }
    
    measured_data = {
        'beta': beta_data,
        'nu': nu_data
    }
    
    print("\n1. Comparing methods on slightly skewed data...")
    
    # Compare methods
    comparison = validator.compare_methods(
        predicted_exponents,
        measured_exponents,
        measured_errors,
        measured_data
    )
    
    print(f"\n2. Data Characteristics:")
    char = comparison['data_characteristics']
    print(f"   Sample size: {char.n_samples}")
    print(f"   Normal: {char.is_normal} (p={char.normality_p_value:.4f})")
    print(f"   Recommendation: {char.recommendation}")
    
    print(f"\n3. Overall Agreement:")
    print(f"   Methods agree: {comparison['agreement']}")
    print(f"   Bonferroni validated: {comparison['bonferroni']['all_validated']}")
    print(f"   Bootstrap validated: {comparison['bootstrap']['all_validated']}")
    
    print(f"\n4. Exponent-by-Exponent Comparison:")
    for exp_name, exp_comp in comparison['exponent_comparison'].items():
        print(f"\n   {exp_name}:")
        print(f"      Bonferroni: {exp_comp['bonferroni_validated']} (p={exp_comp['bonferroni_p_value']:.4f})")
        print(f"      Bootstrap: {exp_comp['bootstrap_validated']} (p={exp_comp['bootstrap_p_value']:.4f})")
        print(f"      Agree: {exp_comp['agree']}")
        print(f"      Bonferroni more conservative: {exp_comp['bonferroni_more_conservative']}")


def example_5_disagreement_case():
    """Example 5: Case where methods disagree."""
    print("\n" + "=" * 80)
    print("Example 5: Method Disagreement")
    print("=" * 80)
    
    # Initialize validator
    validator = HybridValidator(alpha=0.05, n_bootstrap=1000)
    
    # Create data where Bonferroni might reject but bootstrap accepts
    # (due to outliers affecting standard error)
    np.random.seed(42)
    beta_data = np.concatenate([
        np.random.normal(0.48, 0.02, 45),  # Main distribution
        np.array([0.3, 0.7, 0.25, 0.75, 0.35])  # Outliers
    ])
    
    predicted_exponents = {'beta': 0.5}
    measured_exponents = {'beta': np.mean(beta_data)}
    measured_errors = {'beta': np.std(beta_data) / np.sqrt(len(beta_data))}
    measured_data = {'beta': beta_data}
    
    print("\n1. Data with outliers:")
    print(f"   Mean: {measured_exponents['beta']:.3f}")
    print(f"   Std: {np.std(beta_data):.3f}")
    print(f"   Outliers present: {np.sum(np.abs(beta_data - np.mean(beta_data)) > 3*np.std(beta_data))}")
    
    # Validate with both methods
    result = validator.validate_hypothesis(
        hypothesis_id='outlier_test',
        predicted_exponents=predicted_exponents,
        measured_exponents=measured_exponents,
        measured_errors=measured_errors,
        measured_data=measured_data,
        force_method='both'
    )
    
    print(f"\n2. Hybrid Validation Result:")
    print(f"   Validated: {result.validated}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Methods agree: {result.agreement}")
    print(f"   Recommendation: {result.recommendation}")
    
    print(f"\n3. Method Details:")
    if result.bonferroni_results:
        bonf = result.bonferroni_results['exponent_results']['beta']
        print(f"   Bonferroni: validated={bonf['validated']}, p={bonf['p_value']:.4f}")
    
    if result.bootstrap_results:
        boot = result.bootstrap_results['exponent_results']['beta']
        print(f"   Bootstrap: validated={boot['validated']}, p={boot['p_value']:.4f}")
        print(f"   Bootstrap CI: [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
    
    print(f"\n4. Interpretation:")
    print("   When methods disagree, bootstrap is typically more reliable")
    print("   for non-normal data or data with outliers, as it makes no")
    print("   distributional assumptions.")


def example_6_multiple_comparisons():
    """Example 6: Demonstrate Bonferroni correction for multiple comparisons."""
    print("\n" + "=" * 80)
    print("Example 6: Multiple Comparisons Correction")
    print("=" * 80)
    
    # Initialize validator
    validator = HybridValidator(alpha=0.05, n_bootstrap=1000)
    
    print("\n1. Testing 5 exponents simultaneously...")
    
    # Hypothesis with 5 exponents
    predicted_exponents = {
        'beta': 0.326,
        'nu': 0.630,
        'gamma': 1.237,
        'alpha': 0.110,
        'delta': 4.789
    }
    
    # Measured values (some match, some don't)
    np.random.seed(42)
    measured_exponents = {
        'beta': 0.320,
        'nu': 0.666,
        'gamma': 1.259,
        'alpha': 0.105,
        'delta': 4.8
    }
    
    measured_errors = {
        'beta': 0.007,
        'nu': 0.013,
        'gamma': 0.020,
        'alpha': 0.010,
        'delta': 0.05
    }
    
    # Validate
    result = validator.validate_hypothesis(
        hypothesis_id='multiple_test',
        predicted_exponents=predicted_exponents,
        measured_exponents=measured_exponents,
        measured_errors=measured_errors,
        force_method='bonferroni'
    )
    
    print(f"\n2. Bonferroni Correction:")
    bonf = result.bonferroni_results
    print(f"   Number of comparisons: {bonf['n_comparisons']}")
    print(f"   Original α: {bonf['alpha_original']:.3f}")
    print(f"   Adjusted α: {bonf['alpha_adjusted']:.4f}")
    print(f"   → Each test must pass at {bonf['alpha_adjusted']:.4f} level")
    
    print(f"\n3. Individual Exponent Results:")
    for exp_name, exp_result in bonf['exponent_results'].items():
        pred = exp_result['predicted']
        meas = exp_result['measured']
        p_val = exp_result['p_value']
        p_adj = exp_result['p_value_adjusted']
        validated = exp_result['validated']
        
        print(f"\n   {exp_name}:")
        print(f"      Predicted: {pred:.3f}, Measured: {meas:.3f}")
        print(f"      P-value: {p_val:.4f}")
        print(f"      P-value (adjusted): {p_adj:.4f}")
        print(f"      Validated: {validated}")
    
    print(f"\n4. Overall Result:")
    print(f"   All validated: {result.validated}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   → Bonferroni prevents false positives from multiple testing")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BONFERRONI-BOOTSTRAP HYBRID VALIDATOR EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating complementary strengths of two validation methods:")
    print("- Bonferroni: Conservative, controls family-wise error rate")
    print("- Bootstrap: Robust, handles non-normal distributions")
    
    example_1_bonferroni_validation()
    example_2_bootstrap_validation()
    example_3_adaptive_selection()
    example_4_method_comparison()
    example_5_disagreement_case()
    example_6_multiple_comparisons()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. COMPLEMENTARY STRENGTHS:
   - Bonferroni: Provides specificity (prevents false positives)
   - Bootstrap: Provides sensitivity (handles non-normality)
   - Hybrid: Combines both for optimal validation

2. ADAPTIVE SELECTION:
   - Normal data + adequate sample → Bonferroni
   - Non-normal or outliers → Bootstrap
   - Small sample or extra confidence → Both methods

3. BONFERRONI CORRECTION:
   - Adjusts significance level for multiple comparisons
   - α_adjusted = α / n_comparisons
   - Prevents false positives from testing many exponents

4. BOOTSTRAP ROBUSTNESS:
   - Makes no distributional assumptions
   - Robust to outliers and skewness
   - Provides confidence intervals directly from data

5. DISAGREEMENT HANDLING:
   - When methods disagree, trust bootstrap for non-normal data
   - Disagreement often indicates violated assumptions
   - Hybrid approach uses more conservative result

6. PRACTICAL RECOMMENDATIONS:
   - Use Bonferroni for clean, normal data
   - Use bootstrap for real-world messy data
   - Use both when stakes are high
   - Let validator choose automatically based on data

This hybrid approach provides both statistical rigor (Bonferroni)
and robustness to real-world data issues (bootstrap), making it
ideal for validating physics hypotheses where data may not be
perfectly normal.
""")
    
    print("=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
