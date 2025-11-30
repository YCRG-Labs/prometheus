"""
Example: Error Handling and Recovery - Simple Examples

This example demonstrates the error handling and recovery capabilities without
running the full discovery pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research import (
    ErrorRecoveryManager,
    ErrorContext,
    ErrorCategory,
)


def example_1_manual_error_recovery():
    """Example 1: Manual error recovery using ErrorRecoveryManager."""
    print("=" * 80)
    print("Example 1: Manual Error Recovery")
    print("=" * 80)
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(max_retries=3)
    
    # Simulate different types of errors and recovery strategies
    
    # 1. Simulation error - non-equilibrated system
    print("\n1. Handling simulation error (non-equilibrated system):")
    sim_error = Exception("System not equilibrated after 1000 steps")
    sim_params = {
        'n_equilibration': 1000,
        'n_steps_between': 10,
        'lattice_size': 32
    }
    strategy = recovery_manager.handle_simulation_error(sim_error, sim_params)
    print(f"   Action: {strategy.action.value}")
    print(f"   Message: {strategy.message}")
    if strategy.parameter_adjustments:
        print(f"   Adjustments: {strategy.parameter_adjustments}")
    
    # 2. VAE training error - poor convergence
    print("\n2. Handling VAE training error (poor convergence):")
    vae_error = Exception("Loss not converging after 50 epochs")
    vae_params = {
        'learning_rate': 1e-3,
        'n_epochs': 50,
        'patience': 10
    }
    strategy = recovery_manager.handle_vae_error(vae_error, vae_params)
    print(f"   Action: {strategy.action.value}")
    print(f"   Message: {strategy.message}")
    if strategy.parameter_adjustments:
        print(f"   Adjustments: {strategy.parameter_adjustments}")
    
    # 3. VAE training error - GPU memory
    print("\n3. Handling VAE training error (GPU memory):")
    gpu_error = Exception("CUDA out of memory")
    gpu_params = {
        'batch_size': 128,
        'use_gpu': True
    }
    strategy = recovery_manager.handle_vae_error(gpu_error, gpu_params)
    print(f"   Action: {strategy.action.value}")
    print(f"   Message: {strategy.message}")
    if strategy.parameter_adjustments:
        print(f"   Adjustments: {strategy.parameter_adjustments}")
    
    # 4. Analysis error - poor fit quality
    print("\n4. Handling analysis error (poor fit quality):")
    analysis_error = Exception("Poor fit quality: R² = 0.45")
    analysis_params = {
        'r_squared': 0.45,
        'n_temperatures': 20,
        'n_samples': 100
    }
    strategy = recovery_manager.handle_analysis_error(analysis_error, analysis_params)
    print(f"   Action: {strategy.action.value}")
    print(f"   Message: {strategy.message}")
    if strategy.parameter_adjustments:
        print(f"   Adjustments: {strategy.parameter_adjustments}")
    
    # 5. Analysis error - Tc detection failure
    print("\n5. Handling analysis error (Tc detection failure):")
    tc_error = Exception("Critical temperature not found in range")
    tc_params = {
        't_min': 2.0,
        't_max': 3.5,
        'n_temperatures': 20
    }
    strategy = recovery_manager.handle_analysis_error(tc_error, tc_params)
    print(f"   Action: {strategy.action.value}")
    print(f"   Message: {strategy.message}")
    if strategy.parameter_adjustments:
        print(f"   Adjustments: {strategy.parameter_adjustments}")
    
    print("\n" + "=" * 80)


def example_2_error_categorization():
    """Example 2: Error categorization and statistics."""
    print("=" * 80)
    print("Example 2: Error Categorization and Statistics")
    print("=" * 80)
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager()
    
    # Simulate various errors
    errors = [
        (Exception("System not equilibrated"), "simulation"),
        (Exception("CUDA out of memory"), "vae_training"),
        (Exception("Poor fit quality"), "analysis"),
        (Exception("Invalid configuration"), "simulation"),
        (Exception("NaN in loss"), "vae_training"),
        (Exception("Insufficient data points"), "analysis"),
        (Exception("Numerical overflow"), "simulation"),
    ]
    
    print("\nCategorizing errors:")
    for error, stage in errors:
        category = recovery_manager.categorize_error(error, stage)
        print(f"  {stage:15s} | {str(error):30s} → {category.value}")
        
        # Add to history
        context = ErrorContext(
            category=category,
            error=error,
            parameters={},
            attempt_number=1
        )
        recovery_manager.error_history.append(context)
    
    # Get statistics
    stats = recovery_manager.get_error_statistics()
    print(f"\nError Statistics:")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  By category:")
    for category, count in stats['by_category'].items():
        print(f"    {category}: {count}")
    print(f"  Most common: {stats['most_common']['category']} "
          f"({stats['most_common']['count']} occurrences)")
    
    print("\n" + "=" * 80)


def example_3_recovery_with_retry():
    """Example 3: Recovery with retry logic."""
    print("=" * 80)
    print("Example 3: Recovery with Retry Logic")
    print("=" * 80)
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(max_retries=3)
    
    # Simulate a function that fails initially but succeeds after parameter adjustment
    attempt_count = [0]  # Use list to allow modification in nested function
    
    def flaky_function(params):
        """Function that fails on first attempt but succeeds after adjustment."""
        attempt_count[0] += 1
        print(f"  Attempt {attempt_count[0]}: params = {params}")
        
        if attempt_count[0] == 1:
            # First attempt fails
            raise Exception("Convergence failure")
        else:
            # Subsequent attempts succeed
            print(f"  Success!")
            return {"result": "success", "params": params}
    
    # Create error context
    print("\nSimulating recovery with retry:")
    context = ErrorContext(
        category=ErrorCategory.VAE_TRAINING_ERROR,
        error=Exception("Convergence failure"),
        parameters={'learning_rate': 1e-3, 'n_epochs': 50},
        attempt_number=1
    )
    
    # Execute recovery
    result = recovery_manager.execute_recovery(context, flaky_function)
    
    if result:
        print(f"\nRecovery successful!")
        print(f"  Result: {result}")
    else:
        print(f"\nRecovery failed after all retries")
    
    print("\n" + "=" * 80)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ERROR HANDLING AND RECOVERY - SIMPLE EXAMPLES")
    print("=" * 80 + "\n")
    
    # Run examples
    example_1_manual_error_recovery()
    print("\n")
    
    example_2_error_categorization()
    print("\n")
    
    example_3_recovery_with_retry()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
