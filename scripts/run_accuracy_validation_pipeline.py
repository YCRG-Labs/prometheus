#!/usr/bin/env python3
"""
Run Complete Accuracy Validation Pipeline

This script implements task 7.5: Create complete accuracy validation pipeline
that validates critical exponent accuracy > 90% for both 2D and 3D systems.

Usage:
    python scripts/run_accuracy_validation_pipeline.py --target-accuracy 90 --output-dir results/validation
"""

import argparse
import sys
import time
from pathlib import Path
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.accuracy_validation_pipeline import (
    AccuracyValidationPipeline, create_accuracy_validation_pipeline
)
from src.utils.logging_utils import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description='Run complete accuracy validation pipeline')
    parser.add_argument('--target-accuracy', type=float, default=90.0,
                       help='Target accuracy percentage (default: 90.0)')
    parser.add_argument('--output-dir', type=str, 
                       default='results/accuracy_validation',
                       help='Output directory for validation results')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Run validations in parallel (default: True)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run validations sequentially (overrides --parallel)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--systems', type=str, nargs='+',
                       choices=['ising_2d_small', 'ising_2d_medium', 'ising_3d_small', 'ising_3d_medium'],
                       help='Specific systems to validate (default: all)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger(__name__)
    
    # Determine parallel execution
    parallel_validation = args.parallel and not args.sequential
    
    print("=" * 80)
    print("PROMETHEUS ACCURACY VALIDATION PIPELINE")
    print("=" * 80)
    print(f"Target Accuracy: {args.target_accuracy}%")
    print(f"Output Directory: {args.output_dir}")
    print(f"Random Seed: {args.random_seed}")
    print(f"Parallel Execution: {parallel_validation}")
    if args.systems:
        print(f"Systems to Validate: {', '.join(args.systems)}")
    else:
        print("Systems to Validate: All configured systems")
    print()
    
    try:
        # Create validation pipeline
        logger.info("Creating accuracy validation pipeline")
        pipeline = create_accuracy_validation_pipeline(
            target_accuracy=args.target_accuracy,
            random_seed=args.random_seed,
            parallel_validation=parallel_validation,
            output_dir=args.output_dir
        )
        
        # Filter systems if specified
        if args.systems:
            original_configs = pipeline.system_configs.copy()
            pipeline.system_configs = {
                name: config for name, config in original_configs.items()
                if name in args.systems
            }
            logger.info(f"Filtered to {len(pipeline.system_configs)} systems: {list(pipeline.system_configs.keys())}")
        
        # Run validation
        logger.info("Starting complete accuracy validation")
        start_time = time.time()
        
        validation_result = pipeline.run_complete_validation()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print results summary
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"Overall Accuracy: {validation_result.overall_accuracy:.2f}%")
        print(f"Target Accuracy: {validation_result.target_accuracy_percent}%")
        print(f"Pipeline Success: {'✅ YES' if validation_result.pipeline_success else '❌ NO'}")
        print(f"Systems Meeting Target: {validation_result.systems_meeting_target}/{validation_result.total_systems}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Time: {validation_result.total_validation_time:.1f}s")
        print(f"  Peak Memory: {validation_result.peak_memory_usage:.0f}MB")
        print(f"  Models Converged: {'✅' if validation_result.all_models_converged else '❌'}")
        print(f"  Physics Consistent: {'✅' if validation_result.all_physics_consistent else '❌'}")
        
        print(f"\nSystem Results:")
        for system_name, system_result in validation_result.system_results.items():
            status = "✅" if system_result.meets_target_accuracy else "❌"
            print(f"  {system_name}: {system_result.overall_accuracy:.1f}% {status}")
        
        print(f"\nKey Recommendations:")
        for i, recommendation in enumerate(validation_result.recommendations[:5], 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\nOutput Files:")
        output_dir = Path(args.output_dir)
        print(f"  Results: {output_dir / 'validation_results.json'}")
        print(f"  Report: {output_dir / 'validation_report.txt'}")
        print(f"  Plots: {output_dir / 'validation_summary.png'}")
        print(f"  Detailed: {output_dir / 'detailed_accuracy.png'}")
        
        # Final assessment
        print("\n" + "=" * 80)
        if validation_result.pipeline_success:
            print("🎉 VALIDATION PIPELINE SUCCESSFUL!")
            print("   All systems meet the target accuracy requirements.")
            print("   The VAE-based critical exponent extraction approach")
            print("   demonstrates excellent performance across multiple systems.")
        else:
            print("⚠️  VALIDATION PIPELINE NEEDS IMPROVEMENT")
            print(f"   Current accuracy ({validation_result.overall_accuracy:.1f}%) is below")
            print(f"   target ({validation_result.target_accuracy_percent}%).")
            print("   Review recommendations for improvement strategies.")
        
        print(f"\n📊 Complete validation report available in: {args.output_dir}")
        print("=" * 80)
        
        # Exit with appropriate code
        if validation_result.pipeline_success:
            logger.info("Validation pipeline completed successfully")
            sys.exit(0)
        else:
            logger.warning("Validation pipeline did not meet target accuracy")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Validation pipeline failed: {e}")
        print(f"\n❌ Validation pipeline failed: {e}")
        
        # Print traceback in verbose mode
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()