"""
Example: Discovery Campaign Orchestrator

This example demonstrates how to use the DiscoveryCampaignOrchestrator to
run a systematic discovery campaign across multiple Ising model variants.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research import (
    DiscoveryCampaignOrchestrator,
    CampaignConfig,
)


def create_example_campaign_config():
    """Create example campaign configuration."""
    
    # Define target variants to explore
    target_variants = [
        '2d_ising_standard',  # Baseline for validation
        'long_range_ising_alpha_2.2',
        'diluted_ising_p_0.7',
    ]
    
    config = CampaignConfig(
        campaign_name="example_discovery_campaign",
        computational_budget=1000.0,  # GPU-hours
        target_variants=target_variants,
        validation_threshold=0.90,
        output_directory="results/campaigns",
        enable_parallel=False,
        max_parallel_tasks=2,
        checkpoint_interval=10,
    )
    
    return config


def run_campaign_example():
    """Run example discovery campaign."""
    
    print("=" * 80)
    print("Discovery Campaign Orchestrator Example")
    print("=" * 80)
    print()
    
    # Create campaign configuration
    print("Creating campaign configuration...")
    config = create_example_campaign_config()
    
    print(f"Campaign name: {config.campaign_name}")
    print(f"Computational budget: {config.computational_budget} GPU-hours")
    print(f"Target variants: {len(config.target_variants)}")
    for variant in config.target_variants:
        print(f"  - {variant}")
    print(f"Validation threshold: {config.validation_threshold}")
    print()
    
    # Initialize orchestrator
    print("Initializing campaign orchestrator...")
    orchestrator = DiscoveryCampaignOrchestrator(config)
    print(f"Output directory: {orchestrator.output_path}")
    print()
    
    # Note: In a real campaign, you would call:
    # results = orchestrator.run_campaign()
    #
    # For this example, we'll demonstrate the monitoring and checkpoint features
    
    print("=" * 80)
    print("Campaign Orchestrator Features")
    print("=" * 80)
    print()
    
    # Feature 1: Progress Monitoring
    print("1. Progress Monitoring")
    print("-" * 40)
    print("The orchestrator tracks:")
    print("  - Variants completed vs total")
    print("  - Current variant being explored")
    print("  - Discoveries found")
    print("  - Computational resources used")
    print("  - Time elapsed and estimated remaining")
    print()
    
    # Demonstrate resource monitoring
    print("Resource monitoring example:")
    resource_stats = orchestrator.monitor_resources()
    print(f"  Compute used: {resource_stats['computational_resources']['compute_used_gpu_hours']:.1f} GPU-hours")
    print(f"  Compute budget: {resource_stats['computational_resources']['compute_budget_gpu_hours']:.1f} GPU-hours")
    print(f"  Progress: {resource_stats['progress']['completion_percent']:.1f}%")
    print()
    
    # Feature 2: Checkpoint and Resumption
    print("2. Checkpoint and Resumption")
    print("-" * 40)
    print("The orchestrator can:")
    print("  - Save checkpoints at regular intervals")
    print("  - Resume from checkpoints after interruption")
    print("  - Handle failures gracefully")
    print()
    
    # Demonstrate checkpoint saving
    print("Checkpoint example:")
    checkpoint_path = orchestrator.save_checkpoint("example_checkpoint")
    print(f"  Checkpoint saved: {checkpoint_path}")
    print()
    
    # Feature 3: Main Campaign Execution Loop
    print("3. Main Campaign Execution Loop")
    print("-" * 40)
    print("For each target variant, the orchestrator:")
    print("  1. Executes exploration plan")
    print("  2. Applies validation pipeline")
    print("  3. Assesses for novel physics")
    print("  4. Generates publications if discoveries found")
    print("  5. Saves checkpoint")
    print("  6. Generates progress report")
    print()
    
    # Feature 4: Progress Reports
    print("4. Progress Reports")
    print("-" * 40)
    print("The orchestrator generates reports with:")
    print("  - Overall progress (variants completed)")
    print("  - Computational resource usage")
    print("  - Time estimation")
    print("  - Discoveries summary")
    print()
    
    # Demonstrate progress report generation
    print("Generating progress report...")
    orchestrator._generate_progress_report()
    report_path = orchestrator.output_path / 'progress_report.txt'
    print(f"  Progress report saved: {report_path}")
    print()
    
    # Display report content
    if report_path.exists():
        print("Progress Report Preview:")
        print("-" * 40)
        with open(report_path, 'r') as f:
            lines = f.readlines()
            # Print first 20 lines
            for line in lines[:20]:
                print(line.rstrip())
        print()
    
    print("=" * 80)
    print("Campaign Workflow")
    print("=" * 80)
    print()
    
    print("To run a complete campaign:")
    print()
    print("  # Create configuration")
    print("  config = CampaignConfig(...)")
    print()
    print("  # Initialize orchestrator")
    print("  orchestrator = DiscoveryCampaignOrchestrator(config)")
    print()
    print("  # Run campaign")
    print("  results = orchestrator.run_campaign()")
    print()
    print("  # Or resume from checkpoint")
    print("  results = orchestrator.resume_campaign(checkpoint_path)")
    print()
    
    print("=" * 80)
    print("Campaign Results")
    print("=" * 80)
    print()
    
    print("After completion, the campaign produces:")
    print()
    print("  results/campaigns/{campaign_name}/")
    print("  ├── variants/           # Exploration results per variant")
    print("  ├── discoveries/        # Validated discoveries")
    print("  ├── publications/       # Publication packages")
    print("  ├── checkpoints/        # Campaign checkpoints")
    print("  ├── provenance/         # Complete provenance tracking")
    print("  ├── campaign_config.json")
    print("  ├── campaign_results.json")
    print("  └── progress_report.txt")
    print()
    
    print("=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print()
    print(f"Campaign directory created: {orchestrator.output_path}")
    print()


def demonstrate_checkpoint_resumption():
    """Demonstrate checkpoint and resumption functionality."""
    
    print("=" * 80)
    print("Checkpoint and Resumption Example")
    print("=" * 80)
    print()
    
    # Create configuration
    config = create_example_campaign_config()
    
    # Initialize orchestrator
    orchestrator = DiscoveryCampaignOrchestrator(config)
    
    # Simulate some progress
    orchestrator.campaign_db['progress']['variants_completed'] = 1
    orchestrator.campaign_db['progress']['current_variant'] = 'long_range_ising_alpha_2.2'
    orchestrator.campaign_db['total_compute_hours'] = 150.0
    
    # Save checkpoint
    print("Saving checkpoint...")
    checkpoint_path = orchestrator.save_checkpoint("demo_checkpoint")
    print(f"Checkpoint saved: {checkpoint_path}")
    print()
    
    # Create new orchestrator and load checkpoint
    print("Creating new orchestrator and loading checkpoint...")
    new_orchestrator = DiscoveryCampaignOrchestrator(config)
    new_orchestrator.load_checkpoint(checkpoint_path)
    
    print("Checkpoint loaded successfully!")
    print(f"  Variants completed: {new_orchestrator.campaign_db['progress']['variants_completed']}")
    print(f"  Current variant: {new_orchestrator.campaign_db['progress']['current_variant']}")
    print(f"  Compute used: {new_orchestrator.campaign_db['total_compute_hours']:.1f} GPU-hours")
    print()
    
    print("Campaign can now be resumed from this checkpoint.")
    print()


if __name__ == "__main__":
    # Run main example
    run_campaign_example()
    
    print()
    print()
    
    # Demonstrate checkpoint/resumption
    demonstrate_checkpoint_resumption()
