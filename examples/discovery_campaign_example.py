"""
Example: Discovery Campaign Infrastructure

This example demonstrates the discovery campaign orchestrator that coordinates
systematic exploration of multiple Ising model variants to discover novel physics.

The campaign infrastructure provides:
- Target variant selection and prioritization
- Exploration planning and execution
- Results database and provenance tracking
- Computational resource management
- Checkpoint and resumption capabilities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research import (
    DiscoveryCampaignOrchestrator,
    CampaignConfig,
    ModelVariantRegistry,
)


def main():
    """Run discovery campaign infrastructure example."""
    
    print("=" * 80)
    print("Discovery Campaign Infrastructure Example")
    print("=" * 80)
    
    # Create campaign configuration
    # This is a minimal example - real campaigns would explore more variants
    campaign_config = CampaignConfig(
        campaign_name='example_campaign',
        computational_budget=1000.0,  # GPU-hours
        target_variants=['ising_2d', 'ising_3d'],  # Start with standard models
        validation_threshold=0.90,
        output_directory='results/discovery_campaign',
        enable_parallel=False,  # Set to True for parallel execution
        max_parallel_tasks=4,
        checkpoint_interval=10,
    )
    
    print(f"\nCampaign Configuration:")
    print(f"  Name: {campaign_config.campaign_name}")
    print(f"  Budget: {campaign_config.computational_budget} GPU-hours")
    print(f"  Target variants: {campaign_config.target_variants}")
    print(f"  Validation threshold: {campaign_config.validation_threshold}")
    
    # Initialize campaign orchestrator
    print(f"\nInitializing campaign orchestrator...")
    orchestrator = DiscoveryCampaignOrchestrator(campaign_config)
    
    print(f"\nCampaign infrastructure created:")
    print(f"  Output directory: {orchestrator.output_path}")
    print(f"  Subdirectories:")
    print(f"    - variants/     (per-variant exploration results)")
    print(f"    - discoveries/  (validated discoveries)")
    print(f"    - publications/ (publication-ready outputs)")
    print(f"    - checkpoints/  (campaign checkpoints)")
    print(f"    - provenance/   (complete provenance tracking)")
    
    # Demonstrate directory structure
    print(f"\nDirectory structure created:")
    for subdir in ['variants', 'discoveries', 'publications', 'checkpoints', 'provenance']:
        subdir_path = orchestrator.output_path / subdir
        print(f"  ✓ {subdir_path}")
    
    # Show campaign database structure
    print(f"\nCampaign database initialized:")
    print(f"  - variants: {{}}")
    print(f"  - discoveries: []")
    print(f"  - provenance: []")
    print(f"  - start_time: None")
    print(f"  - total_compute_hours: 0.0")
    
    # Note: We don't actually run the campaign in this example
    # as it would take significant computational resources
    print(f"\n" + "=" * 80)
    print("Infrastructure Setup Complete")
    print("=" * 80)
    print(f"\nTo run the actual campaign, call:")
    print(f"  results = orchestrator.run_campaign()")
    print(f"\nThis will:")
    print(f"  1. Explore each target variant systematically")
    print(f"  2. Apply all validation patterns to findings")
    print(f"  3. Track complete provenance for reproducibility")
    print(f"  4. Save checkpoints for resumption")
    print(f"  5. Generate publication-ready outputs")
    
    print(f"\nCampaign infrastructure is ready for use!")


if __name__ == '__main__':
    main()
