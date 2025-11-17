"""
Example demonstrating the Enhanced 3D Monte Carlo Simulator.

This example shows how to use the new 3D Monte Carlo framework
for Ising model simulations with 6-neighbor interactions.
"""

import numpy as np
import sys
import os

# Add src to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.enhanced_monte_carlo import (
    EnhancedMonteCarloSimulator, 
    create_3d_simulator,
    create_2d_simulator
)


def demonstrate_3d_simulation():
    """Demonstrate 3D Ising model simulation."""
    print("=== 3D Ising Model Simulation ===")
    
    # Create 3D simulator
    lattice_size = (8, 8, 8)  # 8x8x8 lattice
    temperature = 4.5  # Near critical temperature for 3D Ising (Tc ≈ 4.511)
    
    simulator = create_3d_simulator(lattice_size, temperature)
    
    print(f"Created 3D Ising simulator:")
    print(f"  Lattice size: {lattice_size}")
    print(f"  Temperature: {temperature}")
    print(f"  Total sites: {np.prod(lattice_size)}")
    print(f"  Neighbors per site: 6")
    
    # Initial state
    initial_config = simulator.get_configuration()
    print(f"\nInitial state:")
    print(f"  Magnetization: {initial_config.magnetization:.4f}")
    print(f"  Energy per spin: {initial_config.energy:.4f}")
    
    # Run equilibration
    print(f"\nRunning equilibration...")
    n_sweeps = 100
    
    for sweep in range(n_sweeps):
        accepted_moves = simulator.sweep()
        
        if sweep % 20 == 0:
            config = simulator.get_configuration()
            print(f"  Sweep {sweep:3d}: M={config.magnetization:6.3f}, "
                  f"E/spin={config.energy:7.4f}, accepted={accepted_moves:3d}")
    
    # Final state
    final_config = simulator.get_configuration()
    print(f"\nFinal state after {n_sweeps} sweeps:")
    print(f"  Magnetization: {final_config.magnetization:.4f}")
    print(f"  Energy per spin: {final_config.energy:.4f}")
    print(f"  Acceptance rate: {simulator.get_acceptance_rate():.3f}")
    
    return simulator


def compare_2d_vs_3d():
    """Compare 2D and 3D simulations at similar conditions."""
    print("\n=== 2D vs 3D Comparison ===")
    
    # Create simulators with similar total sites
    sim_2d = create_2d_simulator((16, 16), temperature=2.3)  # 256 sites, near 2D Tc ≈ 2.269
    sim_3d = create_3d_simulator((8, 8, 4), temperature=4.5)  # 256 sites, near 3D Tc ≈ 4.511
    
    print(f"2D simulator: {sim_2d.lattice_size}, T={sim_2d.temperature}")
    print(f"3D simulator: {sim_3d.lattice_size}, T={sim_3d.temperature}")
    
    # Run short simulations
    n_sweeps = 50
    
    for sim, label in [(sim_2d, "2D"), (sim_3d, "3D")]:
        print(f"\n{label} simulation:")
        
        for sweep in range(n_sweeps):
            sim.sweep()
            
            if sweep % 10 == 0:
                config = sim.get_configuration()
                print(f"  Sweep {sweep:2d}: M={config.magnetization:6.3f}, "
                      f"E/spin={config.energy:7.4f}")
        
        final_config = sim.get_configuration()
        print(f"  Final: M={final_config.magnetization:6.3f}, "
              f"E/spin={final_config.energy:7.4f}, "
              f"acceptance={sim.get_acceptance_rate():.3f}")


if __name__ == "__main__":
    # Run demonstrations
    simulator_3d = demonstrate_3d_simulation()
    compare_2d_vs_3d()
    
    print("\n=== Lattice Information ===")
    info = simulator_3d.get_lattice_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n=== Validation ===")
    validation = simulator_3d.validate_lattice_integrity()
    print(f"  Lattice valid: {validation['is_valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    
    print("\nExample completed successfully!")