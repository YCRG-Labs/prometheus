"""
Example: Creating and Using Custom Model Plugins

This example demonstrates how to create a custom Ising model variant plugin
and integrate it with the research explorer system.
"""

import numpy as np
import sys
sys.path.append('.')

from src.research.base_types import ModelVariantConfig
from src.research.base_plugin import ModelVariantPlugin, SpinFlipProposal
from src.research.plugin_registry import get_global_plugin_registry
from src.research.model_registry import ModelVariantRegistry
from src.research.discovery_pipeline import DiscoveryPipeline, DiscoveryConfig
from src.research.parameter_explorer import ExplorationStrategy


# Example 1: Simple Custom Model - Ising with Next-Nearest-Neighbor Interactions
class NextNearestNeighborIsing(ModelVariantPlugin):
    """Ising model with both nearest and next-nearest neighbor interactions.
    
    Energy: E = -J1 Σ_<i,j> s_i s_j - J2 Σ_<<i,j>> s_i s_j - h Σ_i s_i
    where <i,j> are nearest neighbors and <<i,j>> are next-nearest neighbors.
    """
    
    def __init__(self, config: ModelVariantConfig):
        self.config = config
        self.dimensions = config.dimensions
        self.J1 = config.interaction_params.get('J1', 1.0)
        self.J2 = config.interaction_params.get('J2', 0.5)
        self.h = config.external_field
        
        if self.dimensions != 2:
            raise NotImplementedError("Only 2D implemented for this example")
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute energy with nearest and next-nearest neighbor interactions."""
        L = configuration.shape[0]
        energy = 0.0
        
        # Nearest-neighbor interactions (J1)
        energy -= self.J1 * np.sum(configuration[:, :-1] * configuration[:, 1:])
        energy -= self.J1 * np.sum(configuration[:-1, :] * configuration[1:, :])
        
        # Next-nearest-neighbor interactions (J2) - diagonal
        energy -= self.J2 * np.sum(configuration[:-1, :-1] * configuration[1:, 1:])
        energy -= self.J2 * np.sum(configuration[:-1, 1:] * configuration[1:, :-1])
        
        # External field
        energy -= self.h * np.sum(configuration)
        
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: tuple) -> SpinFlipProposal:
        """Propose spin flip with NNN interactions."""
        i, j = site
        L = configuration.shape[0]
        spin = configuration[i, j]
        
        # Nearest neighbors (4 sites)
        nn_sum = (configuration[(i+1) % L, j] + configuration[(i-1) % L, j] +
                  configuration[i, (j+1) % L] + configuration[i, (j-1) % L])
        
        # Next-nearest neighbors (4 diagonal sites)
        nnn_sum = (configuration[(i+1) % L, (j+1) % L] + 
                   configuration[(i+1) % L, (j-1) % L] +
                   configuration[(i-1) % L, (j+1) % L] + 
                   configuration[(i-1) % L, (j-1) % L])
        
        # Total field
        field = self.J1 * nn_sum + self.J2 * nnn_sum
        
        delta_e = 2 * spin * (field + self.h)
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self):
        """Return theoretical properties."""
        # Properties depend on J2/J1 ratio
        if abs(self.J2) < 0.1:
            # Approximately standard 2D Ising
            return {
                'tc': 2.269,
                'exponents': {'beta': 0.125, 'nu': 1.0},
                'universality_class': '2D_Ising'
            }
        else:
            # Modified by NNN interactions
            return {
                'tc': None,  # Depends on J2/J1
                'exponents': None,
                'universality_class': 'NNN_Ising'
            }


# Example 2: Dipolar Interactions (simplified)
class DipolarIsingModel(ModelVariantPlugin):
    """Ising model with dipolar interactions (simplified).
    
    Energy: E = -Σ_{i<j} J_ij s_i s_j where J_ij ~ (1 - 3cos²θ_ij) / r_ij³
    This is a simplified version for demonstration.
    """
    
    def __init__(self, config: ModelVariantConfig):
        self.config = config
        self.dimensions = config.dimensions
        self.J0 = config.interaction_params.get('J0', 1.0)
        self.h = config.external_field
        
        # Precomputation cache
        self._couplings = None
        self._lattice_size = None
    
    def _precompute_couplings(self, L: int) -> None:
        """Precompute dipolar coupling matrix."""
        if self._lattice_size == L:
            return
        
        self._lattice_size = L
        n_sites = L ** 2
        self._couplings = np.zeros((n_sites, n_sites))
        
        for i in range(n_sites):
            ix, iy = i // L, i % L
            for j in range(i+1, n_sites):
                jx, jy = j // L, j % L
                
                # Minimum image convention
                dx = min(abs(ix - jx), L - abs(ix - jx))
                dy = min(abs(iy - jy), L - abs(iy - jy))
                r = np.sqrt(dx**2 + dy**2)
                
                if r > 0:
                    # Simplified dipolar: J ~ 1/r³ (ignoring angular dependence)
                    coupling = self.J0 / (r ** 3)
                    self._couplings[i, j] = coupling
                    self._couplings[j, i] = coupling
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute energy with dipolar interactions."""
        L = configuration.shape[0]
        self._precompute_couplings(L)
        
        spins = configuration.flatten()
        energy = 0.0
        
        for i in range(len(spins)):
            for j in range(i+1, len(spins)):
                energy -= self._couplings[i, j] * spins[i] * spins[j]
        
        energy -= self.h * np.sum(spins)
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: tuple) -> SpinFlipProposal:
        """Propose spin flip with dipolar interactions."""
        L = configuration.shape[0]
        self._precompute_couplings(L)
        
        i, j = site
        site_idx = i * L + j
        spins = configuration.flatten()
        spin = spins[site_idx]
        
        # Compute field from all other spins
        field = 0.0
        for k in range(len(spins)):
            if k != site_idx:
                field += self._couplings[site_idx, k] * spins[k]
        
        delta_e = 2 * spin * (field + self.h)
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self):
        return {
            'tc': None,
            'exponents': None,
            'universality_class': 'dipolar'
        }


def example_1_register_and_test_plugin():
    """Example 1: Register a custom plugin and test it."""
    print("=" * 70)
    print("Example 1: Register and Test Custom Plugin")
    print("=" * 70)
    
    # Get the global plugin registry
    plugin_registry = get_global_plugin_registry()
    
    # Register our custom plugin
    plugin_registry.register_plugin(NextNearestNeighborIsing, 'nnn_ising')
    print(f"✓ Registered plugin: nnn_ising")
    
    # List all available plugins
    print(f"\nAvailable plugins: {plugin_registry.list_plugins()}")
    
    # Create a configuration for the custom model
    config = ModelVariantConfig(
        name='nnn_ising_test',
        dimensions=2,
        lattice_geometry='square',
        interaction_type='nnn',
        interaction_params={'J1': 1.0, 'J2': 0.3},
        external_field=0.0
    )
    
    # Create an instance of the plugin
    model = plugin_registry.create_instance('nnn_ising', config)
    print(f"✓ Created model instance: {model.__class__.__name__}")
    
    # Test the model with a small configuration
    test_config = np.random.choice([-1, 1], size=(8, 8))
    energy = model.compute_energy(test_config)
    print(f"✓ Computed energy for 8x8 lattice: {energy:.2f}")
    
    # Test spin flip proposal
    proposal = model.propose_spin_flip(test_config, (4, 4))
    print(f"✓ Spin flip at (4,4): ΔE = {proposal.energy_change:.2f}")
    
    # Get theoretical properties
    props = model.get_theoretical_properties()
    print(f"✓ Theoretical properties: {props}")
    
    print("\n✓ Plugin validation successful!\n")


def example_2_use_plugin_in_discovery():
    """Example 2: Use custom plugin in discovery pipeline."""
    print("=" * 70)
    print("Example 2: Use Custom Plugin in Discovery Pipeline")
    print("=" * 70)
    
    # Register the plugin
    plugin_registry = get_global_plugin_registry()
    if not plugin_registry.plugin_exists('nnn_ising'):
        plugin_registry.register_plugin(NextNearestNeighborIsing, 'nnn_ising')
    
    # Create model variant registry
    model_registry = ModelVariantRegistry()
    
    # Register the variant configuration
    config = ModelVariantConfig(
        name='nnn_ising_j2_0.3',
        dimensions=2,
        lattice_geometry='square',
        interaction_type='nnn',
        interaction_params={'J1': 1.0, 'J2': 0.3},
        external_field=0.0
    )
    
    variant_id = model_registry.register_variant(config)
    print(f"✓ Registered variant: {variant_id}")
    
    # Note: To fully integrate with discovery pipeline, you would need to:
    # 1. Create a simulator for the custom model
    # 2. Update the simulator factory to recognize the plugin
    # 3. Run the discovery pipeline
    
    print("\n✓ Custom plugin ready for discovery pipeline!")
    print("  (Full integration requires simulator implementation)")
    print()


def example_3_dipolar_model():
    """Example 3: Register and test dipolar model."""
    print("=" * 70)
    print("Example 3: Dipolar Ising Model")
    print("=" * 70)
    
    plugin_registry = get_global_plugin_registry()
    
    # Register dipolar model
    plugin_registry.register_plugin(DipolarIsingModel, 'dipolar')
    print(f"✓ Registered dipolar model")
    
    # Create configuration
    config = ModelVariantConfig(
        name='dipolar_test',
        dimensions=2,
        lattice_geometry='square',
        interaction_type='dipolar',
        interaction_params={'J0': 1.0},
        external_field=0.0
    )
    
    # Create instance
    model = plugin_registry.create_instance('dipolar', config)
    print(f"✓ Created dipolar model instance")
    
    # Test on small lattice (dipolar is computationally expensive)
    test_config = np.random.choice([-1, 1], size=(6, 6))
    energy = model.compute_energy(test_config)
    print(f"✓ Computed energy for 6x6 lattice: {energy:.2f}")
    
    print("\n✓ Dipolar model working!\n")


def example_4_plugin_info():
    """Example 4: Get information about registered plugins."""
    print("=" * 70)
    print("Example 4: Plugin Information")
    print("=" * 70)
    
    plugin_registry = get_global_plugin_registry()
    
    # Ensure our custom plugins are registered
    if not plugin_registry.plugin_exists('nnn_ising'):
        plugin_registry.register_plugin(NextNearestNeighborIsing, 'nnn_ising')
    if not plugin_registry.plugin_exists('dipolar'):
        plugin_registry.register_plugin(DipolarIsingModel, 'dipolar')
    
    # List all plugins
    print("Registered Plugins:")
    print("-" * 70)
    for plugin_name in plugin_registry.list_plugins():
        info = plugin_registry.get_plugin_info(plugin_name)
        print(f"\nPlugin: {info['name']}")
        print(f"  Class: {info['class_name']}")
        print(f"  Module: {info['module']}")
        if info['docstring']:
            # Print first line of docstring
            first_line = info['docstring'].strip().split('\n')[0]
            print(f"  Description: {first_line}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CUSTOM PLUGIN EXAMPLES")
    print("=" * 70 + "\n")
    
    # Run all examples
    example_1_register_and_test_plugin()
    example_2_use_plugin_in_discovery()
    example_3_dipolar_model()
    example_4_plugin_info()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Copy src/research/plugin_template.py to create your own model")
    print("2. Implement the three required methods")
    print("3. Register your plugin with the PluginRegistry")
    print("4. Use it in the discovery pipeline")
    print("\nSee docs/PLUGIN_DEVELOPMENT_GUIDE.md for detailed instructions.")
    print("=" * 70 + "\n")
