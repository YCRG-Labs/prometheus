"""
Model Variant Registry for managing Ising model variant definitions.

This module provides a registry system for storing, retrieving, and managing
model variant configurations. It supports JSON-based persistence and validation
of model configurations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from .base_types import ModelVariantConfig
from .base_plugin import ModelVariantPlugin, BaseSimulator
from .plugin_registry import get_global_plugin_registry


class ModelVariantRegistry:
    """Registry for managing model variant definitions.
    
    The registry stores model variant configurations and provides methods for
    registration, retrieval, and listing of variants. Configurations are
    persisted to JSON files for reproducibility.
    
    Attributes:
        registry_path: Path to the JSON file storing variant configurations
        _variants: In-memory cache of registered variants
        _plugin_classes: Registered plugin classes for custom models
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the model variant registry.
        
        Args:
            registry_path: Path to JSON file for persistence. If None, uses
                          default path '.kiro/research/model_registry.json'
        """
        if registry_path is None:
            registry_path = '.kiro/research/model_registry.json'
        
        self.registry_path = Path(registry_path)
        self._variants: Dict[str, ModelVariantConfig] = {}
        self._plugin_classes: Dict[str, Type[ModelVariantPlugin]] = {}
        
        # Create directory if it doesn't exist
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry if available
        if self.registry_path.exists():
            self._load_registry()
    
    def register_variant(self, config: ModelVariantConfig) -> str:
        """Register a new model variant.
        
        Args:
            config: ModelVariantConfig object defining the variant
            
        Returns:
            variant_id: Unique identifier for the registered variant
            
        Raises:
            ValueError: If variant with same name already exists
        """
        # Validate configuration (triggers __post_init__ validation)
        self._validate_config(config)
        
        # Check for duplicate names
        if config.name in self._variants:
            raise ValueError(f"Variant '{config.name}' already registered")
        
        # Store variant
        variant_id = config.name
        self._variants[variant_id] = config
        
        # Persist to disk
        self._save_registry()
        
        return variant_id
    
    def get_variant(self, variant_id: str) -> ModelVariantConfig:
        """Retrieve variant configuration by ID.
        
        Args:
            variant_id: Unique identifier of the variant
            
        Returns:
            ModelVariantConfig object
            
        Raises:
            KeyError: If variant_id not found in registry
        """
        if variant_id not in self._variants:
            raise KeyError(f"Variant '{variant_id}' not found in registry")
        
        return self._variants[variant_id]
    
    def list_variants(self, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """List registered variants with optional filtering.
        
        Args:
            filters: Optional dictionary of filters to apply. Supported keys:
                    - 'dimensions': Filter by dimensionality (2 or 3)
                    - 'lattice_geometry': Filter by lattice type
                    - 'interaction_type': Filter by interaction type
                    - 'has_disorder': Filter variants with disorder (bool)
                    
        Returns:
            List of variant IDs matching the filters
        """
        variant_ids = list(self._variants.keys())
        
        if filters is None:
            return variant_ids
        
        # Apply filters
        filtered_ids = []
        for variant_id in variant_ids:
            config = self._variants[variant_id]
            
            # Check each filter
            if 'dimensions' in filters:
                if config.dimensions != filters['dimensions']:
                    continue
            
            if 'lattice_geometry' in filters:
                if config.lattice_geometry != filters['lattice_geometry']:
                    continue
            
            if 'interaction_type' in filters:
                if config.interaction_type != filters['interaction_type']:
                    continue
            
            if 'has_disorder' in filters:
                has_disorder = config.disorder_type is not None
                if has_disorder != filters['has_disorder']:
                    continue
            
            filtered_ids.append(variant_id)
        
        return filtered_ids
    
    def register_plugin(self, plugin_class: Type[ModelVariantPlugin],
                       variant_name: str) -> None:
        """Register a custom model plugin class.
        
        This method delegates to the global PluginRegistry for consistency.
        
        Args:
            plugin_class: Class implementing ModelVariantPlugin interface
            variant_name: Name to associate with this plugin
            
        Raises:
            ValueError: If plugin_class doesn't inherit from ModelVariantPlugin
        """
        # Delegate to global plugin registry
        plugin_registry = get_global_plugin_registry()
        plugin_registry.register_plugin(plugin_class, variant_name)
        
        # Also store locally for backwards compatibility
        self._plugin_classes[variant_name] = plugin_class
    
    def create_plugin_instance(self, variant_id: str) -> ModelVariantPlugin:
        """Create a plugin instance for the variant.
        
        Args:
            variant_id: ID of the variant
            
        Returns:
            ModelVariantPlugin instance
            
        Raises:
            KeyError: If variant_id not found
        """
        config = self.get_variant(variant_id)
        plugin_registry = get_global_plugin_registry()
        
        # Determine plugin name from interaction type
        plugin_name = self._get_plugin_name_for_variant(config)
        
        return plugin_registry.create_instance(plugin_name, config)
    
    def _get_plugin_name_for_variant(self, config: ModelVariantConfig) -> str:
        """Determine plugin name from variant configuration.
        
        Args:
            config: Model variant configuration
            
        Returns:
            Plugin name to use
        """
        # Map interaction types to plugin names
        type_to_plugin = {
            'nearest_neighbor': 'standard',
            'long_range': 'long_range',
            'frustrated': 'frustrated',
        }
        
        # Check if disorder is specified
        if config.disorder_type == 'quenched':
            return 'quenched_disorder'
        
        # Check interaction type
        if config.interaction_type in type_to_plugin:
            return type_to_plugin[config.interaction_type]
        
        # Default to standard
        return 'standard'
    
    def create_simulator(self, variant_id: str, lattice_size: int,
                        temperature: float, seed: Optional[int] = None,
                        **kwargs) -> BaseSimulator:
        """Create Monte Carlo simulator for the variant.
        
        Args:
            variant_id: ID of the variant to simulate
            lattice_size: Linear size of the lattice
            temperature: Simulation temperature
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to simulator
            
        Returns:
            BaseSimulator instance configured for the variant
            
        Raises:
            KeyError: If variant_id not found
            NotImplementedError: If simulator not available for variant type
        """
        config = self.get_variant(variant_id)
        
        # Import here to avoid circular dependencies
        from .simulators import create_simulator_for_variant
        
        return create_simulator_for_variant(
            config, lattice_size, temperature, seed, **kwargs
        )
    
    def _validate_config(self, config: ModelVariantConfig) -> None:
        """Validate model variant configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Basic validation is done in ModelVariantConfig.__post_init__
        # Additional validation can be added here
        
        # Validate interaction parameters based on interaction type
        if config.interaction_type == 'long_range':
            if 'alpha' not in config.interaction_params:
                raise ValueError(
                    "Long-range interactions require 'alpha' parameter"
                )
            alpha = config.interaction_params['alpha']
            if alpha <= 0:
                raise ValueError(f"Alpha must be positive, got {alpha}")
        
        # Validate disorder parameters
        if config.disorder_type is not None:
            if config.disorder_strength <= 0:
                raise ValueError(
                    "Disorder type specified but disorder_strength is zero"
                )
    
    def _save_registry(self) -> None:
        """Save registry to JSON file."""
        # Convert variants to serializable format
        data = {}
        for variant_id, config in self._variants.items():
            # Convert config to dict, excluding non-serializable fields
            config_dict = {
                'name': config.name,
                'dimensions': config.dimensions,
                'lattice_geometry': config.lattice_geometry,
                'interaction_type': config.interaction_type,
                'interaction_params': config.interaction_params,
                'disorder_type': config.disorder_type,
                'disorder_strength': config.disorder_strength,
                'external_field': config.external_field,
                'theoretical_tc': config.theoretical_tc,
                'theoretical_exponents': config.theoretical_exponents,
            }
            data[variant_id] = config_dict
        
        # Write to file
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_registry(self) -> None:
        """Load registry from JSON file."""
        with open(self.registry_path, 'r') as f:
            data = json.load(f)
        
        # Convert dict entries to ModelVariantConfig objects
        for variant_id, config_dict in data.items():
            config = ModelVariantConfig(**config_dict)
            self._variants[variant_id] = config
    
    def remove_variant(self, variant_id: str) -> None:
        """Remove a variant from the registry.
        
        Args:
            variant_id: ID of variant to remove
            
        Raises:
            KeyError: If variant_id not found
        """
        if variant_id not in self._variants:
            raise KeyError(f"Variant '{variant_id}' not found in registry")
        
        del self._variants[variant_id]
        self._save_registry()
    
    def clear_registry(self) -> None:
        """Clear all variants from the registry."""
        self._variants.clear()
        self._save_registry()
    
    def get_variant_info(self, variant_id: str) -> Dict[str, Any]:
        """Get detailed information about a variant.
        
        Args:
            variant_id: ID of the variant
            
        Returns:
            Dictionary with variant information
        """
        config = self.get_variant(variant_id)
        
        return {
            'name': config.name,
            'dimensions': config.dimensions,
            'lattice_geometry': config.lattice_geometry,
            'interaction_type': config.interaction_type,
            'interaction_params': config.interaction_params,
            'disorder_type': config.disorder_type,
            'disorder_strength': config.disorder_strength,
            'external_field': config.external_field,
            'has_theoretical_tc': config.theoretical_tc is not None,
            'has_theoretical_exponents': config.theoretical_exponents is not None,
        }

