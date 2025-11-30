"""
Plugin Registry for custom model variants.

This module provides a dedicated registry system for managing custom model
variant plugins, separate from the model configuration registry. It enables
researchers to register, validate, and instantiate custom model implementations.
"""

from typing import Dict, Type, Optional, Any
from .base_plugin import ModelVariantPlugin
from .base_types import ModelVariantConfig


class PluginRegistry:
    """Registry for custom model variant plugins.
    
    The PluginRegistry manages custom ModelVariantPlugin implementations,
    providing registration, validation, and factory methods for creating
    plugin instances. This enables researchers to extend the system with
    novel model types without modifying core code.
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_plugin(MyCustomModel, 'my_custom_model')
        >>> model = registry.create_instance('my_custom_model', config=config)
    
    Attributes:
        _plugins: Dictionary mapping plugin names to plugin classes
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, Type[ModelVariantPlugin]] = {}
        
        # Register built-in plugins
        self._register_builtin_plugins()
    
    def register_plugin(self, plugin_class: Type[ModelVariantPlugin],
                       plugin_name: str) -> None:
        """Register a custom model variant plugin.
        
        Args:
            plugin_class: Class implementing ModelVariantPlugin interface
            plugin_name: Unique name for the plugin
            
        Raises:
            ValueError: If plugin_class doesn't inherit from ModelVariantPlugin
                       or if plugin_name already registered
        """
        # Validate that plugin_class is a proper subclass
        if not issubclass(plugin_class, ModelVariantPlugin):
            raise ValueError(
                f"Plugin class must inherit from ModelVariantPlugin, "
                f"got {plugin_class.__name__}"
            )
        
        # Check for duplicate names
        if plugin_name in self._plugins:
            raise ValueError(
                f"Plugin '{plugin_name}' already registered. "
                f"Use a different name or unregister the existing plugin first."
            )
        
        # Validate plugin implementation
        self._validate_plugin_class(plugin_class)
        
        # Register the plugin
        self._plugins[plugin_name] = plugin_class
    
    def create_instance(self, plugin_name: str,
                       config: ModelVariantConfig,
                       **kwargs) -> ModelVariantPlugin:
        """Create an instance of a registered plugin.
        
        Args:
            plugin_name: Name of the registered plugin
            config: ModelVariantConfig for the plugin
            **kwargs: Additional arguments passed to plugin constructor
            
        Returns:
            Instance of the plugin class
            
        Raises:
            KeyError: If plugin_name not found in registry
        """
        if plugin_name not in self._plugins:
            raise KeyError(
                f"Plugin '{plugin_name}' not found in registry. "
                f"Available plugins: {list(self._plugins.keys())}"
            )
        
        plugin_class = self._plugins[plugin_name]
        
        # Create instance with config
        try:
            instance = plugin_class(config, **kwargs)
        except TypeError as e:
            # Try without kwargs if constructor doesn't accept them
            instance = plugin_class(config)
        
        # Validate the instance
        self._validate_plugin_instance(instance)
        
        return instance
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Raises:
            KeyError: If plugin_name not found
        """
        if plugin_name not in self._plugins:
            raise KeyError(f"Plugin '{plugin_name}' not found in registry")
        
        del self._plugins[plugin_name]
    
    def list_plugins(self) -> list[str]:
        """List all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def get_plugin_class(self, plugin_name: str) -> Type[ModelVariantPlugin]:
        """Get the plugin class for a given name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin class
            
        Raises:
            KeyError: If plugin_name not found
        """
        if plugin_name not in self._plugins:
            raise KeyError(f"Plugin '{plugin_name}' not found in registry")
        
        return self._plugins[plugin_name]
    
    def plugin_exists(self, plugin_name: str) -> bool:
        """Check if a plugin is registered.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            True if plugin is registered, False otherwise
        """
        return plugin_name in self._plugins
    
    def _validate_plugin_class(self, plugin_class: Type[ModelVariantPlugin]) -> None:
        """Validate that a plugin class properly implements the interface.
        
        Args:
            plugin_class: Plugin class to validate
            
        Raises:
            ValueError: If plugin class doesn't implement required methods
        """
        required_methods = [
            'compute_energy',
            'propose_spin_flip',
            'get_theoretical_properties'
        ]
        
        for method_name in required_methods:
            if not hasattr(plugin_class, method_name):
                raise ValueError(
                    f"Plugin class {plugin_class.__name__} must implement "
                    f"method '{method_name}'"
                )
            
            method = getattr(plugin_class, method_name)
            if not callable(method):
                raise ValueError(
                    f"Plugin class {plugin_class.__name__} attribute "
                    f"'{method_name}' must be callable"
                )
    
    def _validate_plugin_instance(self, instance: ModelVariantPlugin) -> None:
        """Validate a plugin instance through test operations.
        
        Args:
            instance: Plugin instance to validate
            
        Raises:
            ValueError: If plugin instance fails validation
        """
        import numpy as np
        
        # Test with a small configuration
        test_config = np.ones((4, 4)) if hasattr(instance, 'dimensions') and instance.dimensions == 2 else np.ones((4, 4, 4))
        
        try:
            # Test energy computation
            energy = instance.compute_energy(test_config)
            if not isinstance(energy, (int, float, np.number)):
                raise ValueError(
                    f"compute_energy must return a number, got {type(energy)}"
                )
            
            # Test spin flip proposal
            site = (0, 0) if test_config.ndim == 2 else (0, 0, 0)
            proposal = instance.propose_spin_flip(test_config, site)
            
            if not hasattr(proposal, 'site'):
                raise ValueError("propose_spin_flip must return object with 'site' attribute")
            if not hasattr(proposal, 'energy_change'):
                raise ValueError("propose_spin_flip must return object with 'energy_change' attribute")
            
            # Test theoretical properties
            props = instance.get_theoretical_properties()
            if not isinstance(props, dict):
                raise ValueError(
                    f"get_theoretical_properties must return dict, got {type(props)}"
                )
            
        except Exception as e:
            raise ValueError(
                f"Plugin validation failed: {str(e)}"
            )
    
    def _register_builtin_plugins(self) -> None:
        """Register built-in model variant plugins."""
        from .model_plugins import (
            StandardIsingModel,
            LongRangeIsingModel,
            QuenchedDisorderModel,
            FrustratedGeometryModel
        )
        
        # Register built-in plugins
        self._plugins['standard'] = StandardIsingModel
        self._plugins['long_range'] = LongRangeIsingModel
        self._plugins['quenched_disorder'] = QuenchedDisorderModel
        self._plugins['frustrated'] = FrustratedGeometryModel
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get information about a registered plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Dictionary with plugin information
            
        Raises:
            KeyError: If plugin_name not found
        """
        if plugin_name not in self._plugins:
            raise KeyError(f"Plugin '{plugin_name}' not found in registry")
        
        plugin_class = self._plugins[plugin_name]
        
        return {
            'name': plugin_name,
            'class_name': plugin_class.__name__,
            'module': plugin_class.__module__,
            'docstring': plugin_class.__doc__,
        }


# Global plugin registry instance
_global_plugin_registry = None


def get_global_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance.
    
    Returns:
        Global PluginRegistry instance
    """
    global _global_plugin_registry
    if _global_plugin_registry is None:
        _global_plugin_registry = PluginRegistry()
    return _global_plugin_registry
