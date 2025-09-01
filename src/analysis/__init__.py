"""
Analysis module for the Prometheus project.

This module provides tools for analyzing trained VAE models and discovering
physical insights from latent space representations.
"""

from .latent_analysis import LatentAnalyzer, LatentRepresentation
from .order_parameter_discovery import OrderParameterAnalyzer, CorrelationResult, OrderParameterCandidate

__all__ = [
    'LatentAnalyzer',
    'LatentRepresentation', 
    'OrderParameterAnalyzer',
    'CorrelationResult',
    'OrderParameterCandidate'
]