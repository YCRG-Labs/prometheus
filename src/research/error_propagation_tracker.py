"""
Error Propagation Chain Tracker for Discovery Pipeline.

This module tracks how uncertainty propagates through the validation chain:
VAE → Bootstrap → Effect Size → Anomaly Threshold

This provides insight into which stages contribute most to final uncertainty
and helps identify dominant error sources for targeted improvements.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..utils.logging_utils import get_logger


class PropagationStage(Enum):
    """Stages in the error propagation chain."""
    VAE_MEASUREMENT = "vae_measurement"
    BOOTSTRAP_RESAMPLING = "bootstrap_resampling"
    EFFECT_SIZE_CALCULATION = "effect_size"
    ANOMALY_THRESHOLD = "anomaly_threshold"
    FINAL_CLASSIFICATION = "final_classification"


@dataclass
class StageUncertainty:
    """Uncertainty at a specific propagation stage.
    
    Attributes:
        stage: The propagation stage
        value: Central value at this stage
        uncertainty: Absolute uncertainty (standard error)
        relative_uncertainty: Relative uncertainty (uncertainty/value)
        sources: Contributing uncertainty sources
        metadata: Additional stage-specific information
    """
    stage: PropagationStage
    value: float
    uncertainty: float
    relative_uncertainty: float
    sources: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PropagationChain:
    """Complete error propagation chain for a single measurement.
    
    Attributes:
        measurement_id: Identifier for this measurement
        stages: List of uncertainties at each stage
        total_uncertainty: Final combined uncertainty
        dominant_source: Stage contributing most to uncertainty
        amplification_factor: Ratio of final to initial uncertainty
    """
    measurement_id: str
    stages: List[StageUncertainty]
    total_uncertainty: float
    dominant_source: PropagationStage
    amplification_factor: float


class ErrorPropagationTracker:
    """Track uncertainty propagation through validation chain.
    
    This class monitors how measurement uncertainty from VAE analysis
    propagates through bootstrap resampling, effect size calculation,
    and anomaly threshold comparison to produce final classifications.
    
    Key capabilities:
    - Track uncertainty at each pipeline stage
    - Identify dominant error sources
    - Visualize error propagation
    - Compare propagation across measurements
    - Suggest targeted improvements
    
    Attributes:
        logger: Logger instance
        chains: Stored propagation chains
        anomaly_threshold: Threshold for anomaly detection (in sigma)
    """
    
    def __init__(self, anomaly_threshold: float = 3.0):
        """Initialize error propagation tracker.
        
        Args:
            anomaly_threshold: Threshold for anomaly detection in sigma
        """
        self.logger = get_logger(__name__)
        self.chains: List[PropagationChain] = []
        self.anomaly_threshold = anomaly_threshold
        self.logger.info(
            f"Initialized ErrorPropagationTracker with "
            f"anomaly_threshold={anomaly_threshold}σ"
        )
    
    def track_vae_measurement(
        self,
        measurement_id: str,
        exponent_value: float,
        r_squared: float,
        n_data_points: int,
        fit_residuals: Optional[np.ndarray] = None
    ) -> StageUncertainty:
        """Track uncertainty from VAE measurement stage.
        
        VAE measurement uncertainty comes from:
        - Fit quality (R²)
        - Number of data points
        - Residual variance
        
        Args:
            measurement_id: Identifier for this measurement
            exponent_value: Measured exponent value
            r_squared: R² value from fit
            n_data_points: Number of data points in fit
            fit_residuals: Optional array of fit residuals
            
        Returns:
            StageUncertainty for VAE measurement
        """
        # Estimate uncertainty from R² and sample size
        # Standard error ≈ sqrt((1-R²) / (n-2))
        if n_data_points > 2:
            base_uncertainty = np.sqrt((1 - r_squared) / (n_data_points - 2))
        else:
            base_uncertainty = 1.0  # Large uncertainty for insufficient data
        
        # Scale by exponent value to get absolute uncertainty
        absolute_uncertainty = abs(exponent_value) * base_uncertainty
        
        # Decompose uncertainty sources
        sources = {
            'fit_quality': (1 - r_squared) * absolute_uncertainty,
            'sample_size': (1 / np.sqrt(n_data_points)) * absolute_uncertainty
        }
        
        # Add residual contribution if available
        if fit_residuals is not None:
            residual_std = np.std(fit_residuals)
            sources['residual_variance'] = residual_std
            absolute_uncertainty = np.sqrt(
                absolute_uncertainty**2 + residual_std**2
            )
        
        relative_uncertainty = absolute_uncertainty / abs(exponent_value) if exponent_value != 0 else float('inf')
        
        stage = StageUncertainty(
            stage=PropagationStage.VAE_MEASUREMENT,
            value=exponent_value,
            uncertainty=absolute_uncertainty,
            relative_uncertainty=relative_uncertainty,
            sources=sources,
            metadata={
                'r_squared': r_squared,
                'n_data_points': n_data_points,
                'measurement_id': measurement_id
            }
        )
        
        self.logger.debug(
            f"VAE measurement: value={exponent_value:.4f}, "
            f"uncertainty={absolute_uncertainty:.4f}, "
            f"R²={r_squared:.3f}"
        )
        
        return stage
    
    def track_bootstrap_resampling(
        self,
        previous_stage: StageUncertainty,
        bootstrap_samples: np.ndarray,
        n_bootstrap: int = 1000
    ) -> StageUncertainty:
        """Track uncertainty from bootstrap resampling stage.
        
        Bootstrap resampling adds uncertainty from:
        - Sampling variability
        - Distribution shape (skewness, kurtosis)
        - Finite bootstrap samples
        
        Args:
            previous_stage: Uncertainty from previous stage
            bootstrap_samples: Array of bootstrap resampled values
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            StageUncertainty for bootstrap resampling
        """
        # Calculate bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_samples)
        bootstrap_std = np.std(bootstrap_samples, ddof=1)
        bootstrap_skew = stats.skew(bootstrap_samples) if len(bootstrap_samples) > 2 else 0.0
        
        # Bootstrap adds sampling uncertainty
        # Total uncertainty = sqrt(measurement² + bootstrap²)
        measurement_uncertainty = previous_stage.uncertainty
        sampling_uncertainty = bootstrap_std / np.sqrt(n_bootstrap)
        
        total_uncertainty = np.sqrt(
            measurement_uncertainty**2 + 
            bootstrap_std**2 +
            sampling_uncertainty**2
        )
        
        # Decompose sources
        sources = {
            'measurement_error': measurement_uncertainty,
            'bootstrap_variance': bootstrap_std,
            'sampling_error': sampling_uncertainty,
            'distribution_skewness': abs(bootstrap_skew) * bootstrap_std
        }
        
        relative_uncertainty = total_uncertainty / abs(bootstrap_mean) if bootstrap_mean != 0 else float('inf')
        
        stage = StageUncertainty(
            stage=PropagationStage.BOOTSTRAP_RESAMPLING,
            value=bootstrap_mean,
            uncertainty=total_uncertainty,
            relative_uncertainty=relative_uncertainty,
            sources=sources,
            metadata={
                'n_bootstrap': n_bootstrap,
                'bootstrap_std': bootstrap_std,
                'skewness': bootstrap_skew
            }
        )
        
        self.logger.debug(
            f"Bootstrap resampling: value={bootstrap_mean:.4f}, "
            f"uncertainty={total_uncertainty:.4f}, "
            f"bootstrap_std={bootstrap_std:.4f}"
        )
        
        return stage
    
    def track_effect_size_calculation(
        self,
        previous_stage: StageUncertainty,
        predicted_value: float,
        predicted_uncertainty: float = 0.0
    ) -> StageUncertainty:
        """Track uncertainty from effect size calculation stage.
        
        Effect size (Cohen's d) uncertainty comes from:
        - Measured value uncertainty
        - Predicted value uncertainty (if known)
        - Denominator (standard error) uncertainty
        
        Args:
            previous_stage: Uncertainty from previous stage
            predicted_value: Predicted/theoretical value for comparison
            predicted_uncertainty: Uncertainty in predicted value
            
        Returns:
            StageUncertainty for effect size calculation
        """
        measured_value = previous_stage.value
        measured_uncertainty = previous_stage.uncertainty
        
        # Cohen's d = (measured - predicted) / SE
        effect_size = (measured_value - predicted_value) / measured_uncertainty
        
        # Propagate uncertainty through Cohen's d calculation
        # δd/δmeasured = 1/SE
        # δd/δSE = -(measured - predicted)/SE²
        
        # Uncertainty in numerator
        numerator_uncertainty = np.sqrt(
            measured_uncertainty**2 + predicted_uncertainty**2
        )
        
        # Uncertainty in denominator (SE itself has uncertainty)
        # Approximate SE uncertainty as SE/sqrt(2n) for bootstrap
        n_bootstrap = previous_stage.metadata.get('n_bootstrap', 1000)
        denominator_uncertainty = measured_uncertainty / np.sqrt(2 * n_bootstrap)
        
        # Total effect size uncertainty using error propagation
        effect_size_uncertainty = np.sqrt(
            (numerator_uncertainty / measured_uncertainty)**2 +
            (effect_size * denominator_uncertainty / measured_uncertainty)**2
        )
        
        # Decompose sources
        sources = {
            'measured_uncertainty': (measured_uncertainty / measured_uncertainty) * effect_size_uncertainty,
            'predicted_uncertainty': (predicted_uncertainty / measured_uncertainty) * effect_size_uncertainty if predicted_uncertainty > 0 else 0.0,
            'denominator_uncertainty': (denominator_uncertainty / measured_uncertainty) * effect_size_uncertainty
        }
        
        relative_uncertainty = effect_size_uncertainty / abs(effect_size) if effect_size != 0 else float('inf')
        
        stage = StageUncertainty(
            stage=PropagationStage.EFFECT_SIZE_CALCULATION,
            value=effect_size,
            uncertainty=effect_size_uncertainty,
            relative_uncertainty=relative_uncertainty,
            sources=sources,
            metadata={
                'predicted_value': predicted_value,
                'predicted_uncertainty': predicted_uncertainty,
                'measured_value': measured_value
            }
        )
        
        self.logger.debug(
            f"Effect size: d={effect_size:.4f}, "
            f"uncertainty={effect_size_uncertainty:.4f}"
        )
        
        return stage
    
    def track_anomaly_threshold(
        self,
        previous_stage: StageUncertainty,
        threshold_sigma: Optional[float] = None
    ) -> StageUncertainty:
        """Track uncertainty at anomaly threshold comparison stage.
        
        Anomaly detection compares effect size to threshold:
        - Effect size > threshold → anomaly
        - Uncertainty affects classification confidence
        
        Args:
            previous_stage: Uncertainty from previous stage
            threshold_sigma: Anomaly threshold in sigma (default: self.anomaly_threshold)
            
        Returns:
            StageUncertainty for anomaly threshold
        """
        if threshold_sigma is None:
            threshold_sigma = self.anomaly_threshold
        
        effect_size = previous_stage.value
        effect_size_uncertainty = previous_stage.uncertainty
        
        # Distance from threshold in units of uncertainty
        distance_from_threshold = abs(effect_size) - threshold_sigma
        significance = distance_from_threshold / effect_size_uncertainty
        
        # Classification confidence based on how far from threshold
        # High confidence if |effect_size| >> threshold or << threshold
        # Low confidence if |effect_size| ≈ threshold
        if significance > 2:
            # Clearly above threshold
            confidence = 0.95
            classification = "anomaly"
        elif significance < -2:
            # Clearly below threshold
            confidence = 0.95
            classification = "normal"
        else:
            # Near threshold - uncertain
            confidence = 0.5 + 0.225 * significance  # Linear interpolation
            classification = "uncertain"
        
        # Uncertainty in classification
        classification_uncertainty = effect_size_uncertainty / threshold_sigma
        
        sources = {
            'effect_size_uncertainty': effect_size_uncertainty,
            'threshold_proximity': abs(distance_from_threshold),
            'significance': abs(significance)
        }
        
        stage = StageUncertainty(
            stage=PropagationStage.ANOMALY_THRESHOLD,
            value=abs(effect_size),
            uncertainty=classification_uncertainty,
            relative_uncertainty=classification_uncertainty / threshold_sigma,
            sources=sources,
            metadata={
                'threshold': threshold_sigma,
                'distance_from_threshold': distance_from_threshold,
                'significance': significance,
                'confidence': confidence,
                'classification': classification
            }
        )
        
        self.logger.debug(
            f"Anomaly threshold: |d|={abs(effect_size):.4f}, "
            f"threshold={threshold_sigma:.1f}σ, "
            f"classification={classification} (confidence={confidence:.2f})"
        )
        
        return stage

    
    def create_propagation_chain(
        self,
        measurement_id: str,
        vae_params: Dict[str, Any],
        bootstrap_samples: np.ndarray,
        predicted_value: float,
        predicted_uncertainty: float = 0.0,
        threshold_sigma: Optional[float] = None
    ) -> PropagationChain:
        """Create complete error propagation chain for a measurement.
        
        Args:
            measurement_id: Identifier for this measurement
            vae_params: VAE measurement parameters (exponent_value, r_squared, n_data_points)
            bootstrap_samples: Bootstrap resampled values
            predicted_value: Predicted/theoretical value
            predicted_uncertainty: Uncertainty in prediction
            threshold_sigma: Anomaly threshold in sigma
            
        Returns:
            Complete PropagationChain object
        """
        stages = []
        
        # Stage 1: VAE Measurement
        vae_stage = self.track_vae_measurement(
            measurement_id=measurement_id,
            exponent_value=vae_params['exponent_value'],
            r_squared=vae_params['r_squared'],
            n_data_points=vae_params['n_data_points'],
            fit_residuals=vae_params.get('fit_residuals')
        )
        stages.append(vae_stage)
        
        # Stage 2: Bootstrap Resampling
        bootstrap_stage = self.track_bootstrap_resampling(
            previous_stage=vae_stage,
            bootstrap_samples=bootstrap_samples,
            n_bootstrap=len(bootstrap_samples)
        )
        stages.append(bootstrap_stage)
        
        # Stage 3: Effect Size Calculation
        effect_stage = self.track_effect_size_calculation(
            previous_stage=bootstrap_stage,
            predicted_value=predicted_value,
            predicted_uncertainty=predicted_uncertainty
        )
        stages.append(effect_stage)
        
        # Stage 4: Anomaly Threshold
        threshold_stage = self.track_anomaly_threshold(
            previous_stage=effect_stage,
            threshold_sigma=threshold_sigma
        )
        stages.append(threshold_stage)
        
        # Calculate total uncertainty and dominant source
        total_uncertainty = threshold_stage.uncertainty
        
        # Find dominant source by comparing relative contributions
        max_contribution = 0.0
        dominant_source = PropagationStage.VAE_MEASUREMENT
        
        for stage in stages:
            contribution = stage.uncertainty / total_uncertainty if total_uncertainty > 0 else 0.0
            if contribution > max_contribution:
                max_contribution = contribution
                dominant_source = stage.stage
        
        # Calculate amplification factor
        initial_uncertainty = vae_stage.relative_uncertainty
        final_uncertainty = threshold_stage.relative_uncertainty
        amplification_factor = final_uncertainty / initial_uncertainty if initial_uncertainty > 0 else 1.0
        
        chain = PropagationChain(
            measurement_id=measurement_id,
            stages=stages,
            total_uncertainty=total_uncertainty,
            dominant_source=dominant_source,
            amplification_factor=amplification_factor
        )
        
        self.chains.append(chain)
        
        self.logger.info(
            f"Created propagation chain for {measurement_id}: "
            f"dominant_source={dominant_source.value}, "
            f"amplification={amplification_factor:.2f}x"
        )
        
        return chain
    
    def identify_dominant_error_sources(
        self,
        chains: Optional[List[PropagationChain]] = None
    ) -> Dict[PropagationStage, float]:
        """Identify dominant error sources across multiple chains.
        
        Args:
            chains: List of propagation chains to analyze (default: all stored chains)
            
        Returns:
            Dictionary mapping stages to their average contribution to total uncertainty
        """
        if chains is None:
            chains = self.chains
        
        if not chains:
            self.logger.warning("No propagation chains available for analysis")
            return {}
        
        # Count how often each stage is dominant
        stage_counts = {stage: 0 for stage in PropagationStage}
        stage_contributions = {stage: [] for stage in PropagationStage}
        
        for chain in chains:
            stage_counts[chain.dominant_source] += 1
            
            # Calculate each stage's contribution to total uncertainty
            total_unc = chain.total_uncertainty
            for stage in chain.stages:
                contribution = stage.uncertainty / total_unc if total_unc > 0 else 0.0
                stage_contributions[stage.stage].append(contribution)
        
        # Calculate average contributions
        avg_contributions = {}
        for stage, contributions in stage_contributions.items():
            if contributions:
                avg_contributions[stage] = np.mean(contributions)
            else:
                avg_contributions[stage] = 0.0
        
        # Sort by contribution
        sorted_contributions = dict(
            sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)
        )
        
        self.logger.info("Dominant error sources identified:")
        for stage, contribution in sorted_contributions.items():
            count = stage_counts[stage]
            self.logger.info(
                f"  {stage.value}: {contribution:.1%} average contribution, "
                f"dominant in {count}/{len(chains)} chains"
            )
        
        return sorted_contributions
    
    def visualize_propagation_chain(
        self,
        chain: PropagationChain,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """Visualize error propagation through a single chain.
        
        Creates a figure showing:
        - Uncertainty at each stage
        - Relative contributions
        - Cumulative uncertainty growth
        
        Args:
            chain: PropagationChain to visualize
            save_path: Optional path to save figure
            show: Whether to display figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Error Propagation Chain: {chain.measurement_id}\n'
            f'Dominant Source: {chain.dominant_source.value} | '
            f'Amplification: {chain.amplification_factor:.2f}x',
            fontsize=14, fontweight='bold'
        )
        
        # Extract data
        stage_names = [s.stage.value for s in chain.stages]
        uncertainties = [s.uncertainty for s in chain.stages]
        relative_uncertainties = [s.relative_uncertainty for s in chain.stages]
        values = [s.value for s in chain.stages]
        
        # Plot 1: Absolute uncertainty at each stage
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(stage_names)), uncertainties, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(stage_names)))
        ax1.set_xticklabels(stage_names, rotation=45, ha='right')
        ax1.set_ylabel('Absolute Uncertainty', fontsize=11)
        ax1.set_title('Uncertainty at Each Stage', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight dominant source
        dominant_idx = [i for i, s in enumerate(chain.stages) if s.stage == chain.dominant_source][0]
        bars1[dominant_idx].set_color('crimson')
        bars1[dominant_idx].set_alpha(0.8)
        
        # Plot 2: Relative uncertainty (%)
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(stage_names)), 
                        [r * 100 for r in relative_uncertainties], 
                        color='darkorange', alpha=0.7)
        ax2.set_xticks(range(len(stage_names)))
        ax2.set_xticklabels(stage_names, rotation=45, ha='right')
        ax2.set_ylabel('Relative Uncertainty (%)', fontsize=11)
        ax2.set_title('Relative Uncertainty Growth', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        bars2[dominant_idx].set_color('crimson')
        bars2[dominant_idx].set_alpha(0.8)
        
        # Plot 3: Cumulative uncertainty
        ax3 = axes[1, 0]
        cumulative = np.cumsum(uncertainties)
        ax3.plot(range(len(stage_names)), cumulative, 'o-', 
                linewidth=2, markersize=8, color='forestgreen')
        ax3.fill_between(range(len(stage_names)), 0, cumulative, alpha=0.2, color='forestgreen')
        ax3.set_xticks(range(len(stage_names)))
        ax3.set_xticklabels(stage_names, rotation=45, ha='right')
        ax3.set_ylabel('Cumulative Uncertainty', fontsize=11)
        ax3.set_title('Uncertainty Accumulation', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Source contributions for each stage
        ax4 = axes[1, 1]
        
        # Get sources from the stage with most sources
        max_sources_stage = max(chain.stages, key=lambda s: len(s.sources))
        all_source_names = list(max_sources_stage.sources.keys())
        
        # Create stacked bar chart
        bottom = np.zeros(len(stage_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_source_names)))
        
        for i, source_name in enumerate(all_source_names):
            source_values = []
            for stage in chain.stages:
                source_values.append(stage.sources.get(source_name, 0.0))
            
            ax4.bar(range(len(stage_names)), source_values, bottom=bottom,
                   label=source_name, color=colors[i], alpha=0.8)
            bottom += source_values
        
        ax4.set_xticks(range(len(stage_names)))
        ax4.set_xticklabels(stage_names, rotation=45, ha='right')
        ax4.set_ylabel('Uncertainty Contribution', fontsize=11)
        ax4.set_title('Uncertainty Source Breakdown', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved propagation chain visualization to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def visualize_multiple_chains(
        self,
        chains: Optional[List[PropagationChain]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """Visualize error propagation across multiple chains.
        
        Creates comparison plots showing:
        - Average uncertainty at each stage
        - Distribution of dominant sources
        - Amplification factor distribution
        
        Args:
            chains: List of chains to visualize (default: all stored chains)
            save_path: Optional path to save figure
            show: Whether to display figure
            
        Returns:
            Matplotlib figure object
        """
        if chains is None:
            chains = self.chains
        
        if not chains:
            self.logger.warning("No propagation chains available for visualization")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Error Propagation Analysis: {len(chains)} Measurements',
            fontsize=14, fontweight='bold'
        )
        
        # Collect data across all chains
        stage_uncertainties = {stage: [] for stage in PropagationStage}
        dominant_sources = []
        amplification_factors = []
        
        for chain in chains:
            dominant_sources.append(chain.dominant_source.value)
            amplification_factors.append(chain.amplification_factor)
            
            for stage in chain.stages:
                stage_uncertainties[stage.stage].append(stage.uncertainty)
        
        # Plot 1: Average uncertainty at each stage
        ax1 = axes[0, 0]
        stage_names = [s.value for s in PropagationStage if stage_uncertainties[s]]
        avg_uncertainties = [np.mean(stage_uncertainties[s]) for s in PropagationStage if stage_uncertainties[s]]
        std_uncertainties = [np.std(stage_uncertainties[s]) for s in PropagationStage if stage_uncertainties[s]]
        
        bars = ax1.bar(range(len(stage_names)), avg_uncertainties, 
                      yerr=std_uncertainties, capsize=5,
                      color='steelblue', alpha=0.7, error_kw={'linewidth': 2})
        ax1.set_xticks(range(len(stage_names)))
        ax1.set_xticklabels(stage_names, rotation=45, ha='right')
        ax1.set_ylabel('Average Uncertainty', fontsize=11)
        ax1.set_title('Average Uncertainty by Stage', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Distribution of dominant sources
        ax2 = axes[0, 1]
        source_counts = {}
        for source in dominant_sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        sources = list(source_counts.keys())
        counts = list(source_counts.values())
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(sources)))
        
        wedges, texts, autotexts = ax2.pie(counts, labels=sources, autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title('Dominant Error Sources', fontsize=12, fontweight='bold')
        
        # Plot 3: Amplification factor distribution
        ax3 = axes[1, 0]
        ax3.hist(amplification_factors, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(amplification_factors), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(amplification_factors):.2f}x')
        ax3.set_xlabel('Amplification Factor', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Error Amplification Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Uncertainty vs amplification
        ax4 = axes[1, 1]
        final_uncertainties = [chain.stages[-1].uncertainty for chain in chains]
        scatter = ax4.scatter(amplification_factors, final_uncertainties, 
                            c=amplification_factors, cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='black')
        ax4.set_xlabel('Amplification Factor', fontsize=11)
        ax4.set_ylabel('Final Uncertainty', fontsize=11)
        ax4.set_title('Final Uncertainty vs Amplification', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Amplification')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved multi-chain visualization to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def generate_improvement_recommendations(
        self,
        chains: Optional[List[PropagationChain]] = None
    ) -> List[Dict[str, Any]]:
        """Generate targeted recommendations for reducing uncertainty.
        
        Analyzes dominant error sources and suggests specific improvements.
        
        Args:
            chains: List of chains to analyze (default: all stored chains)
            
        Returns:
            List of recommendation dictionaries
        """
        if chains is None:
            chains = self.chains
        
        if not chains:
            return []
        
        # Identify dominant sources
        dominant_sources = self.identify_dominant_error_sources(chains)
        
        recommendations = []
        
        for stage, contribution in dominant_sources.items():
            if contribution < 0.1:  # Skip minor contributors
                continue
            
            rec = {
                'stage': stage.value,
                'contribution': contribution,
                'priority': 'HIGH' if contribution > 0.4 else 'MEDIUM' if contribution > 0.2 else 'LOW',
                'recommendations': []
            }
            
            if stage == PropagationStage.VAE_MEASUREMENT:
                rec['recommendations'] = [
                    "Increase number of temperature points for better fits",
                    "Improve VAE training (more epochs, better architecture)",
                    "Increase Monte Carlo samples for smoother data",
                    "Use ensemble methods to reduce measurement variance"
                ]
            
            elif stage == PropagationStage.BOOTSTRAP_RESAMPLING:
                rec['recommendations'] = [
                    "Increase number of bootstrap samples (current: 1000)",
                    "Check for distribution skewness - may need robust methods",
                    "Ensure sufficient data for stable bootstrap estimates",
                    "Consider parametric bootstrap if distribution is known"
                ]
            
            elif stage == PropagationStage.EFFECT_SIZE_CALCULATION:
                rec['recommendations'] = [
                    "Reduce measurement uncertainty (see VAE stage)",
                    "Get better theoretical predictions with error bars",
                    "Use multiple reference values for comparison",
                    "Consider alternative effect size metrics"
                ]
            
            elif stage == PropagationStage.ANOMALY_THRESHOLD:
                rec['recommendations'] = [
                    "Adjust anomaly threshold based on field standards",
                    "Use adaptive thresholds based on measurement quality",
                    "Implement confidence intervals for classifications",
                    "Consider Bayesian approach for threshold decisions"
                ]
            
            recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        self.logger.info(f"Generated {len(recommendations)} improvement recommendations")
        
        return recommendations
    
    def print_recommendations(
        self,
        recommendations: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Print improvement recommendations in readable format.
        
        Args:
            recommendations: List of recommendations (default: generate new ones)
        """
        if recommendations is None:
            recommendations = self.generate_improvement_recommendations()
        
        if not recommendations:
            print("No recommendations available. Create propagation chains first.")
            return
        
        print("\n" + "="*70)
        print("ERROR PROPAGATION IMPROVEMENT RECOMMENDATIONS")
        print("="*70)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['stage'].upper()}")
            print(f"   Priority: {rec['priority']}")
            print(f"   Contribution to Total Uncertainty: {rec['contribution']:.1%}")
            print(f"   Recommendations:")
            for j, suggestion in enumerate(rec['recommendations'], 1):
                print(f"      {j}) {suggestion}")
        
        print("\n" + "="*70)
    
    def get_summary_statistics(
        self,
        chains: Optional[List[PropagationChain]] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for error propagation analysis.
        
        Args:
            chains: List of chains to analyze (default: all stored chains)
            
        Returns:
            Dictionary with summary statistics
        """
        if chains is None:
            chains = self.chains
        
        if not chains:
            return {'n_chains': 0}
        
        # Calculate statistics
        amplifications = [c.amplification_factor for c in chains]
        final_uncertainties = [c.stages[-1].uncertainty for c in chains]
        
        dominant_sources = self.identify_dominant_error_sources(chains)
        most_dominant = max(dominant_sources.items(), key=lambda x: x[1])
        
        return {
            'n_chains': len(chains),
            'amplification': {
                'mean': np.mean(amplifications),
                'std': np.std(amplifications),
                'min': np.min(amplifications),
                'max': np.max(amplifications)
            },
            'final_uncertainty': {
                'mean': np.mean(final_uncertainties),
                'std': np.std(final_uncertainties),
                'min': np.min(final_uncertainties),
                'max': np.max(final_uncertainties)
            },
            'dominant_source': {
                'stage': most_dominant[0].value,
                'contribution': most_dominant[1]
            },
            'stage_contributions': {k.value: v for k, v in dominant_sources.items()}
        }
    
    def clear_chains(self) -> None:
        """Clear all stored propagation chains."""
        self.chains.clear()
        self.logger.info("Cleared all propagation chains")


# Import scipy.stats for skewness calculation
from scipy import stats
