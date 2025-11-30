"""
DTFIM Anomaly Characterization Module.

Implements Task 11: Characterize DTFIM anomalies
- 11.1 Classify: Griffiths phase? Infinite-randomness? New?
- 11.2 Measure dynamical exponent z
- 11.3 Check for rare region effects
- 11.4 Compare with known disordered QCP behavior

This module provides comprehensive characterization of anomalies found in the
Disordered Transverse Field Ising Model (DTFIM) exploration.

Key Physics Background:
- Griffiths Phase: Rare regions with locally ordered spins cause power-law
  singularities with non-universal exponents. Characterized by z > 1.
- Infinite-Randomness Fixed Point (IRFP): Strong disorder RG flows to a fixed
  point with z → ∞ (activated scaling). Characterized by log(L) scaling.
- Clean QCP: Standard quantum critical point with z = 1 for 1D TFIM.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging
from scipy.optimize import curve_fit
from scipy.stats import linregress

from .dtfim_refined_explorer import RefinedExplorationResult, RefinedScanPoint


class AnomalyType(Enum):
    """Classification of DTFIM anomaly types."""
    CLEAN_QCP = "clean_qcp"  # Standard quantum critical point (z=1)
    GRIFFITHS_PHASE = "griffiths_phase"  # Griffiths singularities (1 < z < ∞)
    INFINITE_RANDOMNESS = "infinite_randomness"  # IRFP (z → ∞, activated scaling)
    WEAK_DISORDER = "weak_disorder"  # Perturbative disorder effects
    NOVEL_PHASE = "novel_phase"  # Potentially new physics
    UNKNOWN = "unknown"


@dataclass
class KnownDisorderedQCP:
    """Known disordered quantum critical point behavior."""
    name: str
    dimension: int
    z: float  # Dynamical exponent
    z_error: float
    nu: float  # Correlation length exponent
    nu_error: float
    psi: float  # Tunneling exponent (for IRFP)
    psi_error: float
    description: str
    scaling_type: str  # "power_law" or "activated"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'dimension': self.dimension,
            'z': self.z,
            'z_error': self.z_error,
            'nu': self.nu,
            'nu_error': self.nu_error,
            'psi': self.psi,
            'psi_error': self.psi_error,
            'description': self.description,
            'scaling_type': self.scaling_type,
        }


# Database of known disordered QCP behaviors
KNOWN_DISORDERED_QCPS = {
    '1d_clean_tfim': KnownDisorderedQCP(
        name='1D Clean TFIM',
        dimension=1,
        z=1.0, z_error=0.0,
        nu=1.0, nu_error=0.0,
        psi=0.0, psi_error=0.0,
        description='Clean 1D transverse field Ising model',
        scaling_type='power_law'
    ),
    '1d_irfp': KnownDisorderedQCP(
        name='1D Infinite-Randomness Fixed Point',
        dimension=1,
        z=float('inf'), z_error=0.0,  # Activated scaling
        nu=2.0, nu_error=0.1,  # Exact for IRFP
        psi=0.5, psi_error=0.05,  # Tunneling exponent
        description='1D TFIM with strong disorder - IRFP',
        scaling_type='activated'
    ),
    '1d_griffiths': KnownDisorderedQCP(
        name='1D Griffiths Phase',
        dimension=1,
        z=2.0, z_error=0.5,  # Non-universal, typically 1 < z < ∞
        nu=1.5, nu_error=0.3,
        psi=0.0, psi_error=0.0,
        description='1D TFIM Griffiths phase with rare regions',
        scaling_type='power_law'
    ),
    '2d_clean_tfim': KnownDisorderedQCP(
        name='2D Clean TFIM',
        dimension=2,
        z=1.0, z_error=0.0,
        nu=0.6301, nu_error=0.0004,  # 3D Ising universality
        psi=0.0, psi_error=0.0,
        description='Clean 2D transverse field Ising model',
        scaling_type='power_law'
    ),
    '2d_disordered_tfim': KnownDisorderedQCP(
        name='2D Disordered TFIM',
        dimension=2,
        z=1.5, z_error=0.2,  # Approximate
        nu=1.0, nu_error=0.2,
        psi=0.0, psi_error=0.0,
        description='2D TFIM with disorder - modified exponents',
        scaling_type='power_law'
    ),
}


@dataclass
class DynamicalExponentResult:
    """Result of dynamical exponent z measurement."""
    z: float
    z_error: float
    method: str  # 'gap_scaling', 'correlation_time', 'susceptibility'
    fit_quality: float  # R² or similar
    system_sizes: List[int]
    raw_data: Dict[str, np.ndarray]
    is_activated: bool  # True if z → ∞ (activated scaling)
    psi: Optional[float] = None  # Tunneling exponent for activated scaling
    psi_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'z': self.z if not self.is_activated else 'inf',
            'z_error': self.z_error,
            'method': self.method,
            'fit_quality': self.fit_quality,
            'system_sizes': self.system_sizes,
            'is_activated': self.is_activated,
            'psi': self.psi,
            'psi_error': self.psi_error,
        }


@dataclass
class RareRegionAnalysis:
    """Analysis of rare region effects."""
    has_rare_regions: bool
    rare_region_strength: float  # 0 to 1
    distribution_type: str  # 'gaussian', 'power_law', 'bimodal'
    tail_exponent: Optional[float]  # For power-law tails
    evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_rare_regions': self.has_rare_regions,
            'rare_region_strength': self.rare_region_strength,
            'distribution_type': self.distribution_type,
            'tail_exponent': self.tail_exponent,
            'evidence': self.evidence,
        }


@dataclass
class AnomalyClassification:
    """Complete classification of a DTFIM anomaly."""
    anomaly_type: AnomalyType
    confidence: float  # 0 to 1
    z_result: DynamicalExponentResult
    rare_region_analysis: RareRegionAnalysis
    comparison_results: Dict[str, float]  # Deviation from known QCPs
    best_match: str  # Name of best matching known QCP
    is_novel: bool  # True if doesn't match any known behavior
    novelty_sigma: float  # Deviation from best match in sigma
    evidence_summary: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_type': self.anomaly_type.value,
            'confidence': self.confidence,
            'z_result': self.z_result.to_dict(),
            'rare_region_analysis': self.rare_region_analysis.to_dict(),
            'comparison_results': self.comparison_results,
            'best_match': self.best_match,
            'is_novel': self.is_novel,
            'novelty_sigma': self.novelty_sigma,
            'evidence_summary': self.evidence_summary,
            'recommendations': self.recommendations,
        }




class DTFIMAnomalyCharacterizer:
    """
    Comprehensive characterization of DTFIM anomalies.
    
    Implements:
    - Classification into Griffiths, IRFP, or novel phases
    - Dynamical exponent z measurement
    - Rare region effect detection
    - Comparison with known disordered QCP behavior
    """
    
    def __init__(
        self,
        known_qcps: Optional[Dict[str, KnownDisorderedQCP]] = None,
        novelty_threshold: float = 3.0,  # Sigma threshold for novelty
    ):
        """
        Initialize characterizer.
        
        Args:
            known_qcps: Database of known QCPs (default: KNOWN_DISORDERED_QCPS)
            novelty_threshold: Threshold in sigma for declaring novelty
        """
        self.known_qcps = known_qcps or KNOWN_DISORDERED_QCPS
        self.novelty_threshold = novelty_threshold
        self.logger = logging.getLogger(__name__)
    
    def characterize_anomaly(
        self,
        exploration_result: RefinedExplorationResult,
        h_target: Optional[float] = None,
        W_target: Optional[float] = None,
    ) -> AnomalyClassification:
        """
        Perform complete characterization of a DTFIM anomaly.
        
        Args:
            exploration_result: Results from refined exploration
            h_target: Target h value (default: center of anomalous region)
            W_target: Target W value (default: center of anomalous region)
            
        Returns:
            Complete AnomalyClassification
        """
        # Default to center of anomalous region
        if h_target is None:
            h_target = exploration_result.anomalous_region.h_center
        if W_target is None:
            W_target = exploration_result.anomalous_region.W_center
        
        self.logger.info(f"Characterizing anomaly at h={h_target:.3f}, W={W_target:.3f}")
        
        # Step 1: Measure dynamical exponent z
        z_result = self.measure_dynamical_exponent(
            exploration_result, h_target, W_target
        )
        
        # Step 2: Analyze rare region effects
        rare_region_analysis = self.analyze_rare_regions(
            exploration_result, h_target, W_target
        )
        
        # Step 3: Compare with known QCPs
        comparison_results = self.compare_with_known_qcps(
            z_result, exploration_result, h_target, W_target
        )
        
        # Step 4: Classify anomaly type
        anomaly_type, confidence = self._classify_anomaly_type(
            z_result, rare_region_analysis, comparison_results
        )
        
        # Step 5: Determine novelty
        best_match, novelty_sigma = self._find_best_match(comparison_results)
        is_novel = novelty_sigma > self.novelty_threshold
        
        # Step 6: Generate evidence summary and recommendations
        evidence_summary = self._generate_evidence_summary(
            anomaly_type, z_result, rare_region_analysis, comparison_results
        )
        recommendations = self._generate_recommendations(
            anomaly_type, is_novel, z_result, rare_region_analysis
        )
        
        return AnomalyClassification(
            anomaly_type=anomaly_type,
            confidence=confidence,
            z_result=z_result,
            rare_region_analysis=rare_region_analysis,
            comparison_results=comparison_results,
            best_match=best_match,
            is_novel=is_novel,
            novelty_sigma=novelty_sigma,
            evidence_summary=evidence_summary,
            recommendations=recommendations,
        )
    
    def measure_dynamical_exponent(
        self,
        exploration_result: RefinedExplorationResult,
        h_target: float,
        W_target: float,
    ) -> DynamicalExponentResult:
        """
        Measure dynamical exponent z using finite-size scaling.
        
        The dynamical exponent z relates energy gap Δ to system size L:
        - Power-law scaling: Δ ~ L^(-z)
        - Activated scaling (IRFP): Δ ~ exp(-c * L^ψ)
        
        Args:
            exploration_result: Refined exploration results
            h_target: Target h value
            W_target: Target W value
            
        Returns:
            DynamicalExponentResult with z measurement
        """
        system_sizes = exploration_result.system_sizes
        
        # Extract energy gaps at target point for each system size
        gaps = []
        sizes = []
        
        for L in system_sizes:
            points = exploration_result.get_points_for_size(L)
            # Find closest point to target
            closest = min(
                points,
                key=lambda p: (p.h - h_target)**2 + (p.W - W_target)**2
            )
            if closest.energy_gap > 0:
                gaps.append(closest.energy_gap)
                sizes.append(L)
        
        if len(sizes) < 2:
            self.logger.warning("Insufficient data for z measurement")
            return DynamicalExponentResult(
                z=1.0, z_error=float('inf'),
                method='gap_scaling',
                fit_quality=0.0,
                system_sizes=sizes,
                raw_data={'L': np.array(sizes), 'gap': np.array(gaps)},
                is_activated=False,
            )
        
        sizes = np.array(sizes)
        gaps = np.array(gaps)
        
        # Try power-law fit: Δ = A * L^(-z)
        # log(Δ) = log(A) - z * log(L)
        log_L = np.log(sizes)
        log_gap = np.log(gaps)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_L, log_gap)
        z_power_law = -slope
        z_error_power_law = std_err
        r_squared_power_law = r_value**2
        
        # Try activated fit: Δ = A * exp(-c * L^ψ)
        # log(Δ) = log(A) - c * L^ψ
        # For ψ = 0.5 (typical IRFP): log(Δ) = log(A) - c * sqrt(L)
        sqrt_L = np.sqrt(sizes)
        slope_act, intercept_act, r_value_act, _, std_err_act = linregress(sqrt_L, log_gap)
        r_squared_activated = r_value_act**2
        
        # Determine which scaling is better
        is_activated = r_squared_activated > r_squared_power_law + 0.05
        
        if is_activated:
            # Activated scaling detected
            psi = 0.5  # Assume standard IRFP value
            return DynamicalExponentResult(
                z=float('inf'),
                z_error=0.0,
                method='gap_scaling_activated',
                fit_quality=r_squared_activated,
                system_sizes=list(sizes),
                raw_data={'L': sizes, 'gap': gaps, 'log_gap': log_gap},
                is_activated=True,
                psi=psi,
                psi_error=0.1,
            )
        else:
            # Power-law scaling
            return DynamicalExponentResult(
                z=z_power_law,
                z_error=z_error_power_law,
                method='gap_scaling_power_law',
                fit_quality=r_squared_power_law,
                system_sizes=list(sizes),
                raw_data={'L': sizes, 'gap': gaps, 'log_gap': log_gap},
                is_activated=False,
            )
    
    def analyze_rare_regions(
        self,
        exploration_result: RefinedExplorationResult,
        h_target: float,
        W_target: float,
    ) -> RareRegionAnalysis:
        """
        Analyze rare region effects in the DTFIM.
        
        Rare regions are locally ordered regions that persist into the
        disordered phase, causing Griffiths singularities.
        
        Signatures:
        - Non-Gaussian distribution of observables
        - Power-law tails in susceptibility distribution
        - Strong sample-to-sample fluctuations
        
        Args:
            exploration_result: Refined exploration results
            h_target: Target h value
            W_target: Target W value
            
        Returns:
            RareRegionAnalysis with rare region characterization
        """
        evidence = {}
        
        # Get data at target point for largest system size
        L_max = max(exploration_result.system_sizes)
        points = exploration_result.get_points_for_size(L_max)
        closest = min(
            points,
            key=lambda p: (p.h - h_target)**2 + (p.W - W_target)**2
        )
        
        # Check for large sample-to-sample fluctuations
        # Rare regions cause large variance in observables
        relative_std_mag = closest.magnetization_z_std / (closest.magnetization_z + 1e-10)
        relative_std_chi = closest.susceptibility_z_std / (closest.susceptibility_z + 1e-10)
        
        evidence['relative_std_magnetization'] = relative_std_mag
        evidence['relative_std_susceptibility'] = relative_std_chi
        
        # Large relative fluctuations indicate rare regions
        fluctuation_strength = (relative_std_mag + relative_std_chi) / 2
        
        # Check entanglement spectrum for rare region signatures
        # Rare regions cause broad entanglement spectrum
        if closest.entanglement_spectrum_mean is not None:
            spectrum = closest.entanglement_spectrum_mean
            # Participation ratio: measures how spread out the spectrum is
            participation_ratio = np.sum(spectrum)**2 / np.sum(spectrum**2)
            evidence['entanglement_participation_ratio'] = participation_ratio
            
            # High participation ratio indicates broad spectrum (rare regions)
            spectrum_broadness = participation_ratio / len(spectrum)
        else:
            spectrum_broadness = 0.0
        
        # Determine distribution type based on fluctuations
        if fluctuation_strength > 0.5:
            distribution_type = 'power_law'  # Heavy tails
            tail_exponent = 1.0 / (fluctuation_strength + 0.1)  # Rough estimate
        elif fluctuation_strength > 0.2:
            distribution_type = 'bimodal'  # Some rare regions
            tail_exponent = None
        else:
            distribution_type = 'gaussian'  # Normal fluctuations
            tail_exponent = None
        
        # Compute rare region strength
        rare_region_strength = min(1.0, fluctuation_strength + spectrum_broadness)
        has_rare_regions = rare_region_strength > 0.3
        
        evidence['fluctuation_strength'] = fluctuation_strength
        evidence['spectrum_broadness'] = spectrum_broadness
        
        return RareRegionAnalysis(
            has_rare_regions=has_rare_regions,
            rare_region_strength=rare_region_strength,
            distribution_type=distribution_type,
            tail_exponent=tail_exponent,
            evidence=evidence,
        )
    
    def compare_with_known_qcps(
        self,
        z_result: DynamicalExponentResult,
        exploration_result: RefinedExplorationResult,
        h_target: float,
        W_target: float,
    ) -> Dict[str, float]:
        """
        Compare measured properties with known disordered QCPs.
        
        Args:
            z_result: Measured dynamical exponent
            exploration_result: Refined exploration results
            h_target: Target h value
            W_target: Target W value
            
        Returns:
            Dictionary of {qcp_name: deviation_in_sigma}
        """
        comparison_results = {}
        
        for qcp_name, qcp in self.known_qcps.items():
            # Compare dynamical exponent
            if z_result.is_activated and qcp.scaling_type == 'activated':
                # Both activated - compare psi
                if z_result.psi is not None and qcp.psi > 0:
                    psi_deviation = abs(z_result.psi - qcp.psi) / (
                        np.sqrt(z_result.psi_error**2 + qcp.psi_error**2) + 0.01
                    )
                else:
                    psi_deviation = 0.0
                comparison_results[qcp_name] = psi_deviation
            elif not z_result.is_activated and qcp.scaling_type == 'power_law':
                # Both power-law - compare z
                z_deviation = abs(z_result.z - qcp.z) / (
                    np.sqrt(z_result.z_error**2 + qcp.z_error**2) + 0.01
                )
                comparison_results[qcp_name] = z_deviation
            else:
                # Different scaling types - large deviation
                comparison_results[qcp_name] = 10.0  # Very different
        
        return comparison_results
    
    def _classify_anomaly_type(
        self,
        z_result: DynamicalExponentResult,
        rare_region_analysis: RareRegionAnalysis,
        comparison_results: Dict[str, float],
    ) -> Tuple[AnomalyType, float]:
        """
        Classify the anomaly type based on all evidence.
        
        Returns:
            (AnomalyType, confidence)
        """
        # Check for activated scaling (IRFP)
        if z_result.is_activated:
            return AnomalyType.INFINITE_RANDOMNESS, 0.9
        
        # Check dynamical exponent
        z = z_result.z
        z_err = z_result.z_error
        
        # Clean QCP: z ≈ 1
        if abs(z - 1.0) < 2 * z_err and not rare_region_analysis.has_rare_regions:
            return AnomalyType.CLEAN_QCP, 0.8
        
        # Weak disorder: z slightly > 1, weak rare regions
        if 1.0 < z < 1.5 and rare_region_analysis.rare_region_strength < 0.3:
            return AnomalyType.WEAK_DISORDER, 0.7
        
        # Griffiths phase: 1 < z < ∞, strong rare regions
        if z > 1.0 and rare_region_analysis.has_rare_regions:
            return AnomalyType.GRIFFITHS_PHASE, 0.85
        
        # Check if matches any known QCP
        min_deviation = min(comparison_results.values()) if comparison_results else float('inf')
        if min_deviation < self.novelty_threshold:
            # Matches a known QCP
            best_match = min(comparison_results, key=comparison_results.get)
            if 'irfp' in best_match.lower():
                return AnomalyType.INFINITE_RANDOMNESS, 0.8
            elif 'griffiths' in best_match.lower():
                return AnomalyType.GRIFFITHS_PHASE, 0.8
            elif 'clean' in best_match.lower():
                return AnomalyType.CLEAN_QCP, 0.8
            else:
                return AnomalyType.WEAK_DISORDER, 0.6
        
        # Doesn't match known behavior - potentially novel
        return AnomalyType.NOVEL_PHASE, 0.7
    
    def _find_best_match(
        self,
        comparison_results: Dict[str, float],
    ) -> Tuple[str, float]:
        """Find best matching known QCP and deviation."""
        if not comparison_results:
            return 'none', float('inf')
        
        best_match = min(comparison_results, key=comparison_results.get)
        novelty_sigma = comparison_results[best_match]
        
        return best_match, novelty_sigma
    
    def _generate_evidence_summary(
        self,
        anomaly_type: AnomalyType,
        z_result: DynamicalExponentResult,
        rare_region_analysis: RareRegionAnalysis,
        comparison_results: Dict[str, float],
    ) -> str:
        """Generate human-readable evidence summary."""
        lines = []
        
        lines.append(f"Classification: {anomaly_type.value}")
        lines.append("")
        
        # Dynamical exponent
        if z_result.is_activated:
            lines.append(f"Dynamical scaling: ACTIVATED (z → ∞)")
            lines.append(f"  Tunneling exponent ψ = {z_result.psi:.3f} ± {z_result.psi_error:.3f}")
        else:
            lines.append(f"Dynamical exponent: z = {z_result.z:.3f} ± {z_result.z_error:.3f}")
        lines.append(f"  Fit quality (R²): {z_result.fit_quality:.3f}")
        lines.append(f"  Method: {z_result.method}")
        lines.append("")
        
        # Rare regions
        lines.append(f"Rare region analysis:")
        lines.append(f"  Has rare regions: {rare_region_analysis.has_rare_regions}")
        lines.append(f"  Strength: {rare_region_analysis.rare_region_strength:.3f}")
        lines.append(f"  Distribution type: {rare_region_analysis.distribution_type}")
        lines.append("")
        
        # Comparison with known QCPs
        lines.append("Comparison with known QCPs (deviation in σ):")
        for qcp_name, deviation in sorted(comparison_results.items(), key=lambda x: x[1]):
            lines.append(f"  {qcp_name}: {deviation:.2f}σ")
        
        return "\n".join(lines)
    
    def _generate_recommendations(
        self,
        anomaly_type: AnomalyType,
        is_novel: bool,
        z_result: DynamicalExponentResult,
        rare_region_analysis: RareRegionAnalysis,
    ) -> List[str]:
        """Generate recommendations for further investigation."""
        recommendations = []
        
        if is_novel:
            recommendations.append(
                "POTENTIAL NOVEL PHYSICS: Anomaly doesn't match known QCP behavior. "
                "Proceed to rigorous validation (Month 3 tasks)."
            )
        
        if anomaly_type == AnomalyType.GRIFFITHS_PHASE:
            recommendations.append(
                "Griffiths phase detected. Measure dynamical exponent z more precisely "
                "using larger system sizes and more disorder realizations."
            )
            recommendations.append(
                "Check for power-law singularities in susceptibility: χ ~ |h-hc|^(-1/z)."
            )
        
        if anomaly_type == AnomalyType.INFINITE_RANDOMNESS:
            recommendations.append(
                "Infinite-randomness fixed point detected. Verify activated scaling "
                "with larger system sizes."
            )
            recommendations.append(
                "Measure tunneling exponent ψ precisely. For 1D TFIM IRFP, ψ = 0.5."
            )
        
        if z_result.fit_quality < 0.9:
            recommendations.append(
                f"Fit quality is low (R² = {z_result.fit_quality:.3f}). "
                "Increase system sizes and disorder realizations for better statistics."
            )
        
        if rare_region_analysis.has_rare_regions:
            recommendations.append(
                "Strong rare region effects detected. Consider using larger system sizes "
                "to reduce finite-size effects from rare regions."
            )
        
        if anomaly_type == AnomalyType.UNKNOWN:
            recommendations.append(
                "Unable to classify anomaly. Collect more data at different disorder "
                "strengths to map out the phase diagram more completely."
            )
        
        return recommendations



@dataclass
class CharacterizationReport:
    """Complete characterization report for DTFIM anomalies."""
    classifications: List[AnomalyClassification]
    summary: Dict[str, Any]
    phase_diagram_analysis: Dict[str, Any]
    recommendations: List[str]
    
    def save(self, filepath: str):
        """Save report to JSON."""
        data = {
            'classifications': [c.to_dict() for c in self.classifications],
            'summary': self.summary,
            'phase_diagram_analysis': self.phase_diagram_analysis,
            'recommendations': self.recommendations,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("DTFIM ANOMALY CHARACTERIZATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        for key, value in self.summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Classifications
        lines.append("ANOMALY CLASSIFICATIONS")
        lines.append("-" * 40)
        for i, classification in enumerate(self.classifications, 1):
            lines.append(f"\n{i}. {classification.anomaly_type.value.upper()}")
            lines.append(f"   Confidence: {classification.confidence:.1%}")
            lines.append(f"   Best match: {classification.best_match}")
            lines.append(f"   Novelty: {classification.novelty_sigma:.2f}σ from best match")
            lines.append(f"   Is novel: {classification.is_novel}")
            lines.append("")
            lines.append("   Evidence:")
            for line in classification.evidence_summary.split('\n'):
                lines.append(f"   {line}")
        
        # Phase diagram analysis
        lines.append("\nPHASE DIAGRAM ANALYSIS")
        lines.append("-" * 40)
        for key, value in self.phase_diagram_analysis.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        
        # Recommendations
        lines.append("\nRECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def characterize_dtfim_anomalies(
    exploration_result: RefinedExplorationResult,
    output_dir: Optional[str] = None,
) -> CharacterizationReport:
    """
    Main function to characterize all DTFIM anomalies from exploration results.
    
    Args:
        exploration_result: Results from refined DTFIM exploration
        output_dir: Directory to save results (optional)
        
    Returns:
        CharacterizationReport with complete analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting DTFIM anomaly characterization")
    
    characterizer = DTFIMAnomalyCharacterizer()
    
    # Characterize at center of anomalous region
    region = exploration_result.anomalous_region
    classification = characterizer.characterize_anomaly(
        exploration_result,
        h_target=region.h_center,
        W_target=region.W_center,
    )
    
    classifications = [classification]
    
    # Also characterize at corners of region for completeness
    corners = [
        (region.h_range[0], region.W_range[0]),
        (region.h_range[1], region.W_range[0]),
        (region.h_range[0], region.W_range[1]),
        (region.h_range[1], region.W_range[1]),
    ]
    
    for h, W in corners:
        try:
            corner_classification = characterizer.characterize_anomaly(
                exploration_result, h_target=h, W_target=W
            )
            classifications.append(corner_classification)
        except Exception as e:
            logger.warning(f"Failed to characterize at ({h}, {W}): {e}")
    
    # Generate summary
    summary = {
        'total_classifications': len(classifications),
        'anomaly_types': {
            t.value: sum(1 for c in classifications if c.anomaly_type == t)
            for t in AnomalyType
        },
        'novel_count': sum(1 for c in classifications if c.is_novel),
        'avg_confidence': np.mean([c.confidence for c in classifications]),
        'region': {
            'h_center': region.h_center,
            'W_center': region.W_center,
            'h_range': region.h_range,
            'W_range': region.W_range,
        },
    }
    
    # Phase diagram analysis
    phase_diagram_analysis = _analyze_phase_diagram(exploration_result, classifications)
    
    # Collect all recommendations
    all_recommendations = []
    for c in classifications:
        all_recommendations.extend(c.recommendations)
    # Deduplicate
    recommendations = list(dict.fromkeys(all_recommendations))
    
    report = CharacterizationReport(
        classifications=classifications,
        summary=summary,
        phase_diagram_analysis=phase_diagram_analysis,
        recommendations=recommendations,
    )
    
    # Save if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report.save(str(output_path / 'characterization_report.json'))
        
        with open(output_path / 'characterization_report.txt', 'w', encoding='utf-8') as f:
            f.write(report.generate_text_report())
        
        logger.info(f"Saved characterization report to {output_dir}")
    
    return report


def _analyze_phase_diagram(
    exploration_result: RefinedExplorationResult,
    classifications: List[AnomalyClassification],
) -> Dict[str, Any]:
    """Analyze the phase diagram structure from exploration results."""
    
    analysis = {}
    
    # Get the largest system size data
    L_max = max(exploration_result.system_sizes)
    points = exploration_result.get_points_for_size(L_max)
    
    if not points:
        return {'error': 'No data available'}
    
    # Find critical line (maximum susceptibility)
    h_values = exploration_result.h_values
    W_values = exploration_result.W_values
    
    critical_line = []
    for W in W_values:
        W_points = [p for p in points if abs(p.W - W) < 0.01]
        if W_points:
            max_chi_point = max(W_points, key=lambda p: p.susceptibility_z)
            critical_line.append({
                'W': W,
                'h_c': max_chi_point.h,
                'chi_max': max_chi_point.susceptibility_z,
            })
    
    analysis['critical_line'] = critical_line
    
    # Estimate phase boundaries
    if critical_line:
        h_c_values = [p['h_c'] for p in critical_line]
        analysis['h_c_range'] = (min(h_c_values), max(h_c_values))
        analysis['h_c_mean'] = np.mean(h_c_values)
        analysis['h_c_std'] = np.std(h_c_values)
    
    # Analyze disorder dependence
    if len(critical_line) >= 2:
        W_arr = np.array([p['W'] for p in critical_line])
        h_c_arr = np.array([p['h_c'] for p in critical_line])
        
        # Linear fit: h_c(W) = h_c(0) + slope * W
        if len(W_arr) >= 2:
            slope, intercept, r_value, _, _ = linregress(W_arr, h_c_arr)
            analysis['h_c_vs_W'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
            }
    
    # Identify phase regions
    analysis['phases'] = {
        'ordered': 'h < h_c (ferromagnetic)',
        'disordered': 'h > h_c (paramagnetic)',
        'critical': f'h ≈ {analysis.get("h_c_mean", "?")} (quantum critical)',
    }
    
    # Dominant anomaly type
    type_counts = {}
    for c in classifications:
        t = c.anomaly_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    
    if type_counts:
        dominant_type = max(type_counts, key=type_counts.get)
        analysis['dominant_anomaly_type'] = dominant_type
    
    return analysis

