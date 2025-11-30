# -*- coding: utf-8 -*-
"""
Multi-Variant Comparative Analysis Engine.

This module provides comprehensive comparative analysis across multiple variants,
including clustering, outlier detection, trend analysis, and pattern identification.

Requirements 6.2, 6.3, 6.4: Generate comparative analysis, identify clusters and 
outliers, detect trends in how parameters affect critical behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from .variant_database import VariantDatabase, VariantRecord
from .comparative_analyzer import ComparativeAnalyzer
from ..utils.logging_utils import get_logger


@dataclass
class ClusterResult:
    """Result of clustering analysis.
    
    Attributes:
        cluster_id: Cluster identifier
        variant_ids: Variants in this cluster
        centroid_exponents: Mean exponents for cluster
        cluster_size: Number of variants in cluster
        intra_cluster_variance: Variance within cluster
        representative_variant: Most representative variant
    """
    cluster_id: int
    variant_ids: List[str]
    centroid_exponents: Dict[str, float]
    cluster_size: int
    intra_cluster_variance: float
    representative_variant: str


@dataclass
class OutlierResult:
    """Result of outlier detection.
    
    Attributes:
        variant_id: Variant identified as outlier
        outlier_score: Outlier score (higher = more outlying)
        deviations: Deviations from cluster centroid
        nearest_cluster: Nearest cluster ID
        distance_to_cluster: Distance to nearest cluster
        reason: Human-readable reason for outlier status
    """
    variant_id: str
    outlier_score: float
    deviations: Dict[str, float]
    nearest_cluster: int
    distance_to_cluster: float
    reason: str



@dataclass
class TrendResult:
    """Result of trend analysis.
    
    Attributes:
        parameter_name: Name of parameter analyzed
        exponent_name: Name of exponent analyzed
        trend_type: Type of trend ('linear', 'nonlinear', 'none')
        correlation: Correlation coefficient
        p_value: Statistical significance
        slope: Slope of trend (if linear)
        r_squared: R² value for fit
        description: Human-readable description
    """
    parameter_name: str
    exponent_name: str
    trend_type: str
    correlation: float
    p_value: float
    slope: Optional[float]
    r_squared: float
    description: str


@dataclass
class ComparativeAnalysisReport:
    """Complete comparative analysis report.
    
    Attributes:
        n_variants: Number of variants analyzed
        clusters: Identified clusters
        outliers: Detected outliers
        trends: Detected trends
        exponent_correlations: Correlations between exponents
        summary_statistics: Summary statistics
        figures: Generated figures
    """
    n_variants: int
    clusters: List[ClusterResult]
    outliers: List[OutlierResult]
    trends: List[TrendResult]
    exponent_correlations: Dict[str, float]
    summary_statistics: Dict[str, Any]
    figures: List[plt.Figure] = field(default_factory=list)


class MultiVariantAnalyzer:
    """Comprehensive multi-variant comparative analysis.
    
    This class provides advanced comparative analysis capabilities including:
    - Hierarchical clustering of variants by exponents
    - Outlier detection using statistical methods
    - Trend analysis for parameter effects
    - Correlation analysis between exponents
    - Dimensionality reduction and visualization
    
    Attributes:
        database: Variant database
        comparative_analyzer: Base comparative analyzer
        logger: Logger instance
    """
    
    def __init__(self, database: VariantDatabase):
        """Initialize multi-variant analyzer.
        
        Args:
            database: Variant database with explored variants
        """
        self.database = database
        self.comparative_analyzer = ComparativeAnalyzer()
        self.logger = get_logger(__name__)
        self.logger.info("Initialized MultiVariantAnalyzer")

    
    def generate_comprehensive_analysis(
        self,
        output_dir: Optional[Path] = None
    ) -> ComparativeAnalysisReport:
        """Generate comprehensive comparative analysis across all variants.
        
        Requirement 6.2: Generate comparative analysis across all variants.
        
        Args:
            output_dir: Optional directory to save figures
            
        Returns:
            ComparativeAnalysisReport with complete analysis
        """
        self.logger.info("Generating comprehensive comparative analysis")
        
        variants = self.database.get_all_variants()
        
        if len(variants) < 2:
            self.logger.warning("Need at least 2 variants for comparative analysis")
            return ComparativeAnalysisReport(
                n_variants=len(variants),
                clusters=[],
                outliers=[],
                trends=[],
                exponent_correlations={},
                summary_statistics={}
            )
        
        # Perform clustering
        clusters = self.cluster_variants(variants)
        
        # Detect outliers
        outliers = self.detect_outliers(variants, clusters)
        
        # Analyze trends
        trends = self.analyze_trends(variants)
        
        # Compute exponent correlations
        correlations = self.compute_exponent_correlations(variants)
        
        # Get summary statistics
        summary_stats = self.database.get_summary_statistics()
        
        # Generate visualizations
        figures = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Cluster dendrogram
            fig = self.plot_cluster_dendrogram(variants)
            figures.append(fig)
            fig.savefig(output_dir / 'cluster_dendrogram.png', dpi=300, bbox_inches='tight')
            
            # Exponent scatter plots
            fig = self.plot_exponent_scatter(variants)
            figures.append(fig)
            fig.savefig(output_dir / 'exponent_scatter.png', dpi=300, bbox_inches='tight')
            
            # Trend plots
            fig = self.plot_trends(variants, trends)
            figures.append(fig)
            fig.savefig(output_dir / 'parameter_trends.png', dpi=300, bbox_inches='tight')
            
            # PCA visualization
            fig = self.plot_pca_visualization(variants, clusters)
            figures.append(fig)
            fig.savefig(output_dir / 'pca_visualization.png', dpi=300, bbox_inches='tight')
        
        report = ComparativeAnalysisReport(
            n_variants=len(variants),
            clusters=clusters,
            outliers=outliers,
            trends=trends,
            exponent_correlations=correlations,
            summary_statistics=summary_stats,
            figures=figures
        )
        
        self.logger.info(
            f"Comparative analysis complete: {len(clusters)} clusters, "
            f"{len(outliers)} outliers, {len(trends)} trends"
        )
        
        return report

    
    def cluster_variants(
        self,
        variants: List[VariantRecord],
        n_clusters: Optional[int] = None,
        method: str = 'ward'
    ) -> List[ClusterResult]:
        """Cluster variants by critical exponents.
        
        Requirement 6.3: Identify clusters of similar behavior.
        
        Args:
            variants: List of variant records
            n_clusters: Number of clusters (None = auto-determine)
            method: Linkage method ('ward', 'average', 'complete')
            
        Returns:
            List of cluster results
        """
        self.logger.info(f"Clustering {len(variants)} variants")
        
        if len(variants) < 2:
            return []
        
        # Extract exponent vectors
        exponent_matrix, exponent_names, variant_ids = self._build_exponent_matrix(variants)
        
        if exponent_matrix.shape[0] < 2:
            self.logger.warning("Insufficient variants with common exponents")
            return []
        
        # Normalize exponents (z-score)
        exponent_matrix_norm = (exponent_matrix - np.mean(exponent_matrix, axis=0)) / (np.std(exponent_matrix, axis=0) + 1e-10)
        
        # Compute distance matrix
        distances = pdist(exponent_matrix_norm, metric='euclidean')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method=method)
        
        # Determine number of clusters
        if n_clusters is None:
            # Use elbow method or silhouette score
            n_clusters = self._determine_optimal_clusters(exponent_matrix_norm, linkage_matrix)
        
        # Cut dendrogram to get clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Build cluster results
        clusters = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = cluster_labels == cluster_id
            cluster_variant_ids = [variant_ids[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if not cluster_variant_ids:
                continue
            
            # Compute centroid
            cluster_exponents = exponent_matrix[cluster_mask]
            centroid = np.mean(cluster_exponents, axis=0)
            centroid_dict = {name: float(centroid[i]) for i, name in enumerate(exponent_names)}
            
            # Compute intra-cluster variance
            variance = float(np.mean(np.var(cluster_exponents, axis=0)))
            
            # Find representative variant (closest to centroid)
            distances_to_centroid = np.linalg.norm(cluster_exponents - centroid, axis=1)
            representative_idx = np.argmin(distances_to_centroid)
            representative_variant = cluster_variant_ids[representative_idx]
            
            cluster_result = ClusterResult(
                cluster_id=cluster_id,
                variant_ids=cluster_variant_ids,
                centroid_exponents=centroid_dict,
                cluster_size=len(cluster_variant_ids),
                intra_cluster_variance=variance,
                representative_variant=representative_variant
            )
            clusters.append(cluster_result)
        
        self.logger.info(f"Identified {len(clusters)} clusters")
        return clusters

    
    def detect_outliers(
        self,
        variants: List[VariantRecord],
        clusters: List[ClusterResult],
        threshold: float = 3.0
    ) -> List[OutlierResult]:
        """Detect outlier variants.
        
        Requirement 6.3: Identify outliers.
        
        Args:
            variants: List of variant records
            clusters: Cluster results
            threshold: Outlier threshold in standard deviations
            
        Returns:
            List of outlier results
        """
        self.logger.info(f"Detecting outliers with threshold {threshold}σ")
        
        if not clusters:
            return []
        
        outliers = []
        
        # Build variant ID to cluster mapping
        variant_to_cluster = {}
        for cluster in clusters:
            for variant_id in cluster.variant_ids:
                variant_to_cluster[variant_id] = cluster
        
        # Extract exponent matrix
        exponent_matrix, exponent_names, variant_ids = self._build_exponent_matrix(variants)
        
        # For each variant, compute distance to its cluster centroid
        for i, variant_id in enumerate(variant_ids):
            if variant_id not in variant_to_cluster:
                continue
            
            cluster = variant_to_cluster[variant_id]
            centroid = np.array([cluster.centroid_exponents[name] for name in exponent_names])
            variant_exponents = exponent_matrix[i]
            
            # Compute deviations
            deviations = np.abs(variant_exponents - centroid)
            
            # Compute outlier score (max deviation in standard deviations)
            # Use cluster variance as reference
            std_devs = deviations / (np.sqrt(cluster.intra_cluster_variance) + 1e-10)
            outlier_score = float(np.max(std_devs))
            
            if outlier_score > threshold:
                # This is an outlier
                deviations_dict = {
                    name: float(std_devs[j])
                    for j, name in enumerate(exponent_names)
                }
                
                # Find which exponent is most outlying
                max_dev_idx = np.argmax(std_devs)
                max_dev_exponent = exponent_names[max_dev_idx]
                
                # Compute distance to nearest other cluster
                min_distance = float('inf')
                nearest_cluster_id = cluster.cluster_id
                for other_cluster in clusters:
                    if other_cluster.cluster_id == cluster.cluster_id:
                        continue
                    other_centroid = np.array([
                        other_cluster.centroid_exponents[name] for name in exponent_names
                    ])
                    distance = float(np.linalg.norm(variant_exponents - other_centroid))
                    if distance < min_distance:
                        min_distance = distance
                        nearest_cluster_id = other_cluster.cluster_id
                
                reason = (
                    f"Deviates {outlier_score:.1f}σ from cluster {cluster.cluster_id} "
                    f"(primarily in {max_dev_exponent})"
                )
                
                outlier = OutlierResult(
                    variant_id=variant_id,
                    outlier_score=outlier_score,
                    deviations=deviations_dict,
                    nearest_cluster=nearest_cluster_id,
                    distance_to_cluster=min_distance,
                    reason=reason
                )
                outliers.append(outlier)
        
        self.logger.info(f"Detected {len(outliers)} outliers")
        return outliers

    
    def analyze_trends(
        self,
        variants: List[VariantRecord],
        min_correlation: float = 0.5
    ) -> List[TrendResult]:
        """Analyze trends in how parameters affect critical behavior.
        
        Requirement 6.4: Detect trends in how parameters affect critical behavior.
        
        Args:
            variants: List of variant records
            min_correlation: Minimum correlation to report
            
        Returns:
            List of trend results
        """
        self.logger.info("Analyzing parameter trends")
        
        trends = []
        
        # Group variants by interaction type to analyze parameter effects
        interaction_groups = {}
        for variant in variants:
            itype = variant.variant_config.interaction_type
            if itype not in interaction_groups:
                interaction_groups[itype] = []
            interaction_groups[itype].append(variant)
        
        # For each interaction type, analyze parameter trends
        for itype, group_variants in interaction_groups.items():
            if len(group_variants) < 3:
                continue
            
            # Extract parameter values
            param_names = set()
            for variant in group_variants:
                param_names.update(variant.variant_config.interaction_params.keys())
            
            for param_name in param_names:
                # Get variants with this parameter
                param_variants = [
                    v for v in group_variants
                    if param_name in v.variant_config.interaction_params
                ]
                
                if len(param_variants) < 3:
                    continue
                
                param_values = [
                    v.variant_config.interaction_params[param_name]
                    for v in param_variants
                ]
                
                # Analyze trend for each exponent
                exponent_names = set()
                for variant in param_variants:
                    exponent_names.update(variant.measured_exponents.keys())
                
                for exp_name in exponent_names:
                    # Get exponent values
                    exp_values = []
                    param_vals_for_exp = []
                    
                    for variant in param_variants:
                        if exp_name in variant.measured_exponents:
                            exp_values.append(variant.measured_exponents[exp_name])
                            param_vals_for_exp.append(
                                variant.variant_config.interaction_params[param_name]
                            )
                    
                    if len(exp_values) < 3:
                        continue
                    
                    # Compute correlation
                    correlation, p_value = stats.pearsonr(param_vals_for_exp, exp_values)
                    
                    if abs(correlation) < min_correlation:
                        continue
                    
                    # Fit linear trend
                    slope, intercept, r_value, p_val_fit, std_err = stats.linregress(
                        param_vals_for_exp, exp_values
                    )
                    r_squared = r_value ** 2
                    
                    # Determine trend type
                    if r_squared > 0.8:
                        trend_type = 'linear'
                    elif abs(correlation) > 0.5:
                        trend_type = 'nonlinear'
                    else:
                        trend_type = 'none'
                    
                    description = (
                        f"{exp_name} {'increases' if slope > 0 else 'decreases'} "
                        f"with {param_name} (r={correlation:.3f}, p={p_value:.4f})"
                    )
                    
                    trend = TrendResult(
                        parameter_name=param_name,
                        exponent_name=exp_name,
                        trend_type=trend_type,
                        correlation=float(correlation),
                        p_value=float(p_value),
                        slope=float(slope),
                        r_squared=float(r_squared),
                        description=description
                    )
                    trends.append(trend)
        
        self.logger.info(f"Identified {len(trends)} significant trends")
        return trends

    
    def compute_exponent_correlations(
        self,
        variants: List[VariantRecord]
    ) -> Dict[str, float]:
        """Compute correlations between different exponents.
        
        Args:
            variants: List of variant records
            
        Returns:
            Dictionary mapping exponent pairs to correlations
        """
        self.logger.info("Computing exponent correlations")
        
        # Extract exponent matrix
        exponent_matrix, exponent_names, variant_ids = self._build_exponent_matrix(variants)
        
        if exponent_matrix.shape[0] < 3:
            return {}
        
        correlations = {}
        
        # Compute pairwise correlations
        for i, exp1 in enumerate(exponent_names):
            for j, exp2 in enumerate(exponent_names):
                if i >= j:
                    continue
                
                values1 = exponent_matrix[:, i]
                values2 = exponent_matrix[:, j]
                
                # Remove NaN values
                mask = ~(np.isnan(values1) | np.isnan(values2))
                if np.sum(mask) < 3:
                    continue
                
                correlation, p_value = stats.pearsonr(values1[mask], values2[mask])
                
                if abs(correlation) > 0.3:  # Only report significant correlations
                    correlations[f"{exp1}_vs_{exp2}"] = {
                        'correlation': float(correlation),
                        'p_value': float(p_value)
                    }
        
        return correlations
    
    def _build_exponent_matrix(
        self,
        variants: List[VariantRecord]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Build matrix of exponent values across variants.
        
        Args:
            variants: List of variant records
            
        Returns:
            Tuple of (exponent_matrix, exponent_names, variant_ids)
        """
        # Find common exponents
        all_exponents = set()
        for variant in variants:
            all_exponents.update(variant.measured_exponents.keys())
        
        exponent_names = sorted(all_exponents)
        variant_ids = [v.variant_id for v in variants]
        
        # Build matrix
        exponent_matrix = np.zeros((len(variants), len(exponent_names)))
        
        for i, variant in enumerate(variants):
            for j, exp_name in enumerate(exponent_names):
                if exp_name in variant.measured_exponents:
                    exponent_matrix[i, j] = variant.measured_exponents[exp_name]
                else:
                    exponent_matrix[i, j] = np.nan
        
        # Remove variants with too many missing exponents
        valid_mask = np.sum(~np.isnan(exponent_matrix), axis=1) >= len(exponent_names) * 0.5
        exponent_matrix = exponent_matrix[valid_mask]
        variant_ids = [vid for i, vid in enumerate(variant_ids) if valid_mask[i]]
        
        # Impute missing values with column mean
        for j in range(exponent_matrix.shape[1]):
            col = exponent_matrix[:, j]
            col_mean = np.nanmean(col)
            col[np.isnan(col)] = col_mean
            exponent_matrix[:, j] = col
        
        return exponent_matrix, exponent_names, variant_ids
    
    def _determine_optimal_clusters(
        self,
        data: np.ndarray,
        linkage_matrix: np.ndarray,
        max_clusters: int = 10
    ) -> int:
        """Determine optimal number of clusters using elbow method.
        
        Args:
            data: Data matrix
            linkage_matrix: Linkage matrix from hierarchical clustering
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        max_clusters = min(max_clusters, len(data) - 1)
        
        if max_clusters < 2:
            return 1
        
        # Compute within-cluster sum of squares for different k
        wcss = []
        for k in range(1, max_clusters + 1):
            if k == 1:
                wcss.append(np.sum(np.var(data, axis=0)))
            else:
                labels = fcluster(linkage_matrix, k, criterion='maxclust')
                wc_sum = 0
                for cluster_id in range(1, k + 1):
                    cluster_data = data[labels == cluster_id]
                    if len(cluster_data) > 0:
                        wc_sum += np.sum(np.var(cluster_data, axis=0))
                wcss.append(wc_sum)
        
        # Find elbow using second derivative
        if len(wcss) < 3:
            return 2
        
        wcss = np.array(wcss)
        second_deriv = np.diff(wcss, n=2)
        
        # Elbow is where second derivative is maximum
        elbow_idx = np.argmax(second_deriv) + 2  # +2 because of double diff
        
        # Ensure reasonable number of clusters (2-5 typically)
        optimal_k = min(max(elbow_idx, 2), 5)
        
        return optimal_k

    
    def plot_cluster_dendrogram(
        self,
        variants: List[VariantRecord]
    ) -> plt.Figure:
        """Plot hierarchical clustering dendrogram.
        
        Args:
            variants: List of variant records
            
        Returns:
            matplotlib Figure
        """
        exponent_matrix, exponent_names, variant_ids = self._build_exponent_matrix(variants)
        
        if exponent_matrix.shape[0] < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Insufficient data for dendrogram',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Normalize
        exponent_matrix_norm = (exponent_matrix - np.mean(exponent_matrix, axis=0)) / (np.std(exponent_matrix, axis=0) + 1e-10)
        
        # Compute linkage
        distances = pdist(exponent_matrix_norm, metric='euclidean')
        linkage_matrix = linkage(distances, method='ward')
        
        # Plot dendrogram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dendrogram(
            linkage_matrix,
            labels=variant_ids,
            ax=ax,
            leaf_font_size=10,
            leaf_rotation=90
        )
        
        ax.set_xlabel('Variant ID', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Hierarchical Clustering of Variants by Critical Exponents', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_exponent_scatter(
        self,
        variants: List[VariantRecord]
    ) -> plt.Figure:
        """Plot scatter matrix of exponents.
        
        Args:
            variants: List of variant records
            
        Returns:
            matplotlib Figure
        """
        exponent_matrix, exponent_names, variant_ids = self._build_exponent_matrix(variants)
        
        n_exponents = len(exponent_names)
        
        if n_exponents < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Need at least 2 exponents for scatter plot',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create scatter matrix
        fig, axes = plt.subplots(n_exponents, n_exponents, figsize=(12, 12))
        
        for i in range(n_exponents):
            for j in range(n_exponents):
                ax = axes[i, j] if n_exponents > 1 else axes
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(exponent_matrix[:, i], bins=10, alpha=0.7, color='blue')
                    ax.set_ylabel('Count', fontsize=8)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(exponent_matrix[:, j], exponent_matrix[:, i],
                             alpha=0.6, s=50)
                
                # Labels
                if i == n_exponents - 1:
                    ax.set_xlabel(exponent_names[j], fontsize=10)
                if j == 0:
                    ax.set_ylabel(exponent_names[i], fontsize=10)
                
                ax.tick_params(labelsize=8)
        
        plt.suptitle('Exponent Scatter Matrix', fontsize=14, y=0.995)
        plt.tight_layout()
        return fig

    
    def plot_trends(
        self,
        variants: List[VariantRecord],
        trends: List[TrendResult]
    ) -> plt.Figure:
        """Plot parameter trends.
        
        Args:
            variants: List of variant records
            trends: Trend results
            
        Returns:
            matplotlib Figure
        """
        if not trends:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No significant trends detected',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Plot top 6 trends
        n_plots = min(6, len(trends))
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, trend in enumerate(trends[:n_plots]):
            ax = axes[idx]
            
            # Get data for this trend
            param_values = []
            exp_values = []
            
            for variant in variants:
                if trend.parameter_name in variant.variant_config.interaction_params:
                    if trend.exponent_name in variant.measured_exponents:
                        param_values.append(
                            variant.variant_config.interaction_params[trend.parameter_name]
                        )
                        exp_values.append(variant.measured_exponents[trend.exponent_name])
            
            if not param_values:
                continue
            
            # Plot scatter
            ax.scatter(param_values, exp_values, alpha=0.6, s=100, label='Data')
            
            # Plot trend line
            if trend.slope is not None:
                x_range = np.linspace(min(param_values), max(param_values), 100)
                # Compute intercept from data
                intercept = np.mean(exp_values) - trend.slope * np.mean(param_values)
                y_trend = trend.slope * x_range + intercept
                ax.plot(x_range, y_trend, 'r--', linewidth=2, 
                       label=f'Trend (r²={trend.r_squared:.3f})')
            
            ax.set_xlabel(trend.parameter_name, fontsize=12)
            ax.set_ylabel(trend.exponent_name, fontsize=12)
            ax.set_title(f'{trend.exponent_name} vs {trend.parameter_name}', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Parameter Trends in Critical Exponents', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_pca_visualization(
        self,
        variants: List[VariantRecord],
        clusters: List[ClusterResult]
    ) -> plt.Figure:
        """Plot PCA visualization of variants.
        
        Args:
            variants: List of variant records
            clusters: Cluster results
            
        Returns:
            matplotlib Figure
        """
        exponent_matrix, exponent_names, variant_ids = self._build_exponent_matrix(variants)
        
        if exponent_matrix.shape[0] < 3 or exponent_matrix.shape[1] < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Insufficient data for PCA',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Perform PCA
        pca = PCA(n_components=min(3, exponent_matrix.shape[1]))
        pca_coords = pca.fit_transform(exponent_matrix)
        
        # Create variant ID to cluster mapping
        variant_to_cluster = {}
        for cluster in clusters:
            for variant_id in cluster.variant_ids:
                variant_to_cluster[variant_id] = cluster.cluster_id
        
        # Assign colors by cluster
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        
        fig = plt.figure(figsize=(14, 6))
        
        # 2D plot
        ax1 = fig.add_subplot(121)
        for i, variant_id in enumerate(variant_ids):
            cluster_id = variant_to_cluster.get(variant_id, 0)
            color = colors[cluster_id - 1] if cluster_id > 0 else 'gray'
            ax1.scatter(pca_coords[i, 0], pca_coords[i, 1], 
                       c=[color], s=100, alpha=0.7,
                       label=f'Cluster {cluster_id}' if i == 0 or cluster_id not in [
                           variant_to_cluster.get(variant_ids[j], 0) for j in range(i)
                       ] else '')
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax1.set_title('PCA Visualization (2D)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Variance explained plot
        ax2 = fig.add_subplot(122)
        ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1),
               pca.explained_variance_ratio_, alpha=0.7)
        ax2.set_xlabel('Principal Component', fontsize=12)
        ax2.set_ylabel('Variance Explained', fontsize=12)
        ax2.set_title('PCA Variance Explained', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_text_report(
        self,
        report: ComparativeAnalysisReport,
        output_file: Optional[Path] = None
    ) -> str:
        """Generate human-readable text report.
        
        Args:
            report: Comparative analysis report
            output_file: Optional file to save report
            
        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-VARIANT COMPARATIVE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append(f"Total Variants Analyzed: {report.n_variants}")
        lines.append(f"Clusters Identified: {len(report.clusters)}")
        lines.append(f"Outliers Detected: {len(report.outliers)}")
        lines.append(f"Significant Trends: {len(report.trends)}")
        lines.append("")
        
        # Clusters
        if report.clusters:
            lines.append("-" * 80)
            lines.append("CLUSTERS")
            lines.append("-" * 80)
            for cluster in report.clusters:
                lines.append(f"\nCluster {cluster.cluster_id}:")
                lines.append(f"  Size: {cluster.cluster_size} variants")
                lines.append(f"  Representative: {cluster.representative_variant}")
                lines.append(f"  Centroid Exponents:")
                for exp_name, exp_value in cluster.centroid_exponents.items():
                    lines.append(f"    {exp_name}: {exp_value:.4f}")
                lines.append(f"  Intra-cluster Variance: {cluster.intra_cluster_variance:.6f}")
                lines.append(f"  Variants: {', '.join(cluster.variant_ids)}")
        
        # Outliers
        if report.outliers:
            lines.append("")
            lines.append("-" * 80)
            lines.append("OUTLIERS")
            lines.append("-" * 80)
            for outlier in report.outliers:
                lines.append(f"\n{outlier.variant_id}:")
                lines.append(f"  Outlier Score: {outlier.outlier_score:.2f}σ")
                lines.append(f"  Reason: {outlier.reason}")
                lines.append(f"  Nearest Cluster: {outlier.nearest_cluster}")
        
        # Trends
        if report.trends:
            lines.append("")
            lines.append("-" * 80)
            lines.append("PARAMETER TRENDS")
            lines.append("-" * 80)
            for trend in report.trends:
                lines.append(f"\n{trend.description}")
                lines.append(f"  Type: {trend.trend_type}")
                lines.append(f"  R²: {trend.r_squared:.4f}")
                lines.append(f"  p-value: {trend.p_value:.4e}")
        
        # Exponent correlations
        if report.exponent_correlations:
            lines.append("")
            lines.append("-" * 80)
            lines.append("EXPONENT CORRELATIONS")
            lines.append("-" * 80)
            for pair, corr_data in report.exponent_correlations.items():
                lines.append(
                    f"{pair}: r={corr_data['correlation']:.3f}, "
                    f"p={corr_data['p_value']:.4e}"
                )
        
        lines.append("")
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Saved text report to {output_file}")
        
        return report_text
