# -*- coding: utf-8 -*-
"""
Variant Database for Multi-Variant Comparative Analysis.

This module provides a database for storing and querying information about
all explored Ising model variants, including measured properties, theoretical
predictions, and literature references.

Requirement 6.1: THE Discovery_Campaign SHALL maintain a database of all 
explored variants and their properties.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

from .base_types import VAEAnalysisResults, ModelVariantConfig
from ..utils.logging_utils import get_logger


@dataclass
class VariantRecord:
    """Complete record for a single variant exploration.
    
    Attributes:
        variant_id: Unique identifier for the variant
        variant_config: Model configuration
        exploration_date: When the variant was explored
        n_parameter_points: Number of parameter points explored
        measured_exponents: Measured critical exponents (mean across points)
        exponent_errors: Errors on measured exponents
        measured_tc: Measured critical temperature (mean)
        tc_error: Error on Tc
        theoretical_predictions: Theoretical predictions (if available)
        literature_references: Literature citations
        universality_class: Identified universality class (if any)
        class_confidence: Confidence in universality class assignment
        validation_confidence: Overall validation confidence
        novel_physics_detected: Whether novel physics was detected
        notes: Additional notes or observations
    """
    variant_id: str
    variant_config: ModelVariantConfig
    exploration_date: datetime
    n_parameter_points: int
    measured_exponents: Dict[str, float]
    exponent_errors: Dict[str, float]
    measured_tc: float
    tc_error: float
    theoretical_predictions: Optional[Dict[str, float]] = None
    literature_references: List[str] = field(default_factory=list)
    universality_class: Optional[str] = None
    class_confidence: float = 0.0
    validation_confidence: float = 0.0
    novel_physics_detected: bool = False
    notes: str = ""


class VariantDatabase:
    """Database for storing and querying variant exploration results.
    
    This class provides a centralized database for all explored variants,
    enabling comparative analysis, trend detection, and pattern identification.
    
    Attributes:
        records: Dictionary mapping variant IDs to records
        database_file: Path to database file
        logger: Logger instance
    """
    
    def __init__(self, database_file: Optional[Path] = None):
        """Initialize variant database.
        
        Args:
            database_file: Path to database file (default: results/variant_database.json)
        """
        self.logger = get_logger(__name__)
        self.records: Dict[str, VariantRecord] = {}
        self.database_file = database_file or Path('results/variant_database.json')
        
        # Load existing database if it exists
        if self.database_file.exists():
            self.load()
        
        self.logger.info(f"Initialized VariantDatabase with {len(self.records)} records")
    
    def add_variant(
        self,
        variant_id: str,
        variant_config: ModelVariantConfig,
        vae_results: List[VAEAnalysisResults],
        theoretical_predictions: Optional[Dict[str, float]] = None,
        literature_references: Optional[List[str]] = None,
        universality_class: Optional[str] = None,
        class_confidence: float = 0.0,
        validation_confidence: float = 0.0,
        novel_physics_detected: bool = False,
        notes: str = ""
    ) -> VariantRecord:
        """Add a variant to the database.
        
        Args:
            variant_id: Unique identifier for the variant
            variant_config: Model configuration
            vae_results: List of VAE analysis results
            theoretical_predictions: Theoretical predictions (if available)
            literature_references: Literature citations
            universality_class: Identified universality class
            class_confidence: Confidence in class assignment
            validation_confidence: Overall validation confidence
            novel_physics_detected: Whether novel physics was detected
            notes: Additional notes
            
        Returns:
            VariantRecord that was added
        """
        # Compute mean exponents across all results
        measured_exponents = self._compute_mean_exponents(vae_results)
        exponent_errors = self._compute_exponent_errors(vae_results)
        
        # Compute mean Tc
        tc_values = [r.critical_temperature for r in vae_results]
        measured_tc = float(np.mean(tc_values))
        tc_error = float(np.std(tc_values))
        
        # Create record
        record = VariantRecord(
            variant_id=variant_id,
            variant_config=variant_config,
            exploration_date=datetime.now(),
            n_parameter_points=len(vae_results),
            measured_exponents=measured_exponents,
            exponent_errors=exponent_errors,
            measured_tc=measured_tc,
            tc_error=tc_error,
            theoretical_predictions=theoretical_predictions,
            literature_references=literature_references or [],
            universality_class=universality_class,
            class_confidence=class_confidence,
            validation_confidence=validation_confidence,
            novel_physics_detected=novel_physics_detected,
            notes=notes
        )
        
        # Add to database
        self.records[variant_id] = record
        
        self.logger.info(
            f"Added variant '{variant_id}' to database "
            f"({len(vae_results)} points, exponents: {list(measured_exponents.keys())})"
        )
        
        return record
    
    def get_variant(self, variant_id: str) -> Optional[VariantRecord]:
        """Get a variant record by ID.
        
        Args:
            variant_id: Variant identifier
            
        Returns:
            VariantRecord if found, None otherwise
        """
        return self.records.get(variant_id)
    
    def get_all_variants(self) -> List[VariantRecord]:
        """Get all variant records.
        
        Returns:
            List of all variant records
        """
        return list(self.records.values())
    
    def query_by_universality_class(self, class_name: str) -> List[VariantRecord]:
        """Query variants by universality class.
        
        Args:
            class_name: Name of universality class
            
        Returns:
            List of variants in that class
        """
        return [
            record for record in self.records.values()
            if record.universality_class == class_name
        ]
    
    def query_by_dimensions(self, dimensions: int) -> List[VariantRecord]:
        """Query variants by spatial dimensions.
        
        Args:
            dimensions: Spatial dimensions (2 or 3)
            
        Returns:
            List of variants with those dimensions
        """
        return [
            record for record in self.records.values()
            if record.variant_config.dimensions == dimensions
        ]
    
    def query_by_interaction_type(self, interaction_type: str) -> List[VariantRecord]:
        """Query variants by interaction type.
        
        Args:
            interaction_type: Type of interaction
            
        Returns:
            List of variants with that interaction type
        """
        return [
            record for record in self.records.values()
            if record.variant_config.interaction_type == interaction_type
        ]
    
    def query_novel_physics(self) -> List[VariantRecord]:
        """Query variants where novel physics was detected.
        
        Returns:
            List of variants with novel physics
        """
        return [
            record for record in self.records.values()
            if record.novel_physics_detected
        ]
    
    def query_by_exponent_range(
        self,
        exponent_name: str,
        min_value: float,
        max_value: float
    ) -> List[VariantRecord]:
        """Query variants by exponent value range.
        
        Args:
            exponent_name: Name of exponent (e.g., 'beta', 'nu')
            min_value: Minimum value
            max_value: Maximum value
            
        Returns:
            List of variants with exponent in range
        """
        results = []
        for record in self.records.values():
            if exponent_name in record.measured_exponents:
                value = record.measured_exponents[exponent_name]
                if min_value <= value <= max_value:
                    results.append(record)
        return results
    
    def get_exponent_statistics(self, exponent_name: str) -> Dict[str, float]:
        """Get statistics for an exponent across all variants.
        
        Args:
            exponent_name: Name of exponent
            
        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        values = []
        for record in self.records.values():
            if exponent_name in record.measured_exponents:
                values.append(record.measured_exponents[exponent_name])
        
        if not values:
            return {}
        
        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'n_variants': len(values)
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the entire database.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.records:
            return {
                'n_variants': 0,
                'n_novel_physics': 0,
                'dimensions': {},
                'interaction_types': {},
                'universality_classes': {}
            }
        
        # Count by dimensions
        dimensions_count = {}
        for record in self.records.values():
            dim = record.variant_config.dimensions
            dimensions_count[dim] = dimensions_count.get(dim, 0) + 1
        
        # Count by interaction type
        interaction_count = {}
        for record in self.records.values():
            itype = record.variant_config.interaction_type
            interaction_count[itype] = interaction_count.get(itype, 0) + 1
        
        # Count by universality class
        class_count = {}
        for record in self.records.values():
            if record.universality_class:
                uclass = record.universality_class
                class_count[uclass] = class_count.get(uclass, 0) + 1
        
        # Count novel physics
        n_novel = sum(1 for r in self.records.values() if r.novel_physics_detected)
        
        # Get all exponent names
        all_exponents = set()
        for record in self.records.values():
            all_exponents.update(record.measured_exponents.keys())
        
        # Compute statistics for each exponent
        exponent_stats = {}
        for exp_name in all_exponents:
            exponent_stats[exp_name] = self.get_exponent_statistics(exp_name)
        
        return {
            'n_variants': len(self.records),
            'n_novel_physics': n_novel,
            'dimensions': dimensions_count,
            'interaction_types': interaction_count,
            'universality_classes': class_count,
            'exponent_statistics': exponent_stats,
            'avg_validation_confidence': float(np.mean([
                r.validation_confidence for r in self.records.values()
            ])),
            'avg_parameter_points': float(np.mean([
                r.n_parameter_points for r in self.records.values()
            ]))
        }
    
    def save(self, filepath: Optional[Path] = None) -> None:
        """Save database to file.
        
        Args:
            filepath: Path to save to (default: self.database_file)
        """
        filepath = filepath or self.database_file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert records to serializable format
        data = {
            'records': {
                variant_id: self._record_to_dict(record)
                for variant_id, record in self.records.items()
            },
            'metadata': {
                'n_variants': len(self.records),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved database with {len(self.records)} records to {filepath}")
    
    def load(self, filepath: Optional[Path] = None) -> None:
        """Load database from file.
        
        Args:
            filepath: Path to load from (default: self.database_file)
        """
        filepath = filepath or self.database_file
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load records
            self.records = {}
            for variant_id, record_dict in data.get('records', {}).items():
                self.records[variant_id] = self._dict_to_record(record_dict)
            
            self.logger.info(f"Loaded database with {len(self.records)} records from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
            self.records = {}
    
    def export_to_csv(self, filepath: Path) -> None:
        """Export database to CSV format.
        
        Args:
            filepath: Path to CSV file
        """
        import csv
        
        if not self.records:
            self.logger.warning("No records to export")
            return
        
        # Get all exponent names
        all_exponents = set()
        for record in self.records.values():
            all_exponents.update(record.measured_exponents.keys())
        all_exponents = sorted(all_exponents)
        
        # Create header
        header = [
            'variant_id', 'dimensions', 'lattice_geometry', 'interaction_type',
            'n_parameter_points', 'measured_tc', 'tc_error',
            'universality_class', 'class_confidence', 'validation_confidence',
            'novel_physics_detected'
        ]
        
        # Add exponent columns
        for exp_name in all_exponents:
            header.append(f'{exp_name}_measured')
            header.append(f'{exp_name}_error')
            if any(r.theoretical_predictions and exp_name in r.theoretical_predictions 
                   for r in self.records.values()):
                header.append(f'{exp_name}_theoretical')
        
        # Write CSV
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            
            for record in self.records.values():
                row = {
                    'variant_id': record.variant_id,
                    'dimensions': record.variant_config.dimensions,
                    'lattice_geometry': record.variant_config.lattice_geometry,
                    'interaction_type': record.variant_config.interaction_type,
                    'n_parameter_points': record.n_parameter_points,
                    'measured_tc': record.measured_tc,
                    'tc_error': record.tc_error,
                    'universality_class': record.universality_class or '',
                    'class_confidence': record.class_confidence,
                    'validation_confidence': record.validation_confidence,
                    'novel_physics_detected': record.novel_physics_detected
                }
                
                # Add exponent values
                for exp_name in all_exponents:
                    if exp_name in record.measured_exponents:
                        row[f'{exp_name}_measured'] = record.measured_exponents[exp_name]
                        row[f'{exp_name}_error'] = record.exponent_errors.get(exp_name, 0.0)
                        if record.theoretical_predictions and exp_name in record.theoretical_predictions:
                            row[f'{exp_name}_theoretical'] = record.theoretical_predictions[exp_name]
                
                writer.writerow(row)
        
        self.logger.info(f"Exported database to CSV: {filepath}")
    
    def _compute_mean_exponents(
        self,
        vae_results: List[VAEAnalysisResults]
    ) -> Dict[str, float]:
        """Compute mean exponents across results.
        
        Args:
            vae_results: List of VAE analysis results
            
        Returns:
            Dictionary of mean exponents
        """
        exponent_values = {}
        
        for result in vae_results:
            for exp_name, exp_value in result.exponents.items():
                if exp_name not in exponent_values:
                    exponent_values[exp_name] = []
                exponent_values[exp_name].append(exp_value)
        
        return {
            name: float(np.mean(values))
            for name, values in exponent_values.items()
        }
    
    def _compute_exponent_errors(
        self,
        vae_results: List[VAEAnalysisResults]
    ) -> Dict[str, float]:
        """Compute exponent errors across results.
        
        Args:
            vae_results: List of VAE analysis results
            
        Returns:
            Dictionary of exponent errors (standard error of mean)
        """
        exponent_values = {}
        
        for result in vae_results:
            for exp_name, exp_value in result.exponents.items():
                if exp_name not in exponent_values:
                    exponent_values[exp_name] = []
                exponent_values[exp_name].append(exp_value)
        
        return {
            name: float(np.std(values) / np.sqrt(len(values)))
            for name, values in exponent_values.items()
        }
    
    def _record_to_dict(self, record: VariantRecord) -> Dict[str, Any]:
        """Convert VariantRecord to dictionary for serialization.
        
        Args:
            record: VariantRecord to convert
            
        Returns:
            Dictionary representation
        """
        record_dict = {
            'variant_id': record.variant_id,
            'variant_config': {
                'name': record.variant_config.name,
                'dimensions': record.variant_config.dimensions,
                'lattice_geometry': record.variant_config.lattice_geometry,
                'interaction_type': record.variant_config.interaction_type,
                'interaction_params': record.variant_config.interaction_params,
                'disorder_type': record.variant_config.disorder_type,
                'disorder_strength': record.variant_config.disorder_strength,
                'external_field': record.variant_config.external_field,
                'theoretical_tc': record.variant_config.theoretical_tc,
                'theoretical_exponents': record.variant_config.theoretical_exponents
            },
            'exploration_date': record.exploration_date.isoformat(),
            'n_parameter_points': record.n_parameter_points,
            'measured_exponents': record.measured_exponents,
            'exponent_errors': record.exponent_errors,
            'measured_tc': record.measured_tc,
            'tc_error': record.tc_error,
            'theoretical_predictions': record.theoretical_predictions,
            'literature_references': record.literature_references,
            'universality_class': record.universality_class,
            'class_confidence': record.class_confidence,
            'validation_confidence': record.validation_confidence,
            'novel_physics_detected': record.novel_physics_detected,
            'notes': record.notes
        }
        
        return record_dict
    
    def _dict_to_record(self, record_dict: Dict[str, Any]) -> VariantRecord:
        """Convert dictionary to VariantRecord.
        
        Args:
            record_dict: Dictionary representation
            
        Returns:
            VariantRecord
        """
        # Reconstruct ModelVariantConfig
        config_dict = record_dict['variant_config']
        variant_config = ModelVariantConfig(
            name=config_dict['name'],
            dimensions=config_dict['dimensions'],
            lattice_geometry=config_dict['lattice_geometry'],
            interaction_type=config_dict['interaction_type'],
            interaction_params=config_dict['interaction_params'],
            disorder_type=config_dict.get('disorder_type'),
            disorder_strength=config_dict.get('disorder_strength', 0.0),
            external_field=config_dict.get('external_field', 0.0),
            theoretical_tc=config_dict.get('theoretical_tc'),
            theoretical_exponents=config_dict.get('theoretical_exponents')
        )
        
        # Reconstruct VariantRecord
        record = VariantRecord(
            variant_id=record_dict['variant_id'],
            variant_config=variant_config,
            exploration_date=datetime.fromisoformat(record_dict['exploration_date']),
            n_parameter_points=record_dict['n_parameter_points'],
            measured_exponents=record_dict['measured_exponents'],
            exponent_errors=record_dict['exponent_errors'],
            measured_tc=record_dict['measured_tc'],
            tc_error=record_dict['tc_error'],
            theoretical_predictions=record_dict.get('theoretical_predictions'),
            literature_references=record_dict.get('literature_references', []),
            universality_class=record_dict.get('universality_class'),
            class_confidence=record_dict.get('class_confidence', 0.0),
            validation_confidence=record_dict.get('validation_confidence', 0.0),
            novel_physics_detected=record_dict.get('novel_physics_detected', False),
            notes=record_dict.get('notes', '')
        )
        
        return record
