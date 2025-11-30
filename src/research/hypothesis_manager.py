"""
Hypothesis Manager for defining, tracking, and validating research hypotheses.

This module provides functionality for creating testable hypotheses about phase
transition behavior, tracking validation status, and comparing predictions
against experimental results with statistical rigor.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base_types import ResearchHypothesis, ValidationResult


# Known universality classes with their critical exponents
UNIVERSALITY_CLASSES = {
    '2d_ising': {
        'beta': 0.125,
        'nu': 1.0,
        'gamma': 1.75,
        'alpha': 0.0,
        'delta': 15.0,
        'eta': 0.25,
    },
    '3d_ising': {
        'beta': 0.326,
        'nu': 0.630,
        'gamma': 1.237,
        'alpha': 0.110,
        'delta': 4.789,
        'eta': 0.036,
    },
    'mean_field': {
        'beta': 0.5,
        'nu': 0.5,
        'gamma': 1.0,
        'alpha': 0.0,
        'delta': 3.0,
        'eta': 0.0,
    },
    '2d_xy': {
        'beta': 0.231,
        'nu': 0.670,
        'gamma': 1.316,
        'alpha': -0.007,
        'delta': 4.780,
        'eta': 0.038,
    },
    '3d_xy': {
        'beta': 0.346,
        'nu': 0.672,
        'gamma': 1.316,
        'alpha': -0.011,
        'delta': 4.780,
        'eta': 0.038,
    },
}


class HypothesisManager:
    """Manage research hypotheses and validation.
    
    The hypothesis manager provides functionality for creating, storing, and
    validating research hypotheses about phase transition behavior. It tracks
    hypothesis status and provides statistical validation against experimental
    results.
    
    Attributes:
        storage_path: Path to JSON file for hypothesis persistence
        _hypotheses: In-memory cache of hypotheses
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the hypothesis manager.
        
        Args:
            storage_path: Path to JSON file for persistence. If None, uses
                         default path '.kiro/research/hypotheses.json'
        """
        if storage_path is None:
            storage_path = '.kiro/research/hypotheses.json'
        
        self.storage_path = Path(storage_path)
        self._hypotheses: Dict[str, ResearchHypothesis] = {}
        
        # Create directory if it doesn't exist
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing hypotheses if available
        if self.storage_path.exists():
            self._load_hypotheses()
    
    def create_hypothesis(
        self,
        description: str,
        variant_id: str,
        predictions: Dict[str, float],
        parameter_ranges: Optional[Dict[str, tuple]] = None,
        prediction_errors: Optional[Dict[str, float]] = None,
        universality_class: Optional[str] = None,
    ) -> ResearchHypothesis:
        """Create a new research hypothesis.
        
        Args:
            description: Human-readable description of the hypothesis
            variant_id: ID of the model variant this hypothesis applies to
            predictions: Predicted critical exponent values (e.g., {'beta': 0.35})
            parameter_ranges: Parameter ranges to explore (optional)
            prediction_errors: Acceptable error margins for predictions (optional)
            universality_class: Expected universality class (optional)
            
        Returns:
            ResearchHypothesis object
            
        Raises:
            ValueError: If predictions are invalid or universality class unknown
        """
        # Generate unique hypothesis ID
        hypothesis_id = f"hyp_{uuid.uuid4().hex[:8]}"
        
        # Validate predictions
        self._validate_predictions(predictions)
        
        # Validate universality class if provided
        if universality_class is not None:
            if universality_class not in UNIVERSALITY_CLASSES:
                raise ValueError(
                    f"Unknown universality class: {universality_class}. "
                    f"Known classes: {list(UNIVERSALITY_CLASSES.keys())}"
                )
        
        # Set default parameter ranges if not provided
        if parameter_ranges is None:
            parameter_ranges = {}
        
        # Set default prediction errors if not provided
        if prediction_errors is None:
            # Default to 10% error margin
            prediction_errors = {k: abs(v * 0.1) for k, v in predictions.items()}
        
        # Create hypothesis object
        hypothesis = ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            description=description,
            variant_id=variant_id,
            parameter_ranges=parameter_ranges,
            predictions=predictions,
            prediction_errors=prediction_errors,
            universality_class=universality_class,
            status='pending',
            confidence=0.0,
            validation_results=None,
        )
        
        # Store hypothesis
        self._hypotheses[hypothesis_id] = hypothesis
        self._save_hypotheses()
        
        return hypothesis
    
    def validate_hypothesis(
        self,
        hypothesis_id: str,
        experimental_results: Dict[str, float],
        experimental_errors: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """Validate hypothesis against experimental results.
        
        Performs statistical validation by comparing predicted critical exponents
        against measured values. Uses z-tests to determine if measured values
        fall within acceptable ranges of predictions.
        
        Args:
            hypothesis_id: ID of the hypothesis to validate
            experimental_results: Measured critical exponent values
            experimental_errors: Error estimates for measurements (optional)
            
        Returns:
            ValidationResult object with validation outcome
            
        Raises:
            KeyError: If hypothesis_id not found
            ValueError: If experimental results don't match predictions
        """
        # Get hypothesis
        if hypothesis_id not in self._hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found")
        
        hypothesis = self._hypotheses[hypothesis_id]
        
        # Check that experimental results contain all predicted exponents
        missing_exponents = set(hypothesis.predictions.keys()) - set(experimental_results.keys())
        if missing_exponents:
            raise ValueError(
                f"Experimental results missing exponents: {missing_exponents}"
            )
        
        # Set default experimental errors if not provided
        if experimental_errors is None:
            experimental_errors = {k: abs(v * 0.05) for k, v in experimental_results.items()}
        
        # Perform validation for each exponent
        p_values = {}
        effect_sizes = {}
        bootstrap_intervals = {}
        validated_exponents = []
        
        for exponent_name, predicted_value in hypothesis.predictions.items():
            measured_value = experimental_results[exponent_name]
            prediction_error = hypothesis.prediction_errors.get(exponent_name, abs(predicted_value * 0.1))
            measurement_error = experimental_errors.get(exponent_name, abs(measured_value * 0.05))
            
            # Calculate z-score for hypothesis test
            # H0: measured_value = predicted_value
            # Combined error from prediction and measurement
            combined_error = (prediction_error**2 + measurement_error**2)**0.5
            
            if combined_error > 0:
                z_score = abs(measured_value - predicted_value) / combined_error
                # Two-tailed p-value (approximate using normal distribution)
                import scipy.stats as stats
                p_value = 2 * (1 - stats.norm.cdf(z_score))
            else:
                # If no error, check exact match
                p_value = 1.0 if measured_value == predicted_value else 0.0
            
            p_values[exponent_name] = p_value
            
            # Calculate effect size (Cohen's d)
            if combined_error > 0:
                effect_size = abs(measured_value - predicted_value) / combined_error
            else:
                effect_size = 0.0 if measured_value == predicted_value else float('inf')
            
            effect_sizes[exponent_name] = effect_size
            
            # Bootstrap confidence interval (approximate as measured ± 2*error)
            bootstrap_intervals[exponent_name] = (
                measured_value - 2 * measurement_error,
                measured_value + 2 * measurement_error
            )
            
            # Check if prediction is validated (p-value > 0.05 means we don't reject H0)
            if p_value > 0.05:
                validated_exponents.append(exponent_name)
        
        # Overall validation: all exponents must be validated
        validated = len(validated_exponents) == len(hypothesis.predictions)
        
        # Calculate overall confidence (average of p-values)
        confidence = sum(p_values.values()) / len(p_values) if p_values else 0.0
        
        # Generate validation message
        if validated:
            message = (
                f"Hypothesis validated: All {len(validated_exponents)} predicted "
                f"exponents agree with measurements (p > 0.05)"
            )
            new_status = 'validated'
        else:
            failed_exponents = set(hypothesis.predictions.keys()) - set(validated_exponents)
            message = (
                f"Hypothesis refuted: {len(failed_exponents)} exponent(s) "
                f"disagree with measurements: {failed_exponents}"
            )
            new_status = 'refuted'
        
        # Create validation result
        result = ValidationResult(
            hypothesis_id=hypothesis_id,
            validated=validated,
            confidence=confidence,
            p_values=p_values,
            effect_sizes=effect_sizes,
            bootstrap_intervals=bootstrap_intervals,
            message=message,
        )
        
        # Update hypothesis status
        hypothesis.status = new_status
        hypothesis.confidence = confidence
        hypothesis.validation_results = {
            'experimental_results': experimental_results,
            'experimental_errors': experimental_errors,
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'validated_exponents': validated_exponents,
        }
        
        self._save_hypotheses()
        
        return result

    
    def get_hypothesis_status(self, hypothesis_id: str) -> Dict[str, Any]:
        """Get current status and validation metrics for a hypothesis.
        
        Args:
            hypothesis_id: ID of the hypothesis
            
        Returns:
            Dictionary with status information
            
        Raises:
            KeyError: If hypothesis_id not found
        """
        if hypothesis_id not in self._hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found")
        
        hypothesis = self._hypotheses[hypothesis_id]
        
        return {
            'hypothesis_id': hypothesis.hypothesis_id,
            'description': hypothesis.description,
            'variant_id': hypothesis.variant_id,
            'status': hypothesis.status,
            'confidence': hypothesis.confidence,
            'predictions': hypothesis.predictions,
            'universality_class': hypothesis.universality_class,
            'validation_results': hypothesis.validation_results,
        }
    
    def list_hypotheses(
        self,
        status: Optional[str] = None,
        variant_id: Optional[str] = None,
    ) -> List[ResearchHypothesis]:
        """List hypotheses with optional filtering.
        
        Args:
            status: Filter by status ('pending', 'validated', 'refuted', 'inconclusive')
            variant_id: Filter by model variant ID
            
        Returns:
            List of ResearchHypothesis objects matching filters
        """
        hypotheses = list(self._hypotheses.values())
        
        # Apply status filter
        if status is not None:
            hypotheses = [h for h in hypotheses if h.status == status]
        
        # Apply variant_id filter
        if variant_id is not None:
            hypotheses = [h for h in hypotheses if h.variant_id == variant_id]
        
        return hypotheses
    
    def compare_with_universality_class(
        self,
        hypothesis_id: str,
        class_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare hypothesis predictions with known universality class.
        
        Args:
            hypothesis_id: ID of the hypothesis
            class_name: Name of universality class to compare with. If None,
                       uses the hypothesis's universality_class attribute.
            
        Returns:
            Dictionary with comparison results
            
        Raises:
            KeyError: If hypothesis_id not found
            ValueError: If universality class not specified or unknown
        """
        if hypothesis_id not in self._hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found")
        
        hypothesis = self._hypotheses[hypothesis_id]
        
        # Determine which universality class to compare with
        if class_name is None:
            class_name = hypothesis.universality_class
        
        if class_name is None:
            raise ValueError(
                "No universality class specified in hypothesis or as argument"
            )
        
        if class_name not in UNIVERSALITY_CLASSES:
            raise ValueError(
                f"Unknown universality class: {class_name}. "
                f"Known classes: {list(UNIVERSALITY_CLASSES.keys())}"
            )
        
        # Get universality class exponents
        class_exponents = UNIVERSALITY_CLASSES[class_name]
        
        # Compare each predicted exponent with class value
        comparisons = {}
        agreements = []
        
        for exponent_name, predicted_value in hypothesis.predictions.items():
            if exponent_name not in class_exponents:
                comparisons[exponent_name] = {
                    'predicted': predicted_value,
                    'class_value': None,
                    'difference': None,
                    'agrees': None,
                    'note': f'Exponent not defined for {class_name} class',
                }
                continue
            
            class_value = class_exponents[exponent_name]
            difference = abs(predicted_value - class_value)
            prediction_error = hypothesis.prediction_errors.get(
                exponent_name, abs(predicted_value * 0.1)
            )
            
            # Check if within error margin
            agrees = difference <= prediction_error
            agreements.append(agrees)
            
            comparisons[exponent_name] = {
                'predicted': predicted_value,
                'class_value': class_value,
                'difference': difference,
                'relative_difference': difference / abs(class_value) if class_value != 0 else float('inf'),
                'prediction_error': prediction_error,
                'agrees': agrees,
            }
        
        # Overall agreement
        overall_agreement = all(agreements) if agreements else False
        agreement_fraction = sum(agreements) / len(agreements) if agreements else 0.0
        
        return {
            'hypothesis_id': hypothesis_id,
            'universality_class': class_name,
            'comparisons': comparisons,
            'overall_agreement': overall_agreement,
            'agreement_fraction': agreement_fraction,
            'message': (
                f"Predictions {'agree' if overall_agreement else 'disagree'} "
                f"with {class_name} universality class "
                f"({agreement_fraction:.1%} of exponents match)"
            ),
        }
    
    def get_universality_classes(self) -> Dict[str, Dict[str, float]]:
        """Get database of known universality classes.
        
        Returns:
            Dictionary mapping class names to their critical exponents
        """
        return UNIVERSALITY_CLASSES.copy()
    
    def remove_hypothesis(self, hypothesis_id: str) -> None:
        """Remove a hypothesis from storage.
        
        Args:
            hypothesis_id: ID of hypothesis to remove
            
        Raises:
            KeyError: If hypothesis_id not found
        """
        if hypothesis_id not in self._hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found")
        
        del self._hypotheses[hypothesis_id]
        self._save_hypotheses()
    
    def clear_hypotheses(self) -> None:
        """Clear all hypotheses from storage."""
        self._hypotheses.clear()
        self._save_hypotheses()
    
    def _validate_predictions(self, predictions: Dict[str, float]) -> None:
        """Validate prediction dictionary.
        
        Args:
            predictions: Dictionary of predicted exponent values
            
        Raises:
            ValueError: If predictions are invalid
        """
        if not predictions:
            raise ValueError("Predictions dictionary cannot be empty")
        
        # Check for valid exponent names
        valid_exponents = {'beta', 'nu', 'gamma', 'alpha', 'delta', 'eta'}
        invalid_exponents = set(predictions.keys()) - valid_exponents
        if invalid_exponents:
            raise ValueError(
                f"Invalid exponent names: {invalid_exponents}. "
                f"Valid names: {valid_exponents}"
            )
        
        # Check for reasonable values
        for name, value in predictions.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Exponent '{name}' must be numeric, got {type(value)}"
                )
            
            # Basic sanity checks (exponents should be in reasonable ranges)
            if name in ['beta', 'nu', 'gamma', 'delta', 'eta']:
                if value < -1.0 or value > 20.0:
                    raise ValueError(
                        f"Exponent '{name}' = {value} is outside reasonable range [-1, 20]"
                    )
    
    def _save_hypotheses(self) -> None:
        """Save hypotheses to JSON file."""
        data = {}
        for hypothesis_id, hypothesis in self._hypotheses.items():
            # Convert hypothesis to dict
            hypothesis_dict = {
                'hypothesis_id': hypothesis.hypothesis_id,
                'description': hypothesis.description,
                'variant_id': hypothesis.variant_id,
                'parameter_ranges': hypothesis.parameter_ranges,
                'predictions': hypothesis.predictions,
                'prediction_errors': hypothesis.prediction_errors,
                'universality_class': hypothesis.universality_class,
                'status': hypothesis.status,
                'confidence': hypothesis.confidence,
                'validation_results': hypothesis.validation_results,
            }
            data[hypothesis_id] = hypothesis_dict
        
        # Write to file
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_hypotheses(self) -> None:
        """Load hypotheses from JSON file."""
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        # Convert dict entries to ResearchHypothesis objects
        for hypothesis_id, hypothesis_dict in data.items():
            # Convert parameter_ranges tuples (stored as lists in JSON)
            parameter_ranges = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in hypothesis_dict.get('parameter_ranges', {}).items()
            }
            hypothesis_dict['parameter_ranges'] = parameter_ranges
            
            hypothesis = ResearchHypothesis(**hypothesis_dict)
            self._hypotheses[hypothesis_id] = hypothesis
