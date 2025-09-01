"""
Logging utilities for the Prometheus project.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import LoggingConfig


class PrometheusLogger:
    """Centralized logging system for the Prometheus project."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrometheusLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.loggers = {}
            self.config = None
            self._initialized = True
    
    def setup_logging(self, config: LoggingConfig) -> None:
        """
        Set up the logging system with the provided configuration.
        
        Args:
            config: LoggingConfig object with logging parameters.
        """
        self.config = config
        
        # Create log directory if file output is enabled
        if config.file_output:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(config.format)
        
        # Add console handler if enabled
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, config.level))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if config.file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(config.log_dir) / f"prometheus_{timestamp}.log"
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, config.level))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Log initialization
        logger = self.get_logger("PrometheusLogger")
        logger.info("Logging system initialized")
        logger.info(f"Log level: {config.level}")
        logger.info(f"Console output: {config.console_output}")
        logger.info(f"File output: {config.file_output}")
        if config.file_output:
            logger.info(f"Log directory: {config.log_dir}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for the specified module.
        
        Args:
            name: Name of the logger (typically __name__ of the module).
            
        Returns:
            logging.Logger: Configured logger instance.
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def log_experiment_start(self, experiment_name: str, config_dict: dict) -> None:
        """
        Log the start of an experiment with configuration details.
        
        Args:
            experiment_name: Name of the experiment.
            config_dict: Configuration dictionary to log.
        """
        logger = self.get_logger("Experiment")
        logger.info("=" * 80)
        logger.info(f"EXPERIMENT START: {experiment_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("Configuration:")
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
    
    def log_experiment_end(self, experiment_name: str, results: Optional[dict] = None) -> None:
        """
        Log the end of an experiment with optional results.
        
        Args:
            experiment_name: Name of the experiment.
            results: Optional results dictionary to log.
        """
        logger = self.get_logger("Experiment")
        logger.info("=" * 80)
        logger.info(f"EXPERIMENT END: {experiment_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        if results:
            logger.info("Results:")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
    
    def log_phase_start(self, phase_name: str) -> None:
        """Log the start of a major phase (e.g., data generation, training, analysis)."""
        logger = self.get_logger("Phase")
        logger.info("-" * 60)
        logger.info(f"PHASE START: {phase_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("-" * 60)
    
    def log_phase_end(self, phase_name: str, duration: Optional[float] = None) -> None:
        """Log the end of a major phase with optional duration."""
        logger = self.get_logger("Phase")
        logger.info("-" * 60)
        logger.info(f"PHASE END: {phase_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        if duration is not None:
            logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("-" * 60)


def setup_logging(config: LoggingConfig) -> PrometheusLogger:
    """
    Convenience function to set up logging and return logger instance.
    
    Args:
        config: LoggingConfig object with logging parameters.
        
    Returns:
        PrometheusLogger: Configured logger instance.
    """
    prometheus_logger = PrometheusLogger()
    prometheus_logger.setup_logging(config)
    return prometheus_logger


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger instance.
    
    Args:
        name: Name of the logger (typically __name__ of the module).
        
    Returns:
        logging.Logger: Logger instance.
    """
    prometheus_logger = PrometheusLogger()
    return prometheus_logger.get_logger(name)


class LoggingContext:
    """Context manager for logging phases with automatic timing."""
    
    def __init__(self, phase_name: str, logger: Optional[logging.Logger] = None):
        self.phase_name = phase_name
        self.logger = logger or get_logger("LoggingContext")
        self.start_time = None
        self.prometheus_logger = PrometheusLogger()
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.prometheus_logger.log_phase_start(self.phase_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.prometheus_logger.log_phase_end(self.phase_name, duration)
        
        if exc_type is not None:
            self.logger.error(f"Exception in phase {self.phase_name}: {exc_val}")
            return False  # Re-raise the exception
        
        return True