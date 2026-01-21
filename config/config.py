"""
Configuration module for Hate Speech Detection project.

This module handles all configuration settings including model parameters,
file paths, and application settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for the hate speech detection model."""
    model_name: str = "unitary/toxic-bert"
    device: Optional[str] = None
    use_pipeline: bool = True
    batch_size: int = 32
    max_length: int = 512
    threshold: float = 0.5


@dataclass
class AppConfig:
    """Configuration for the application."""
    debug: bool = False
    log_level: str = "INFO"
    host: str = "localhost"
    port: int = 8501
    title: str = "Hate Speech Detection"
    description: str = "Detect hate speech in text using AI"


@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    synthetic_dataset_size: int = 1000


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    app: AppConfig
    data: DataConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            app=AppConfig(**config_dict.get('app', {})),
            data=DataConfig(**config_dict.get('data', {}))
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': asdict(self.model),
            'app': asdict(self.app),
            'data': asdict(self.data)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data.data_dir,
            self.data.models_dir,
            self.data.results_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config(
        model=ModelConfig(),
        app=AppConfig(),
        data=DataConfig()
    )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    else:
        return get_default_config()


# Default configuration file content
DEFAULT_CONFIG_YAML = """
# Hate Speech Detection Configuration

model:
  model_name: "unitary/toxic-bert"
  device: null  # null for auto-detection
  use_pipeline: true
  batch_size: 32
  max_length: 512
  threshold: 0.5

app:
  debug: false
  log_level: "INFO"
  host: "localhost"
  port: 8501
  title: "Hate Speech Detection"
  description: "Detect hate speech in text using AI"

data:
  data_dir: "data"
  models_dir: "models"
  results_dir: "results"
  synthetic_dataset_size: 1000
"""
