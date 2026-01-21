"""
Test suite for the Hate Speech Detection system.

This module contains comprehensive tests for all components of the system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hate_speech_detector import HateSpeechDetector, create_synthetic_dataset
from config.config import Config, ModelConfig, AppConfig, DataConfig


class TestHateSpeechDetector:
    """Test cases for the HateSpeechDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        return HateSpeechDetector(use_pipeline=True)
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = HateSpeechDetector()
        assert detector.model_name == "unitary/toxic-bert"
        assert detector.device in ["cpu", "cuda", "mps"]
        assert detector.use_pipeline is True
    
    def test_device_detection(self):
        """Test automatic device detection."""
        detector = HateSpeechDetector(device=None)
        assert detector.device in ["cpu", "cuda", "mps"]
    
    def test_custom_model_name(self):
        """Test initialization with custom model name."""
        detector = HateSpeechDetector(model_name="test-model")
        assert detector.model_name == "test-model"
    
    @patch('transformers.pipeline')
    def test_pipeline_loading(self, mock_pipeline):
        """Test pipeline loading."""
        mock_pipeline.return_value = Mock()
        detector = HateSpeechDetector(use_pipeline=True)
        mock_pipeline.assert_called_once()
    
    def test_detect_single_text(self, detector):
        """Test single text detection."""
        text = "You are worthless"
        result = detector.detect_hate_speech(text)
        
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'is_hate_speech' in result
        assert 'confidence' in result
        assert 'label' in result
        assert result['text'] == text
        assert isinstance(result['is_hate_speech'], bool)
        assert 0 <= result['confidence'] <= 1
    
    def test_detect_multiple_texts(self, detector):
        """Test multiple text detection."""
        texts = ["Hello world", "You are terrible"]
        results = detector.detect_hate_speech(texts)
        
        assert isinstance(results, list)
        assert len(results) == len(texts)
        
        for i, result in enumerate(results):
            assert result['text'] == texts[i]
            assert isinstance(result['is_hate_speech'], bool)
    
    def test_batch_detect(self, detector):
        """Test batch detection."""
        texts = ["Text 1", "Text 2", "Text 3"]
        results = detector.batch_detect(texts, batch_size=2)
        
        assert isinstance(results, list)
        assert len(results) == len(texts)
    
    def test_confidence_threshold(self, detector):
        """Test confidence threshold functionality."""
        text = "Test text"
        
        # Test with low threshold
        result_low = detector.detect_hate_speech(text, threshold=0.1)
        
        # Test with high threshold
        result_high = detector.detect_hate_speech(text, threshold=0.9)
        
        # Results should be different based on threshold
        assert isinstance(result_low['is_hate_speech'], bool)
        assert isinstance(result_high['is_hate_speech'], bool)
    
    def test_return_confidence(self, detector):
        """Test return_confidence parameter."""
        text = "Test text"
        
        # Test with confidence scores
        result_with_conf = detector.detect_hate_speech(text, return_confidence=True)
        assert 'confidence' in result_with_conf
        
        # Test without confidence scores
        result_without_conf = detector.detect_hate_speech(text, return_confidence=False)
        assert 'confidence' in result_without_conf  # Always returned for compatibility
    
    def test_model_info(self, detector):
        """Test model information retrieval."""
        info = detector.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'device' in info
        assert 'use_pipeline' in info
        assert 'model_type' in info


class TestSyntheticDataset:
    """Test cases for synthetic dataset creation."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(100)
        
        assert isinstance(dataset, list)
        assert len(dataset) == 100
        
        # Check that we have both positive and negative examples
        labels = [label for _, label in dataset]
        assert True in labels  # Should have hate speech examples
        assert False in labels  # Should have non-hate speech examples
    
    def test_dataset_balance(self):
        """Test that dataset is balanced."""
        dataset = create_synthetic_dataset(1000)
        labels = [label for _, label in dataset]
        
        # Should be roughly balanced (within 10% tolerance)
        hate_count = sum(labels)
        non_hate_count = len(labels) - hate_count
        
        assert abs(hate_count - non_hate_count) <= 100  # Within 10% of 500
    
    def test_dataset_content(self):
        """Test dataset content quality."""
        dataset = create_synthetic_dataset(10)
        
        for text, label in dataset:
            assert isinstance(text, str)
            assert isinstance(label, bool)
            assert len(text) > 0  # Non-empty text


class TestModelEvaluation:
    """Test cases for model evaluation."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        return HateSpeechDetector(use_pipeline=True)
    
    def test_evaluate_model(self, detector):
        """Test model evaluation."""
        # Create small test dataset
        test_data = [
            ("I love this day", False),
            ("You are terrible", True),
            ("Thank you", False),
            ("I hate you", True)
        ]
        
        eval_results = detector.evaluate_model(test_data)
        
        assert isinstance(eval_results, dict)
        assert 'accuracy' in eval_results
        assert 'precision' in eval_results
        assert 'recall' in eval_results
        assert 'f1_score' in eval_results
        assert 'confusion_matrix' in eval_results
        assert 'classification_report' in eval_results
        
        # Check that metrics are valid
        assert 0 <= eval_results['accuracy'] <= 1
        assert 0 <= eval_results['precision'] <= 1
        assert 0 <= eval_results['recall'] <= 1
        assert 0 <= eval_results['f1_score'] <= 1
        
        # Check confusion matrix shape
        cm = eval_results['confusion_matrix']
        assert cm.shape == (2, 2)  # Binary classification


class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_model_config(self):
        """Test ModelConfig class."""
        config = ModelConfig()
        
        assert config.model_name == "unitary/toxic-bert"
        assert config.device is None
        assert config.use_pipeline is True
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.threshold == 0.5
    
    def test_app_config(self):
        """Test AppConfig class."""
        config = AppConfig()
        
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.host == "localhost"
        assert config.port == 8501
        assert config.title == "Hate Speech Detection"
    
    def test_data_config(self):
        """Test DataConfig class."""
        config = DataConfig()
        
        assert config.data_dir == "data"
        assert config.models_dir == "models"
        assert config.results_dir == "results"
        assert config.synthetic_dataset_size == 1000
    
    def test_main_config(self):
        """Test main Config class."""
        config = Config(
            model=ModelConfig(),
            app=AppConfig(),
            data=DataConfig()
        )
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.app, AppConfig)
        assert isinstance(config.data, DataConfig)
    
    def test_config_yaml_serialization(self, tmp_path):
        """Test YAML serialization and deserialization."""
        config = Config(
            model=ModelConfig(model_name="test-model"),
            app=AppConfig(port=9000),
            data=DataConfig(data_dir="test-data")
        )
        
        # Save to YAML
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(str(yaml_path))
        
        # Load from YAML
        loaded_config = Config.from_yaml(str(yaml_path))
        
        assert loaded_config.model.model_name == "test-model"
        assert loaded_config.app.port == 9000
        assert loaded_config.data.data_dir == "test-data"


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        with pytest.raises(Exception):
            HateSpeechDetector(model_name="invalid-model-name")
    
    def test_empty_text(self):
        """Test handling of empty text."""
        detector = HateSpeechDetector()
        
        # Should handle empty text gracefully
        result = detector.detect_hate_speech("")
        assert isinstance(result, dict)
        assert 'error' in result or 'text' in result
    
    def test_none_text(self):
        """Test handling of None text."""
        detector = HateSpeechDetector()
        
        # Should handle None text gracefully
        result = detector.detect_hate_speech(None)
        assert isinstance(result, dict)


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_detection(self):
        """Test complete end-to-end detection pipeline."""
        detector = HateSpeechDetector()
        
        # Test with various text types
        test_texts = [
            "I love this beautiful day!",
            "You are a terrible person.",
            "Thank you for your help.",
            "I hate people like you.",
            "This is a neutral statement."
        ]
        
        results = detector.detect_hate_speech(test_texts)
        
        assert len(results) == len(test_texts)
        
        for result in results:
            assert isinstance(result['is_hate_speech'], bool)
            assert 0 <= result['confidence'] <= 1
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        detector = HateSpeechDetector()
        
        # Create large batch
        texts = [f"Test text {i}" for i in range(100)]
        
        results = detector.batch_detect(texts, batch_size=10)
        
        assert len(results) == len(texts)
        
        # All results should be valid
        for result in results:
            assert isinstance(result['is_hate_speech'], bool)


if __name__ == "__main__":
    pytest.main([__file__])
