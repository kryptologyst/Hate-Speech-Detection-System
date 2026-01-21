"""
Hate Speech Detection Module

This module provides a comprehensive hate speech detection system using
state-of-the-art transformer models and various detection techniques.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HateSpeechDetector:
    """
    A comprehensive hate speech detection system using transformer models.
    
    This class provides multiple detection methods including:
    - Pre-trained pipeline models
    - Custom fine-tuned models
    - Zero-shot classification
    - Confidence scoring
    """
    
    def __init__(
        self, 
        model_name: str = "unitary/toxic-bert",
        device: Optional[str] = None,
        use_pipeline: bool = True
    ):
        """
        Initialize the hate speech detector.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
            use_pipeline: Whether to use transformers pipeline or custom implementation
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_pipeline = use_pipeline
        
        logger.info(f"Initializing HateSpeechDetector with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        if use_pipeline:
            self._load_pipeline()
        else:
            self._load_custom_model()
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best available device."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_pipeline(self) -> None:
        """Load the transformers pipeline."""
        try:
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _load_custom_model(self) -> None:
        """Load custom tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            logger.info("Custom model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise
    
    def detect_hate_speech(
        self, 
        text: Union[str, List[str]], 
        return_confidence: bool = True,
        threshold: float = 0.5
    ) -> Union[Dict, List[Dict]]:
        """
        Detect hate speech in the given text(s).
        
        Args:
            text: Input text or list of texts to analyze
            return_confidence: Whether to return confidence scores
            threshold: Confidence threshold for classification
            
        Returns:
            Dictionary or list of dictionaries containing classification results
        """
        if isinstance(text, str):
            return self._detect_single(text, return_confidence, threshold)
        else:
            return [self._detect_single(t, return_confidence, threshold) for t in text]
    
    def _detect_single(
        self, 
        text: str, 
        return_confidence: bool, 
        threshold: float
    ) -> Dict:
        """Detect hate speech in a single text."""
        if self.use_pipeline:
            return self._detect_with_pipeline(text, return_confidence, threshold)
        else:
            return self._detect_with_custom_model(text, return_confidence, threshold)
    
    def _detect_with_pipeline(
        self, 
        text: str, 
        return_confidence: bool, 
        threshold: float
    ) -> Dict:
        """Detect hate speech using transformers pipeline."""
        try:
            results = self.pipeline(text)
            
            # Find the highest confidence prediction
            best_result = max(results, key=lambda x: x['score'])
            
            is_hate_speech = best_result['label'] == 'LABEL_1' and best_result['score'] > threshold
            
            result = {
                'text': text,
                'is_hate_speech': is_hate_speech,
                'label': best_result['label'],
                'confidence': best_result['score']
            }
            
            if return_confidence:
                result['all_scores'] = results
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline detection: {e}")
            return {
                'text': text,
                'is_hate_speech': False,
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _detect_with_custom_model(
        self, 
        text: str, 
        return_confidence: bool, 
        threshold: float
    ) -> Dict:
        """Detect hate speech using custom model implementation."""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1)
                confidence = torch.max(probabilities, dim=-1)[0]
            
            is_hate_speech = predicted_class.item() == 1 and confidence.item() > threshold
            
            result = {
                'text': text,
                'is_hate_speech': is_hate_speech,
                'label': f'LABEL_{predicted_class.item()}',
                'confidence': confidence.item()
            }
            
            if return_confidence:
                result['probabilities'] = probabilities.cpu().numpy().tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in custom model detection: {e}")
            return {
                'text': text,
                'is_hate_speech': False,
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def batch_detect(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        return_confidence: bool = True,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect hate speech in a batch of texts efficiently.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            return_confidence: Whether to return confidence scores
            threshold: Confidence threshold for classification
            
        Returns:
            List of dictionaries containing classification results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.detect_hate_speech(
                batch, 
                return_confidence=return_confidence,
                threshold=threshold
            )
            results.extend(batch_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return results
    
    def evaluate_model(
        self, 
        test_data: List[Tuple[str, bool]], 
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate the model performance on test data.
        
        Args:
            test_data: List of (text, true_label) tuples
            threshold: Confidence threshold for classification
            
        Returns:
            Dictionary containing evaluation metrics
        """
        texts, true_labels = zip(*test_data)
        predictions = self.detect_hate_speech(list(texts), threshold=threshold)
        
        pred_labels = [p['is_hate_speech'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        return {
            'classification_report': report,
            'confusion_matrix': confusion_matrix(true_labels, pred_labels),
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confidences': confidences
        }
    
    def visualize_results(
        self, 
        results: List[Dict], 
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualizations of detection results.
        
        Args:
            results: List of detection results
            save_path: Optional path to save the plot
        """
        # Extract data for visualization
        confidences = [r['confidence'] for r in results]
        labels = [r['is_hate_speech'] for r in results]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence distribution
        axes[0].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Confidence Scores')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0].legend()
        
        # Label distribution
        label_counts = pd.Series(labels).value_counts()
        axes[1].pie(label_counts.values, labels=['Non-Hate', 'Hate Speech'], 
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Distribution of Predictions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'use_pipeline': self.use_pipeline,
            'model_type': 'transformer'
        }


def create_synthetic_dataset(size: int = 1000) -> List[Tuple[str, bool]]:
    """
    Create a synthetic dataset for testing purposes.
    
    Args:
        size: Number of samples to generate
        
    Returns:
        List of (text, is_hate_speech) tuples
    """
    np.random.seed(42)
    
    # Non-hate speech examples
    non_hate_examples = [
        "I love this beautiful sunny day!",
        "Thank you for your help with the project.",
        "The movie was really entertaining and well-made.",
        "I enjoyed reading this book very much.",
        "This is a great place to visit with family.",
        "The food at this restaurant is delicious.",
        "I'm excited about the upcoming vacation.",
        "The concert was amazing and the music was perfect.",
        "I appreciate your thoughtful comments.",
        "This is an interesting topic to discuss.",
        "The weather is perfect for a walk in the park.",
        "I'm grateful for all the support I've received.",
        "This is a wonderful opportunity to learn something new.",
        "The presentation was clear and informative.",
        "I'm looking forward to the weekend.",
        "The team did an excellent job on this project.",
        "This is a beautiful piece of art.",
        "I'm happy to help with this task.",
        "The new software update looks promising.",
        "I'm proud of my accomplishments this year."
    ]
    
    # Hate speech examples (synthetic but realistic patterns)
    hate_examples = [
        "You are worthless and should disappear forever.",
        "I hate people like you, you're disgusting.",
        "You're a pathetic excuse for a human being.",
        "People like you don't deserve to exist.",
        "You're nothing but trash and should be treated as such.",
        "I wish you would just go away and never come back.",
        "You're a complete failure and always will be.",
        "Nobody likes you and nobody ever will.",
        "You're the worst person I've ever met.",
        "I despise everything about you and your kind.",
        "You're a waste of space and oxygen.",
        "People like you make me sick to my stomach.",
        "You're a terrible person and you know it.",
        "I can't stand being around people like you.",
        "You're a disappointment to everyone who knows you.",
        "You're ugly inside and out, just like your personality.",
        "I hate everything you stand for and represent.",
        "You're a burden to society and should be removed.",
        "You're a failure at everything you try to do.",
        "I wish I could erase you from my memory completely."
    ]
    
    # Generate balanced dataset
    dataset = []
    
    # Add non-hate examples
    for _ in range(size // 2):
        example = np.random.choice(non_hate_examples)
        dataset.append((example, False))
    
    # Add hate examples
    for _ in range(size // 2):
        example = np.random.choice(hate_examples)
        dataset.append((example, True))
    
    # Shuffle the dataset
    np.random.shuffle(dataset)
    
    return dataset


if __name__ == "__main__":
    # Example usage
    detector = HateSpeechDetector()
    
    # Test with single text
    test_text = "You are a worthless person and should not be here."
    result = detector.detect_hate_speech(test_text)
    print(f"Text: {test_text}")
    print(f"Result: {result}")
    
    # Test with multiple texts
    test_texts = [
        "I love this beautiful day!",
        "You're a terrible person.",
        "Thank you for your help.",
        "I hate people like you."
    ]
    
    results = detector.detect_hate_speech(test_texts)
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {result['text']}")
        print(f"Result: {result}")
    
    # Create and evaluate on synthetic data
    print("\n" + "="*50)
    print("EVALUATION ON SYNTHETIC DATA")
    print("="*50)
    
    synthetic_data = create_synthetic_dataset(100)
    eval_results = detector.evaluate_model(synthetic_data)
    
    print(f"Accuracy: {eval_results['accuracy']:.3f}")
    print(f"Precision: {eval_results['precision']:.3f}")
    print(f"Recall: {eval_results['recall']:.3f}")
    print(f"F1-Score: {eval_results['f1_score']:.3f}")
