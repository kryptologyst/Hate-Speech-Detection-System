"""
Command Line Interface for Hate Speech Detection

This module provides a command-line interface for the hate speech detection system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hate_speech_detector import HateSpeechDetector, create_synthetic_dataset
from config.config import load_config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def detect_single_text(text: str, detector: HateSpeechDetector, threshold: float) -> dict:
    """Detect hate speech in a single text."""
    result = detector.detect_hate_speech(text, return_confidence=True, threshold=threshold)
    return result


def detect_batch_texts(texts: List[str], detector: HateSpeechDetector, threshold: float) -> List[dict]:
    """Detect hate speech in a batch of texts."""
    results = detector.batch_detect(texts, return_confidence=True, threshold=threshold)
    return results


def load_texts_from_file(file_path: str) -> List[str]:
    """Load texts from a file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            elif isinstance(data, dict) and 'texts' in data:
                return [str(item) for item in data['texts']]
            else:
                raise ValueError("JSON file must contain a list or dict with 'texts' key")
    
    elif file_path.suffix.lower() == '.txt':
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    elif file_path.suffix.lower() == '.csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        if 'text' in df.columns:
            return df['text'].tolist()
        else:
            raise ValueError("CSV file must contain a 'text' column")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_results(results: List[dict], output_path: str, format: str = 'json'):
    """Save results to file."""
    output_path = Path(output_path)
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif format.lower() == 'csv':
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unsupported output format: {format}")


def print_results(results: List[dict], verbose: bool = False):
    """Print results to console."""
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Text: {result['text']}")
        
        if result['is_hate_speech']:
            print(f"🚨 HATE SPEECH DETECTED (Confidence: {result['confidence']:.2%})")
        else:
            print(f"✅ No hate speech detected (Confidence: {result['confidence']:.2%})")
        
        print(f"Label: {result['label']}")
        
        if verbose and 'all_scores' in result:
            print("All scores:")
            for score in result['all_scores']:
                print(f"  {score['label']}: {score['score']:.3f}")


def evaluate_model(detector: HateSpeechDetector, dataset_size: int, threshold: float):
    """Evaluate model on synthetic data."""
    print(f"Creating synthetic dataset with {dataset_size} samples...")
    synthetic_data = create_synthetic_dataset(dataset_size)
    
    print("Evaluating model...")
    eval_results = detector.evaluate_model(synthetic_data, threshold)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {eval_results['accuracy']:.3f}")
    print(f"Precision: {eval_results['precision']:.3f}")
    print(f"Recall: {eval_results['recall']:.3f}")
    print(f"F1-Score: {eval_results['f1_score']:.3f}")
    
    print("\nConfusion Matrix:")
    cm = eval_results['confusion_matrix']
    print(f"                Predicted")
    print(f"                No Hate  Hate")
    print(f"Actual No Hate    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Hate       {cm[1,0]:4d}    {cm[1,1]:4d}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Hate Speech Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single text
  python cli.py --text "You are worthless"
  
  # Analyze multiple texts
  python cli.py --text "Hello world" "You are terrible"
  
  # Analyze texts from file
  python cli.py --file texts.txt
  
  # Save results to file
  python cli.py --file texts.txt --output results.json
  
  # Evaluate model
  python cli.py --evaluate --dataset-size 200
  
  # Use different model
  python cli.py --model "unitary/toxic-roberta" --text "Test text"
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text', 
        nargs='+', 
        help='Text(s) to analyze'
    )
    input_group.add_argument(
        '--file', 
        help='File containing texts to analyze (supports .txt, .csv, .json)'
    )
    input_group.add_argument(
        '--evaluate', 
        action='store_true',
        help='Evaluate model on synthetic data'
    )
    
    # Model options
    parser.add_argument(
        '--model', 
        default='unitary/toxic-bert',
        help='Model name to use (default: unitary/toxic-bert)'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Confidence threshold for classification (default: 0.5)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for processing (default: 32)'
    )
    
    # Output options
    parser.add_argument(
        '--output', 
        help='Output file to save results (supports .json, .csv)'
    )
    parser.add_argument(
        '--format', 
        choices=['json', 'csv'], 
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Show detailed results'
    )
    
    # Evaluation options
    parser.add_argument(
        '--dataset-size', 
        type=int, 
        default=100,
        help='Size of synthetic dataset for evaluation (default: 100)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize detector
        logger.info(f"Initializing detector with model: {args.model}")
        detector = HateSpeechDetector(
            model_name=args.model,
            use_pipeline=True
        )
        
        if args.evaluate:
            # Evaluate model
            evaluate_model(detector, args.dataset_size, args.threshold)
        
        elif args.text:
            # Analyze single or multiple texts
            results = detect_batch_texts(args.text, detector, args.threshold)
            print_results(results, args.verbose)
            
            if args.output:
                save_results(results, args.output, args.format)
                print(f"\nResults saved to {args.output}")
        
        elif args.file:
            # Analyze texts from file
            logger.info(f"Loading texts from file: {args.file}")
            texts = load_texts_from_file(args.file)
            
            logger.info(f"Analyzing {len(texts)} texts...")
            results = detect_batch_texts(texts, detector, args.threshold)
            
            print_results(results, args.verbose)
            
            if args.output:
                save_results(results, args.output, args.format)
                print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
