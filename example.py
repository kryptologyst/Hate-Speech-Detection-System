#!/usr/bin/env python3
"""
Example script demonstrating the Hate Speech Detection system.

This script shows how to use the system for various tasks including
single text analysis, batch processing, and model evaluation.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from hate_speech_detector import HateSpeechDetector, create_synthetic_dataset


def main():
    """Main example function."""
    print("🛡️ Hate Speech Detection System - Example Usage")
    print("=" * 50)
    
    # Initialize detector
    print("\n1. Initializing detector...")
    detector = HateSpeechDetector()
    print(f"   Model: {detector.model_name}")
    print(f"   Device: {detector.device}")
    
    # Single text analysis
    print("\n2. Single text analysis:")
    test_texts = [
        "I love this beautiful sunny day!",
        "You are a worthless person and should disappear.",
        "Thank you for your help with the project.",
        "I hate people like you, you're disgusting.",
        "The movie was really entertaining and well-made."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Example {i}:")
        print(f"   Text: '{text}'")
        
        result = detector.detect_hate_speech(text, return_confidence=True)
        
        if result['is_hate_speech']:
            print(f"   🚨 HATE SPEECH DETECTED (Confidence: {result['confidence']:.2%})")
        else:
            print(f"   ✅ No hate speech detected (Confidence: {result['confidence']:.2%})")
        
        print(f"   Label: {result['label']}")
    
    # Batch analysis
    print("\n3. Batch analysis:")
    print(f"   Processing {len(test_texts)} texts...")
    
    batch_results = detector.batch_detect(test_texts)
    
    hate_count = sum(1 for r in batch_results if r['is_hate_speech'])
    non_hate_count = len(batch_results) - hate_count
    
    print(f"   Results: {hate_count} hate speech, {non_hate_count} non-hate speech")
    
    # Model evaluation
    print("\n4. Model evaluation on synthetic data:")
    print("   Creating synthetic dataset...")
    
    synthetic_data = create_synthetic_dataset(50)  # Small dataset for demo
    eval_results = detector.evaluate_model(synthetic_data)
    
    print(f"   Accuracy: {eval_results['accuracy']:.3f}")
    print(f"   Precision: {eval_results['precision']:.3f}")
    print(f"   Recall: {eval_results['recall']:.3f}")
    print(f"   F1-Score: {eval_results['f1_score']:.3f}")
    
    # Confusion matrix
    cm = eval_results['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"                Predicted")
    print(f"                No Hate  Hate")
    print(f"   Actual No Hate    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"          Hate       {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Model information
    print("\n5. Model information:")
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 50)
    print("✅ Example completed successfully!")
    print("\nTo run the web interface:")
    print("   streamlit run web_app/app.py")
    print("\nTo use the CLI:")
    print("   python src/cli.py --text 'Your text here'")
    print("\nFor more examples, see the README.md file.")


if __name__ == "__main__":
    main()
