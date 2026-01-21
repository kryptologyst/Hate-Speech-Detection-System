# Hate Speech Detection System

A comprehensive hate speech detection system using state-of-the-art transformer models. This project provides multiple interfaces (CLI, Web UI, Python API) for detecting hate speech in text using AI.

## Features

- **Multiple Model Support**: Use various pre-trained models (Toxic-BERT, RoBERTa, DistilBERT)
- **Multiple Interfaces**: Command-line, web interface, and Python API
- **Batch Processing**: Analyze multiple texts efficiently
- **Confidence Scoring**: Get confidence scores for predictions
- **Model Evaluation**: Evaluate model performance on synthetic data
- **Visualization**: Interactive charts and visualizations
- **Export Results**: Save results in JSON or CSV format
- **Configurable**: YAML-based configuration system

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Hate-Speech-Detection-System.git
cd Hate-Speech-Detection-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface

```bash
# Analyze single text
python src/cli.py --text "You are worthless"

# Analyze multiple texts
python src/cli.py --text "Hello world" "You are terrible"

# Analyze texts from file
python src/cli.py --file texts.txt

# Save results to file
python src/cli.py --file texts.txt --output results.json

# Evaluate model performance
python src/cli.py --evaluate --dataset-size 200
```

#### Web Interface

```bash
# Launch Streamlit web app
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

#### Python API

```python
from src.hate_speech_detector import HateSpeechDetector

# Initialize detector
detector = HateSpeechDetector()

# Analyze single text
result = detector.detect_hate_speech("You are worthless")
print(result)

# Analyze multiple texts
texts = ["Hello world", "You are terrible"]
results = detector.detect_hate_speech(texts)
print(results)
```

## 📁 Project Structure

```
hate-speech-detection/
├── src/                          # Source code
│   ├── hate_speech_detector.py   # Main detection module
│   └── cli.py                    # Command-line interface
├── web_app/                      # Web interface
│   └── app.py                    # Streamlit application
├── config/                       # Configuration
│   └── config.py                 # Configuration management
├── data/                         # Data directory
├── models/                       # Model storage
├── results/                      # Output results
├── tests/                        # Test files
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

The system uses YAML configuration files. Create a `config/config.yaml` file:

```yaml
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
```

## Supported Models

- **unitary/toxic-bert**: BERT-based model fine-tuned for toxicity detection
- **unitary/toxic-roberta**: RoBERTa-based model for toxicity detection
- **unitary/distilbert-base-uncased-toxic**: DistilBERT model for toxicity detection

## Usage Examples

### Single Text Analysis

```python
from src.hate_speech_detector import HateSpeechDetector

detector = HateSpeechDetector()
result = detector.detect_hate_speech("You are worthless")

print(f"Is hate speech: {result['is_hate_speech']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Label: {result['label']}")
```

### Batch Analysis

```python
texts = [
    "I love this beautiful day!",
    "You're a terrible person.",
    "Thank you for your help.",
    "I hate people like you."
]

results = detector.batch_detect(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Hate speech: {result['is_hate_speech']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("---")
```

### Model Evaluation

```python
from src.hate_speech_detector import create_synthetic_dataset

# Create synthetic dataset
synthetic_data = create_synthetic_dataset(100)

# Evaluate model
eval_results = detector.evaluate_model(synthetic_data)

print(f"Accuracy: {eval_results['accuracy']:.3f}")
print(f"Precision: {eval_results['precision']:.3f}")
print(f"Recall: {eval_results['recall']:.3f}")
print(f"F1-Score: {eval_results['f1_score']:.3f}")
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_detector.py
```

## Performance

The system provides various performance metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Advanced Features

### Custom Model Loading

```python
# Use custom model
detector = HateSpeechDetector(
    model_name="your-custom-model",
    device="cuda",  # Use GPU
    use_pipeline=False  # Use custom implementation
)
```

### Confidence Threshold Adjustment

```python
# Adjust confidence threshold
result = detector.detect_hate_speech(
    "Your text here",
    threshold=0.7  # Higher threshold = more conservative
)
```

### Visualization

```python
# Create visualizations
detector.visualize_results(results, save_path="results.png")
```

## 🛠️ Development

### Code Style

The project follows PEP 8 style guidelines. Format code with:

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Adding New Models

To add support for new models:

1. Update the model options in `web_app/app.py`
2. Test the model with `src/cli.py --model "your-model-name"`
3. Update documentation

## API Reference

### HateSpeechDetector Class

#### `__init__(model_name, device, use_pipeline)`
Initialize the detector with specified model and settings.

#### `detect_hate_speech(text, return_confidence, threshold)`
Detect hate speech in text(s).

#### `batch_detect(texts, batch_size, return_confidence, threshold)`
Efficiently process multiple texts.

#### `evaluate_model(test_data, threshold)`
Evaluate model performance on test data.

#### `visualize_results(results, save_path)`
Create visualizations of detection results.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes. Always review results carefully and consider context when making decisions. The model may have biases and limitations that should be understood before use in production environments.

## Support

For questions, issues, or contributions, please:

1. Check the existing issues
2. Create a new issue with detailed description
3. Provide example code and error messages

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Toxic-BERT Model](https://huggingface.co/unitary/toxic-bert)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
# Hate-Speech-Detection-System
