"""
Streamlit Web Interface for Hate Speech Detection

This module provides a user-friendly web interface for the hate speech detection system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hate_speech_detector import HateSpeechDetector, create_synthetic_dataset
from config.config import load_config


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Hate Speech Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.title("🛡️ Hate Speech Detection")
    st.sidebar.markdown("---")
    
    # Model configuration
    st.sidebar.subheader("Model Configuration")
    
    model_options = {
        "Toxic-BERT": "unitary/toxic-bert",
        "RoBERTa Toxic": "unitary/toxic-roberta",
        "DistilBERT Toxic": "unitary/distilbert-base-uncased-toxic"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for hate speech detection"
    )
    
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=64,
        value=16,
        help="Number of texts to process at once"
    )
    
    return {
        "model_name": model_options[selected_model],
        "threshold": threshold,
        "batch_size": batch_size
    }


def display_model_info(detector: HateSpeechDetector):
    """Display information about the loaded model."""
    with st.expander("Model Information", expanded=False):
        model_info = detector.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model", model_info["model_name"])
        
        with col2:
            st.metric("Device", model_info["device"])
        
        with col3:
            st.metric("Type", model_info["model_type"])


def single_text_analysis(detector: HateSpeechDetector, config: Dict):
    """Interface for single text analysis."""
    st.subheader("📝 Single Text Analysis")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your text here...",
        height=100
    )
    
    if st.button("Analyze Text", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                result = detector.detect_hate_speech(
                    text_input, 
                    return_confidence=True,
                    threshold=config["threshold"]
                )
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Analysis Result:**")
                
                if result["is_hate_speech"]:
                    st.error(f"🚨 **Hate Speech Detected** (Confidence: {result['confidence']:.2%})")
                else:
                    st.success(f"✅ **No Hate Speech Detected** (Confidence: {result['confidence']:.2%})")
                
                st.write(f"**Label:** {result['label']}")
                st.write(f"**Confidence Score:** {result['confidence']:.2%}")
            
            with col2:
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['confidence'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence"},
                    delta = {'reference': config["threshold"]},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, config["threshold"]], 'color': "lightgray"},
                            {'range': [config["threshold"], 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': config["threshold"]
                        }
                    }
                ))
                
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed scores if available
            if "all_scores" in result:
                with st.expander("Detailed Scores"):
                    scores_df = pd.DataFrame(result["all_scores"])
                    st.dataframe(scores_df)
        else:
            st.warning("Please enter some text to analyze.")


def batch_analysis(detector: HateSpeechDetector, config: Dict):
    """Interface for batch text analysis."""
    st.subheader("📊 Batch Analysis")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload CSV file with texts",
        type=['csv'],
        help="CSV file should have a 'text' column"
    )
    
    # Manual input option
    st.write("**Or enter multiple texts manually:**")
    
    texts_input = st.text_area(
        "Enter texts (one per line):",
        placeholder="Text 1\nText 2\nText 3\n...",
        height=150
    )
    
    if st.button("Analyze Batch", type="primary"):
        texts = []
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts = df['text'].tolist()
                else:
                    st.error("CSV file must contain a 'text' column")
                    return
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        
        elif texts_input.strip():
            texts = [line.strip() for line in texts_input.split('\n') if line.strip()]
        
        if texts:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                results = detector.batch_detect(
                    texts,
                    batch_size=config["batch_size"],
                    return_confidence=True,
                    threshold=config["threshold"]
                )
            
            # Display results summary
            st.subheader("📈 Analysis Summary")
            
            hate_count = sum(1 for r in results if r["is_hate_speech"])
            non_hate_count = len(results) - hate_count
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Texts", len(results))
            
            with col2:
                st.metric("Hate Speech", hate_count, delta=f"{hate_count/len(results):.1%}")
            
            with col3:
                st.metric("Non-Hate Speech", non_hate_count)
            
            with col4:
                avg_confidence = np.mean([r["confidence"] for r in results])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Visualization
            st.subheader("📊 Visualizations")
            
            # Confidence distribution
            confidences = [r["confidence"] for r in results]
            labels = ["Hate Speech" if r["is_hate_speech"] else "Non-Hate Speech" for r in results]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Confidence Distribution", "Label Distribution"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Confidence histogram
            fig.add_trace(
                go.Histogram(x=confidences, nbinsx=20, name="Confidence"),
                row=1, col=1
            )
            
            # Label pie chart
            label_counts = pd.Series(labels).value_counts()
            fig.add_trace(
                go.Pie(labels=label_counts.index, values=label_counts.values, name="Labels"),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            
            results_df = pd.DataFrame(results)
            results_df['confidence'] = results_df['confidence'].round(3)
            
            # Color coding for hate speech
            def highlight_hate_speech(row):
                if row['is_hate_speech']:
                    return ['background-color: #ffebee'] * len(row)
                else:
                    return ['background-color: #e8f5e8'] * len(row)
            
            st.dataframe(
                results_df.style.apply(highlight_hate_speech, axis=1),
                use_container_width=True
            )
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="hate_speech_analysis_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please provide texts to analyze.")


def model_evaluation(detector: HateSpeechDetector, config: Dict):
    """Interface for model evaluation."""
    st.subheader("🧪 Model Evaluation")
    
    st.write("Evaluate the model performance on synthetic data:")
    
    dataset_size = st.slider(
        "Dataset Size",
        min_value=50,
        max_value=500,
        value=100,
        step=50
    )
    
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Generating synthetic dataset and evaluating model..."):
            # Create synthetic dataset
            synthetic_data = create_synthetic_dataset(dataset_size)
            
            # Evaluate model
            eval_results = detector.evaluate_model(synthetic_data, config["threshold"])
        
        # Display evaluation metrics
        st.subheader("📊 Evaluation Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{eval_results['accuracy']:.3f}")
        
        with col2:
            st.metric("Precision", f"{eval_results['precision']:.3f}")
        
        with col3:
            st.metric("Recall", f"{eval_results['recall']:.3f}")
        
        with col4:
            st.metric("F1-Score", f"{eval_results['f1_score']:.3f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        cm = eval_results['confusion_matrix']
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Predicted", y="Actual"),
            x=["Non-Hate", "Hate Speech"],
            y=["Non-Hate", "Hate Speech"],
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed classification report
        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(eval_results['classification_report']).transpose()
            st.dataframe(report_df)


def main():
    """Main application function."""
    setup_page_config()
    
    # Load configuration
    config_dict = create_sidebar()
    
    # Initialize detector
    @st.cache_resource
    def load_detector(model_name: str):
        return HateSpeechDetector(model_name=model_name)
    
    detector = load_detector(config_dict["model_name"])
    
    # Main content
    st.title("🛡️ Hate Speech Detection System")
    st.markdown("Detect hate speech in text using state-of-the-art AI models")
    
    # Display model info
    display_model_info(detector)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Analysis", "Model Evaluation"])
    
    with tab1:
        single_text_analysis(detector, config_dict)
    
    with tab2:
        batch_analysis(detector, config_dict)
    
    with tab3:
        model_evaluation(detector, config_dict)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This tool is for educational and research purposes. "
        "Always review results carefully and consider context when making decisions."
    )


if __name__ == "__main__":
    main()
