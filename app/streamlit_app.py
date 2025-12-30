"""
Streamlit UI for X-ray Anomaly Detection.
Displays X-ray images with visual anomaly localization (bounding boxes and heatmaps).
"""
import streamlit as st
import sys
from pathlib import Path
import base64
import io
from PIL import Image
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow


# Page configuration
st.set_page_config(
    page_title="X-Ray Anomaly Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .finding-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .anomaly-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .anomaly-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .anomaly-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the MLflow model (cached)."""
    model_path = "xray_model"
    
    if not Path(model_path).exists():
        st.error(f"Model not found at {model_path}")
        st.info("Please run: `python models/xray_model.py` to create the model first")
        return None
    
    try:
        model = mlflow.pyfunc.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def get_anomaly_level(score: float) -> tuple:
    """
    Get anomaly level and color based on score.
    
    Returns:
        (level_text, css_class)
    """
    if score >= 0.7:
        return "HIGH RISK", "anomaly-high"
    elif score >= 0.4:
        return "SUSPICIOUS", "anomaly-medium"
    else:
        return "NORMAL", "anomaly-low"


def display_results(result: dict, original_image: Image.Image):
    """
    Display inference results with visualizations.
    
    Args:
        result: Inference result dictionary
        original_image: Original PIL Image
    """
    # Extract results
    anomaly_score = result['anomaly_score']
    top_findings = result['top_findings']
    bounding_boxes = result['bounding_boxes']
    heatmap_b64 = result.get('heatmap_base64', '')
    
    # Anomaly level
    level_text, level_class = get_anomaly_level(anomaly_score)
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Analysis Results")
        
        # Anomaly score
        st.markdown(f"""
        <div class="metric-card">
            <h3>Anomaly Score</h3>
            <h1 class="{level_class}">{anomaly_score:.1%}</h1>
            <p class="{level_class}">{level_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top findings
        st.markdown("### 🔍 Detected Findings")
        
        if top_findings:
            for i, finding in enumerate(top_findings[:5]):
                confidence = finding['confidence']
                finding_name = finding['finding']
                
                # Progress bar color based on confidence
                if confidence >= 0.7:
                    bar_color = "#d32f2f"
                elif confidence >= 0.4:
                    bar_color = "#f57c00"
                else:
                    bar_color = "#388e3c"
                
                st.markdown(f"""
                <div class="finding-item">
                    <strong>{i+1}. {finding_name}</strong><br>
                    <small>Confidence: {confidence:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(confidence)
        else:
            st.info("No significant findings detected")
        
        # Bounding boxes info
        st.markdown("### 📦 Localization")
        st.metric("Anomalous Regions Detected", len(bounding_boxes))
        
        if bounding_boxes:
            with st.expander("View Bounding Box Coordinates"):
                for i, box in enumerate(bounding_boxes):
                    st.text(f"Region {i+1}: x={box['x']}, y={box['y']}, "
                           f"width={box['width']}, height={box['height']}")
    
    with col2:
        st.subheader("🖼️ Visual Localization")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["Heatmap Overlay", "Original Image"])
        
        with tab1:
            if heatmap_b64:
                # Decode and display heatmap
                heatmap_bytes = base64.b64decode(heatmap_b64)
                heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                
                st.image(
                    heatmap_image,
                    caption="Anomaly Heatmap with Bounding Boxes",
                    use_container_width=True
                )
                
                st.markdown("""
                <small>
                🔴 Red overlay indicates regions of interest<br>
                📦 Bounding boxes highlight detected anomalies
                </small>
                """, unsafe_allow_html=True)
            else:
                st.warning("Heatmap not available")
        
        with tab2:
            st.image(
                original_image,
                caption="Original X-ray Image",
                use_container_width=True
            )
    
    # Full prediction details (expandable)
    with st.expander("📋 View All Predictions"):
        predictions = result.get('predictions', {})
        
        # Sort by probability
        sorted_preds = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        st.markdown("#### All Class Probabilities")
        for label, prob in sorted_preds:
            st.text(f"{label:.<30} {prob:.4f}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🏥 X-Ray Anomaly Detection</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Medical Image Analysis with Visual Localization</div>',
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        with st.spinner("Loading model..."):
            model = load_model()
        
        if model:
            st.success("✓ Model loaded successfully")
        else:
            st.error("✗ Model not loaded")
            st.stop()
        
        st.markdown("---")
        
        # Image source selection
        st.subheader("📁 Image Source")
        source = st.radio(
            "Select source:",
            ["Upload Image", "Sample Images"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            This application uses a Vision Transformer model with Grad-CAM 
            to detect and localize anomalies in chest X-ray images.
            
            **Features:**
            - Multi-class disease detection
            - Visual localization with heatmaps
            - Bounding box detection
            - Confidence scoring
            
            **Model:** Swin Transformer fine-tuned for medical imaging
            """)
    
    # Main content
    uploaded_file = None
    selected_sample = None
    
    if source == "Upload Image":
        st.subheader("📤 Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image (JPG, PNG, DICOM)",
            type=["jpg", "jpeg", "png"],
            help="Upload a chest X-ray image for analysis"
        )
    else:
        st.subheader("📂 Select Sample Image")
        
        # Load sample images
        sample_dir = Path("data/sample_xrays")
        
        if not sample_dir.exists():
            st.warning("Sample images not found. Downloading...")
            
            # Try to download samples
            try:
                from scripts.download_samples import download_sample_images
                download_sample_images()
                st.success("✓ Sample images downloaded")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to download samples: {e}")
                st.info("Please run: `python scripts/download_samples.py`")
                st.stop()
        
        sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        
        if sample_files:
            # Display sample thumbnails
            cols = st.columns(3)
            
            for i, sample_path in enumerate(sample_files):
                col_idx = i % 3
                
                with cols[col_idx]:
                    img = Image.open(sample_path)
                    st.image(img, caption=sample_path.name, use_container_width=True)
                    
                    if st.button(f"Analyze", key=f"btn_{i}"):
                        selected_sample = sample_path
        else:
            st.warning("No sample images found")
    
    # Process image
    if uploaded_file is not None or selected_sample is not None:
        st.markdown("---")
        
        # Load image
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_input = uploaded_file
        else:
            image = Image.open(selected_sample)
            image_input = str(selected_sample)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run inference
        with st.spinner("🔍 Analyzing X-ray image..."):
            try:
                result = model.predict({"image": image_input})
                
                if result and len(result) > 0:
                    display_results(result[0], image)
                else:
                    st.error("No results returned from model")
                    
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.exception(e)
    else:
        # Instructions
        st.info("👆 Please upload an X-ray image or select a sample to begin analysis")
        
        # Demo features
        st.markdown("### ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Anomaly Detection**
            
            Identifies abnormalities across 14+ pathology classes including:
            - Pneumonia
            - Effusion
            - Cardiomegaly
            - Nodules
            - And more...
            """)
        
        with col2:
            st.markdown("""
            **🔍 Visual Localization**
            
            Highlights suspicious regions using:
            - Grad-CAM heatmaps
            - Bounding box detection
            - Color-coded overlays
            - Region coordinates
            """)
        
        with col3:
            st.markdown("""
            **📊 Confidence Scoring**
            
            Provides detailed analysis:
            - Overall anomaly score
            - Per-finding confidence
            - Risk level classification
            - Complete probability distribution
            """)


if __name__ == "__main__":
    main()

