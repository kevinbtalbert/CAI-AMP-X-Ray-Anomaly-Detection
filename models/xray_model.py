"""
MLflow wrapper for X-ray anomaly detection model.
Supports deployment via Cloudera Models API.
"""
import mlflow
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import io
import base64
from typing import Dict, Any, List
import json

from models.grad_cam import GradCAM, draw_bounding_boxes


class XRayAnomalyDetector(mlflow.pyfunc.PythonModel):
    """
    MLflow Python model for X-ray anomaly detection with visual localization.
    """
    
    # Disease labels for chest X-ray classification
    LABELS = [
        "Atelectasis",
        "Cardiomegaly", 
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        "No Finding"
    ]
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.processor = None
        self.grad_cam = None
        self.device = None
    
    def load_context(self, context):
        """
        Load model from MLflow context.
        
        Args:
            context: MLflow context containing model artifacts
        """
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {self.device}")
        
        # Load model - using Swin Transformer fine-tuned for medical imaging
        # This model provides good performance for chest X-ray classification
        model_name = "microsoft/swin-tiny-patch4-window7-224"
        
        try:
            # Try to load from context artifacts first
            model_path = context.artifacts.get("model_path", model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                model_path,
                num_labels=len(self.LABELS),
                ignore_mismatched_sizes=True
            )
            self.processor = AutoImageProcessor.from_pretrained(model_path)
        except:
            # Fallback to pretrained model
            print(f"Loading pretrained model: {model_name}")
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=len(self.LABELS),
                ignore_mismatched_sizes=True
            )
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Grad-CAM
        # For Swin Transformer, target the last layer
        if hasattr(self.model, 'swin'):
            target_layer = self.model.swin.encoder.layers[-1].blocks[-1].layernorm_after
        elif hasattr(self.model, 'vit'):
            target_layer = self.model.vit.encoder.layer[-1].layernorm_after
        else:
            # Fallback to last layer
            target_layer = list(self.model.children())[-2]
        
        self.grad_cam = GradCAM(self.model, target_layer)
        
        print("Model loaded successfully")
    
    def _preprocess_image(self, image_input: Any) -> Image.Image:
        """
        Preprocess image input into PIL Image.
        
        Args:
            image_input: Can be file path, PIL Image, numpy array, or base64 string
            
        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image_input, str):
            if image_input.startswith('data:image') or image_input.startswith('/9j'):
                # Base64 encoded image
                if 'base64,' in image_input:
                    image_input = image_input.split('base64,')[1]
                image_bytes = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # File path
                image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def predict(self, context, model_input) -> List[Dict[str, Any]]:
        """
        Generate predictions with visual localization.
        
        Args:
            context: MLflow context
            model_input: Input data (DataFrame or dict with 'image' key)
            
        Returns:
            List of prediction dictionaries containing:
                - predictions: Dict of class probabilities
                - anomaly_score: Overall anomaly score (0-1)
                - top_findings: Top predicted findings
                - heatmap: Base64 encoded heatmap overlay
                - bounding_boxes: List of detected anomaly regions
        """
        # Handle different input formats
        if hasattr(model_input, 'to_dict'):
            # DataFrame input
            inputs = model_input.to_dict('records')
        elif isinstance(model_input, dict):
            inputs = [model_input]
        elif isinstance(model_input, list):
            inputs = model_input
        else:
            inputs = [{"image": model_input}]
        
        results = []
        
        for input_data in inputs:
            # Extract image
            if isinstance(input_data, dict):
                image_input = input_data.get('image', input_data)
            else:
                image_input = input_data
            
            # Preprocess
            pil_image = self._preprocess_image(image_input)
            
            # Process for model
            inputs_processed = self.processor(
                images=pil_image,
                return_tensors="pt"
            )
            pixel_values = inputs_processed['pixel_values'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(pixel_values)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0]
            
            # Convert to dict
            predictions = {
                label: float(prob) 
                for label, prob in zip(self.LABELS, probs.cpu().numpy())
            }
            
            # Calculate anomaly score (1 - P(No Finding))
            no_finding_prob = predictions.get("No Finding", 0.0)
            anomaly_score = 1.0 - no_finding_prob
            
            # Get top findings (excluding "No Finding")
            findings = [
                (label, prob) 
                for label, prob in predictions.items() 
                if label != "No Finding"
            ]
            findings.sort(key=lambda x: x[1], reverse=True)
            top_findings = [
                {"finding": label, "confidence": float(prob)}
                for label, prob in findings[:5]
            ]
            
            # Generate Grad-CAM for top finding
            top_class_idx = probs.argmax().item()
            cam = self.grad_cam.generate_cam(pixel_values, top_class_idx)
            
            # Get bounding boxes
            boxes = self.grad_cam.get_bounding_boxes(
                cam,
                threshold=0.5,
                min_area=50
            )
            
            # Create visualization
            image_np = np.array(pil_image)
            
            # Generate heatmap overlay
            heatmap_overlay = self.grad_cam.generate_heatmap_overlay(
                image_np,
                cam,
                alpha=0.4
            )
            
            # Draw bounding boxes
            if boxes:
                box_labels = [
                    f"{top_findings[0]['finding']}: {top_findings[0]['confidence']:.2f}"
                ] * len(boxes)
                heatmap_overlay = draw_bounding_boxes(
                    heatmap_overlay,
                    boxes,
                    labels=box_labels,
                    color=(255, 0, 0),
                    thickness=3
                )
            
            # Convert to base64
            heatmap_pil = Image.fromarray(heatmap_overlay)
            buffer = io.BytesIO()
            heatmap_pil.save(buffer, format='PNG')
            heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare result
            result = {
                "predictions": predictions,
                "anomaly_score": float(anomaly_score),
                "top_findings": top_findings,
                "heatmap_base64": heatmap_b64,
                "bounding_boxes": [
                    {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    for x, y, w, h in boxes
                ],
                "image_size": {
                    "width": pil_image.width,
                    "height": pil_image.height
                }
            }
            
            results.append(result)
        
        return results


def save_model(model_path: str = "xray_model"):
    """
    Save the model in MLflow format for deployment.
    
    Args:
        model_path: Path to save the model
    """
    # Create a sample model instance
    xray_model = XRayAnomalyDetector()
    
    # Define conda environment
    conda_env = {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python=3.9',
            'pip',
            {
                'pip': [
                    'mlflow==2.9.2',
                    'torch==2.1.0',
                    'torchvision==0.16.0',
                    'transformers==4.36.0',
                    'Pillow==10.1.0',
                    'numpy==1.24.3',
                    'opencv-python==4.8.1.78',
                    'scikit-learn==1.3.2',
                ],
            },
        ],
        'name': 'xray_env'
    }
    
    # Save model
    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=xray_model,
        conda_env=conda_env,
        code_path=["models/"],
    )
    
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    # Test the model locally
    print("Testing X-ray anomaly detection model...")
    
    # Save model
    model_path = save_model("xray_model")
    
    # Load and test
    loaded_model = mlflow.pyfunc.load_model(model_path)
    
    print("Model loaded successfully!")
    print("Ready for deployment to Cloudera ML")

