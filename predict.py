"""
CML Model prediction function for X-ray anomaly detection.
This file is deployed as a Cloudera ML Model.

Based on: https://docs.cloudera.com/machine-learning/1.5.4/models/topics/ml-creating-and-deploying-a-model.html
"""
import json
import base64
import io
from pathlib import Path
from PIL import Image
import numpy as np

# Import CML models decorator for PBJ Runtime
try:
    import cml.models_v1 as models
    HAS_CML = True
except ImportError:
    HAS_CML = False
    print("Warning: cml.models_v1 not available, decorator will be skipped")

# Import model dependencies
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Import Grad-CAM
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.grad_cam import GradCAM, draw_bounding_boxes


# Global variables for model (loaded once)
_model = None
_processor = None
_grad_cam = None
_device = None

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


def load_model():
    """Load the model (called once on startup)"""
    global _model, _processor, _grad_cam, _device
    
    if _model is not None:
        return _model, _processor, _grad_cam, _device
    
    print("Loading model...")
    
    # Determine device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")
    
    # Load pretrained Swin Transformer
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    
    _model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True
    )
    _processor = AutoImageProcessor.from_pretrained(model_name)
    
    _model.to(_device)
    _model.eval()
    
    # Initialize Grad-CAM
    if hasattr(_model, 'swin'):
        target_layer = _model.swin.encoder.layers[-1].blocks[-1].layernorm_after
    elif hasattr(_model, 'vit'):
        target_layer = _model.vit.encoder.layer[-1].layernorm_after
    else:
        target_layer = list(_model.children())[-2]
    
    _grad_cam = GradCAM(_model, target_layer)
    
    print("Model loaded successfully")
    return _model, _processor, _grad_cam, _device


def preprocess_image(image_input):
    """
    Preprocess image input into PIL Image.
    
    Args:
        image_input: Can be base64 string, file path, or dict with 'image' key
        
    Returns:
        PIL Image in RGB mode
    """
    # Handle dict input
    if isinstance(image_input, dict):
        if 'image' in image_input:
            image_input = image_input['image']
        elif 'image_base64' in image_input:
            image_input = image_input['image_base64']
    
    # Handle base64 encoded image
    if isinstance(image_input, str):
        if image_input.startswith('data:image'):
            # Remove data URL prefix
            image_input = image_input.split('base64,')[1]
        
        try:
            # Try to decode as base64
            image_bytes = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_bytes))
        except:
            # Assume it's a file path
            image = Image.open(image_input)
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


# CML Model decorator (if available)
if HAS_CML:
    @models.cml_model
    def predict(args):
        """
        CML Model prediction function.
        
        Args:
            args: Dict with 'image' or 'image_base64' key containing image data
            
        Returns:
            Dict with predictions, anomaly score, heatmap, and bounding boxes
        """
        try:
            # Load model
            model, processor, grad_cam, device = load_model()
            
            # Preprocess image
            pil_image = preprocess_image(args)
            
            # Process for model
            inputs = processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(pixel_values)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0]
            
            # Convert to dict
            predictions = {
                label: float(prob) 
                for label, prob in zip(LABELS, probs.cpu().numpy())
            }
            
            # Calculate anomaly score
            no_finding_prob = predictions.get("No Finding", 0.0)
            anomaly_score = 1.0 - no_finding_prob
            
            # Get top findings
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
            
            # Generate Grad-CAM
            top_class_idx = probs.argmax().item()
            cam = grad_cam.generate_cam(pixel_values, top_class_idx)
            
            # Get bounding boxes
            boxes = grad_cam.get_bounding_boxes(cam, threshold=0.5, min_area=50)
            
            # Create visualization
            image_np = np.array(pil_image)
            heatmap_overlay = grad_cam.generate_heatmap_overlay(image_np, cam, alpha=0.4)
            
            # Draw bounding boxes
            if boxes:
                box_labels = [
                    f"{top_findings[0]['finding']}: {top_findings[0]['confidence']:.2f}"
                ] * len(boxes)
                heatmap_overlay = draw_bounding_boxes(
                    heatmap_overlay, boxes, labels=box_labels, 
                    color=(255, 0, 0), thickness=3
                )
            
            # Convert heatmap to base64
            heatmap_pil = Image.fromarray(heatmap_overlay)
            buffer = io.BytesIO()
            heatmap_pil.save(buffer, format='PNG')
            heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Determine risk level
            if anomaly_score >= 0.7:
                risk_level = "HIGH RISK"
            elif anomaly_score >= 0.4:
                risk_level = "SUSPICIOUS"
            else:
                risk_level = "NORMAL"
            
            # Return result
            return {
                "predictions": predictions,
                "anomaly_score": float(anomaly_score),
                "risk_level": risk_level,
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
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Error processing image"
            }
else:
    # Fallback without decorator for local testing
    def predict(args):
        """
        Prediction function without CML decorator (for local testing).
        """
        try:
            # Load model
            model, processor, grad_cam, device = load_model()
            
            # Preprocess image
            pil_image = preprocess_image(args)
            
            # Process for model
            inputs = processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(pixel_values)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0]
            
            # Convert to dict
            predictions = {
                label: float(prob) 
                for label, prob in zip(LABELS, probs.cpu().numpy())
            }
            
            # Calculate anomaly score
            no_finding_prob = predictions.get("No Finding", 0.0)
            anomaly_score = 1.0 - no_finding_prob
            
            # Get top findings
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
            
            # Generate Grad-CAM
            top_class_idx = probs.argmax().item()
            cam = grad_cam.generate_cam(pixel_values, top_class_idx)
            
            # Get bounding boxes
            boxes = grad_cam.get_bounding_boxes(cam, threshold=0.5, min_area=50)
            
            # Create visualization
            image_np = np.array(pil_image)
            heatmap_overlay = grad_cam.generate_heatmap_overlay(image_np, cam, alpha=0.4)
            
            # Draw bounding boxes
            if boxes:
                box_labels = [
                    f"{top_findings[0]['finding']}: {top_findings[0]['confidence']:.2f}"
                ] * len(boxes)
                heatmap_overlay = draw_bounding_boxes(
                    heatmap_overlay, boxes, labels=box_labels,
                    color=(255, 0, 0), thickness=3
                )
            
            # Convert heatmap to base64
            heatmap_pil = Image.fromarray(heatmap_overlay)
            buffer = io.BytesIO()
            heatmap_pil.save(buffer, format='PNG')
            heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Determine risk level
            if anomaly_score >= 0.7:
                risk_level = "HIGH RISK"
            elif anomaly_score >= 0.4:
                risk_level = "SUSPICIOUS"
            else:
                risk_level = "NORMAL"
            
            # Return result
            return {
                "predictions": predictions,
                "anomaly_score": float(anomaly_score),
                "risk_level": risk_level,
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
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Error processing image"
            }


# For local testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with image path
        test_image = sys.argv[1]
        print(f"Testing with image: {test_image}")
        
        result = predict({"image": test_image})
        
        print("\nResult:")
        print(f"Anomaly Score: {result['anomaly_score']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print("\nTop Findings:")
        for finding in result['top_findings'][:3]:
            print(f"  - {finding['finding']}: {finding['confidence']:.2%}")
        print(f"\nBounding Boxes: {len(result['bounding_boxes'])}")
    else:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py data/sample_xrays/pneumonia_bacteria_01.jpeg")

