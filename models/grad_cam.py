"""
Grad-CAM implementation for visual localization of anomalies in X-ray images.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanation.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for gradient extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (None for highest prediction)
            
        Returns:
            Heatmap as numpy array [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_heatmap_overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create heatmap overlay on original image.
        
        Args:
            image: Original image [H, W, C] in RGB, values 0-255
            cam: CAM heatmap [H, W], values 0-1
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Overlayed image [H, W, C]
        """
        # Resize CAM to match image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = np.uint8(image)
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed
    
    def get_bounding_boxes(
        self,
        cam: np.ndarray,
        threshold: float = 0.5,
        min_area: int = 100
    ) -> list:
        """
        Extract bounding boxes from CAM heatmap.
        
        Args:
            cam: CAM heatmap [H, W], values 0-1
            threshold: Threshold for binarization
            min_area: Minimum contour area
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        # Threshold and convert to binary
        binary_mask = (cam > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract bounding boxes
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))
        
        return boxes


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: list,
    labels: Optional[list] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Image array [H, W, C]
        boxes: List of boxes [(x, y, w, h), ...]
        labels: Optional labels for each box
        color: Box color (R, G, B)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    image_copy = image.copy()
    
    for i, (x, y, w, h) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(
            image_copy,
            (x, y),
            (x + w, y + h),
            color,
            thickness
        )
        
        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                image_copy,
                (x, y - text_h - baseline - 5),
                (x + text_w, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image_copy,
                label,
                (x, y - baseline - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
    
    return image_copy

