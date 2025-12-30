"""
Create sample chest X-ray placeholder images for testing.
These are synthetic images for demonstration purposes.
"""
import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["python3", "-m", "pip", "install", "-q", "Pillow", "numpy"])
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np


def create_synthetic_xray(
    width: int = 512,
    height: int = 512,
    anomaly_type: str = "normal",
    output_path: str = None
) -> Image.Image:
    """
    Create a synthetic X-ray-like image for testing.
    
    Args:
        width: Image width
        height: Image height
        anomaly_type: Type of anomaly to simulate
        output_path: Path to save image
        
    Returns:
        PIL Image
    """
    # Create base grayscale image (chest X-ray appearance)
    # Darker at edges, lighter in center (lung fields)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Radial gradient for chest cavity
    R = np.sqrt(X**2 + Y**2)
    base = 180 - (R * 80).astype(np.uint8)
    base = np.clip(base, 40, 220)
    
    # Add some texture/noise
    noise = np.random.normal(0, 15, (height, width))
    base = base + noise
    base = np.clip(base, 0, 255).astype(np.uint8)
    
    # Add anatomical features
    # Spine (vertical dark line in center)
    spine_x = width // 2
    spine_width = width // 40
    base[:, spine_x-spine_width:spine_x+spine_width] -= 40
    
    # Ribs (horizontal curved lines)
    for i in range(4, height-4, height//8):
        for x in range(width):
            curve = int(20 * np.sin(x / width * np.pi))
            y_pos = i + curve
            if 0 <= y_pos < height:
                base[y_pos-2:y_pos+2, x] -= 20
    
    # Heart shadow (left side)
    heart_center = (width // 3, height // 2)
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - heart_center[1])**2 + (j - heart_center[0])**2)
            if dist < height // 5:
                base[i, j] -= int(30 * (1 - dist / (height // 5)))
    
    # Add anomaly based on type
    if anomaly_type == "pneumonia":
        # Add cloudy opacity in lower lung
        anomaly_y = int(height * 0.6)
        anomaly_x = int(width * 0.3)
        for i in range(height // 4):
            for j in range(width // 3):
                y = anomaly_y + i
                x = anomaly_x + j
                if 0 <= y < height and 0 <= x < width:
                    dist = np.sqrt((i - height//8)**2 + (j - width//6)**2)
                    if dist < height // 6:
                        base[y, x] -= int(40 * (1 - dist / (height // 6)))
    
    elif anomaly_type == "effusion":
        # Add fluid accumulation at bottom
        for i in range(height // 4):
            base[height - i - 1, :] -= int(50 * (i / (height // 4)))
    
    elif anomaly_type == "nodule":
        # Add small bright spots
        nodule_centers = [
            (int(width * 0.4), int(height * 0.4)),
            (int(width * 0.6), int(height * 0.5))
        ]
        for center in nodule_centers:
            for i in range(-15, 15):
                for j in range(-15, 15):
                    y = center[1] + i
                    x = center[0] + j
                    if 0 <= y < height and 0 <= x < width:
                        dist = np.sqrt(i**2 + j**2)
                        if dist < 12:
                            base[y, x] += int(30 * (1 - dist / 12))
    
    elif anomaly_type == "cardiomegaly":
        # Enlarge heart shadow
        heart_center = (width // 3, height // 2)
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - heart_center[1])**2 + (j - heart_center[0])**2)
                if dist < height // 3:  # Larger than normal
                    base[i, j] -= int(40 * (1 - dist / (height // 3)))
    
    # Clip values
    base = np.clip(base, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image (RGB for compatibility)
    image = Image.fromarray(base, mode='L').convert('RGB')
    
    # Add text label
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    label = f"Synthetic CXR - {anomaly_type.title()}"
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    
    # Save if path provided
    if output_path:
        image.save(output_path, 'JPEG', quality=95)
        print(f"✓ Created {output_path}")
    
    return image


def create_sample_dataset(output_dir: str = "data/sample_xrays"):
    """
    Create a set of sample X-ray images.
    
    Args:
        output_dir: Directory to save images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating synthetic X-ray images in {output_dir}...")
    
    # Define samples to create
    samples = [
        ("normal_chest_xray_1.jpg", "normal", "Normal chest X-ray"),
        ("normal_chest_xray_2.jpg", "normal", "Normal chest X-ray (variant)"),
        ("pneumonia_chest_xray_1.jpg", "pneumonia", "Chest X-ray with pneumonia"),
        ("effusion_chest_xray_1.jpg", "effusion", "Chest X-ray with pleural effusion"),
        ("nodule_chest_xray_1.jpg", "nodule", "Chest X-ray with nodules"),
        ("cardiomegaly_chest_xray_1.jpg", "cardiomegaly", "Chest X-ray with cardiomegaly"),
    ]
    
    created = 0
    for filename, anomaly_type, description in samples:
        output_file = output_path / filename
        
        # Skip if already exists
        if output_file.exists():
            print(f"✓ {filename} already exists")
            created += 1
            continue
        
        try:
            create_synthetic_xray(
                width=512,
                height=512,
                anomaly_type=anomaly_type,
                output_path=str(output_file)
            )
            created += 1
        except Exception as e:
            print(f"✗ Failed to create {filename}: {e}")
    
    print(f"\nCreated {created}/{len(samples)} images successfully")
    
    # Create metadata file
    metadata_file = output_path / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Synthetic Chest X-ray Images for Testing\n")
        f.write("=" * 50 + "\n\n")
        f.write("NOTE: These are synthetic images created for demonstration.\n")
        f.write("They simulate the appearance of chest X-rays but are not real medical images.\n\n")
        for filename, anomaly_type, description in samples:
            f.write(f"File: {filename}\n")
            f.write(f"Type: {anomaly_type}\n")
            f.write(f"Description: {description}\n")
            f.write("\n")
    
    print(f"Metadata saved to {metadata_file}")
    print("\n⚠️  Note: These are synthetic images for demonstration only.")
    print("For production use, replace with real medical imaging datasets.")
    
    return output_path


if __name__ == "__main__":
    create_sample_dataset()

