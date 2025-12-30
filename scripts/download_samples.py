"""
Download sample chest X-ray images for testing.
Uses publicly available datasets from NIH Chest X-ray dataset.
"""
import os
import requests
from pathlib import Path
from PIL import Image
import io


# Sample X-ray image URLs (public domain / open datasets)
SAMPLE_IMAGES = [
    {
        "name": "normal_chest_xray_1.jpg",
        "url": "https://openi.nlm.nih.gov/imgs/512/208/208/CXR208_IM-0692-1001.png",
        "description": "Normal chest X-ray"
    },
    {
        "name": "pneumonia_chest_xray_1.jpg", 
        "url": "https://openi.nlm.nih.gov/imgs/512/364/364/CXR364_IM-1699-1001.png",
        "description": "Chest X-ray with pneumonia"
    },
    {
        "name": "effusion_chest_xray_1.jpg",
        "url": "https://openi.nlm.nih.gov/imgs/512/100/100/CXR100_IM-0076-1001.png",
        "description": "Chest X-ray with pleural effusion"
    },
    {
        "name": "cardiomegaly_chest_xray_1.jpg",
        "url": "https://openi.nlm.nih.gov/imgs/512/1/1/CXR1_IM-0001-1001.png",
        "description": "Chest X-ray with cardiomegaly"
    },
    {
        "name": "normal_chest_xray_2.jpg",
        "url": "https://openi.nlm.nih.gov/imgs/512/3/3/CXR3_IM-0003-1001.png",
        "description": "Normal chest X-ray (frontal)"
    },
]


def download_sample_images(output_dir: str = "data/sample_xrays"):
    """
    Download sample X-ray images.
    
    Args:
        output_dir: Directory to save images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading sample X-ray images to {output_dir}...")
    
    downloaded = 0
    for img_info in SAMPLE_IMAGES:
        output_file = output_path / img_info["name"]
        
        # Skip if already exists
        if output_file.exists():
            print(f"✓ {img_info['name']} already exists")
            downloaded += 1
            continue
        
        try:
            print(f"Downloading {img_info['name']}...")
            response = requests.get(img_info["url"], timeout=30)
            response.raise_for_status()
            
            # Open and convert to RGB
            image = Image.open(io.BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            image.save(output_file, 'JPEG', quality=95)
            
            print(f"✓ Downloaded {img_info['name']} - {img_info['description']}")
            downloaded += 1
            
        except Exception as e:
            print(f"✗ Failed to download {img_info['name']}: {e}")
    
    print(f"\nDownloaded {downloaded}/{len(SAMPLE_IMAGES)} images successfully")
    
    # Create a metadata file
    metadata_file = output_path / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Sample Chest X-ray Images\n")
        f.write("=" * 50 + "\n\n")
        for img_info in SAMPLE_IMAGES:
            f.write(f"File: {img_info['name']}\n")
            f.write(f"Description: {img_info['description']}\n")
            f.write(f"Source: {img_info['url']}\n")
            f.write("\n")
    
    print(f"Metadata saved to {metadata_file}")
    
    return output_path


if __name__ == "__main__":
    download_sample_images()

