"""
Prepare existing X-ray datasets for the anomaly detection system.
Uses the real NORMAL and PNEUMONIA datasets already in data/ directory.
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict


def organize_xray_data(
    normal_dir="data/NORMAL",
    pneumonia_dir="data/PNEUMONIA",
    output_dir="data/sample_xrays"
):
    """
    Organize existing X-ray images for the application.
    
    Args:
        normal_dir: Directory containing normal X-rays
        pneumonia_dir: Directory containing pneumonia X-rays
        output_dir: Output directory for organized samples
    """
    normal_path = Path(normal_dir)
    pneumonia_path = Path(pneumonia_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Organizing X-Ray Dataset")
    print("=" * 60)
    
    # Count images
    normal_images = list(normal_path.glob("*.jpeg")) + list(normal_path.glob("*.jpg"))
    pneumonia_images = list(pneumonia_path.glob("*.jpeg")) + list(pneumonia_path.glob("*.jpg"))
    
    print(f"\nFound {len(normal_images)} normal X-rays")
    print(f"Found {len(pneumonia_images)} pneumonia X-rays")
    
    # Select diverse samples (first 10 of each for quick testing)
    samples_to_copy = []
    
    # Normal samples
    for i, img in enumerate(normal_images[:10]):
        samples_to_copy.append((img, output_path / f"normal_{i+1:02d}.jpeg", "Normal"))
    
    # Pneumonia samples (mix of bacterial and viral)
    bacteria_samples = [img for img in pneumonia_images if 'bacteria' in img.name][:5]
    virus_samples = [img for img in pneumonia_images if 'virus' in img.name][:5]
    
    for i, img in enumerate(bacteria_samples):
        samples_to_copy.append((img, output_path / f"pneumonia_bacteria_{i+1:02d}.jpeg", "Pneumonia (Bacterial)"))
    
    for i, img in enumerate(virus_samples):
        samples_to_copy.append((img, output_path / f"pneumonia_virus_{i+1:02d}.jpeg", "Pneumonia (Viral)"))
    
    # Copy samples
    print(f"\nCopying {len(samples_to_copy)} sample images to {output_dir}...")
    
    copied = 0
    for src, dst, description in samples_to_copy:
        try:
            if dst.exists():
                print(f"✓ {dst.name} already exists")
            else:
                shutil.copy2(src, dst)
                print(f"✓ Copied {dst.name} - {description}")
            copied += 1
        except Exception as e:
            print(f"✗ Failed to copy {src.name}: {e}")
    
    print(f"\n✓ Organized {copied}/{len(samples_to_copy)} images successfully")
    
    # Create metadata file
    create_metadata(output_path, samples_to_copy)
    
    # Create dataset statistics
    create_statistics(normal_path, pneumonia_path, output_path)
    
    return output_path


def create_metadata(output_path, samples):
    """Create metadata file for the samples."""
    metadata_file = output_path / "metadata.txt"
    
    with open(metadata_file, 'w') as f:
        f.write("Real Chest X-Ray Dataset - Sample Images\n")
        f.write("=" * 60 + "\n\n")
        f.write("Source: Kaggle Chest X-Ray Images (Pneumonia) Dataset\n")
        f.write("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n\n")
        f.write("Classes:\n")
        f.write("  - NORMAL: Healthy chest X-rays\n")
        f.write("  - PNEUMONIA: X-rays showing pneumonia (bacterial or viral)\n\n")
        f.write("Sample Images:\n")
        f.write("-" * 60 + "\n\n")
        
        for src, dst, description in samples:
            f.write(f"File: {dst.name}\n")
            f.write(f"Original: {src.name}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Size: {src.stat().st_size / 1024:.1f} KB\n")
            f.write("\n")
    
    print(f"\n✓ Metadata saved to {metadata_file}")


def create_statistics(normal_path, pneumonia_path, output_path):
    """Create dataset statistics file."""
    stats_file = output_path / "dataset_statistics.txt"
    
    # Count images
    normal_images = list(normal_path.glob("*.jpeg")) + list(normal_path.glob("*.jpg"))
    pneumonia_images = list(pneumonia_path.glob("*.jpeg")) + list(pneumonia_path.glob("*.jpg"))
    
    # Categorize pneumonia images
    bacteria_count = len([img for img in pneumonia_images if 'bacteria' in img.name])
    virus_count = len([img for img in pneumonia_images if 'virus' in img.name])
    
    # Calculate sizes
    normal_sizes = [img.stat().st_size for img in normal_images]
    pneumonia_sizes = [img.stat().st_size for img in pneumonia_images]
    
    with open(stats_file, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Image Counts:\n")
        f.write(f"  Normal X-rays:        {len(normal_images):>6}\n")
        f.write(f"  Pneumonia X-rays:     {len(pneumonia_images):>6}\n")
        f.write(f"    - Bacterial:        {bacteria_count:>6}\n")
        f.write(f"    - Viral:            {virus_count:>6}\n")
        f.write(f"  Total:                {len(normal_images) + len(pneumonia_images):>6}\n\n")
        
        f.write("Class Distribution:\n")
        total = len(normal_images) + len(pneumonia_images)
        f.write(f"  Normal:               {len(normal_images)/total*100:>5.1f}%\n")
        f.write(f"  Pneumonia:            {len(pneumonia_images)/total*100:>5.1f}%\n\n")
        
        f.write("Image Sizes:\n")
        f.write(f"  Normal (avg):         {sum(normal_sizes)/len(normal_sizes)/1024:>6.1f} KB\n")
        f.write(f"  Pneumonia (avg):      {sum(pneumonia_sizes)/len(pneumonia_sizes)/1024:>6.1f} KB\n\n")
        
        f.write("Sample Directory:\n")
        f.write(f"  Location:             {output_path}\n")
        sample_images = list(output_path.glob("*.jpeg")) + list(output_path.glob("*.jpg"))
        f.write(f"  Sample count:         {len(sample_images)}\n")
    
    print(f"✓ Statistics saved to {stats_file}")


def create_train_test_split(
    normal_dir="data/NORMAL",
    pneumonia_dir="data/PNEUMONIA",
    output_base="data/splits",
    train_ratio=0.8
):
    """
    Create train/test split for model training.
    
    Args:
        normal_dir: Directory containing normal X-rays
        pneumonia_dir: Directory containing pneumonia X-rays
        output_base: Base directory for train/test splits
        train_ratio: Ratio of training data (0.8 = 80% train, 20% test)
    """
    import random
    
    output_path = Path(output_base)
    
    # Create directories
    train_normal = output_path / "train" / "NORMAL"
    train_pneumonia = output_path / "train" / "PNEUMONIA"
    test_normal = output_path / "test" / "NORMAL"
    test_pneumonia = output_path / "test" / "PNEUMONIA"
    
    for dir_path in [train_normal, train_pneumonia, test_normal, test_pneumonia]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Creating Train/Test Split")
    print("=" * 60)
    
    # Get all images
    normal_images = list(Path(normal_dir).glob("*.jpeg")) + list(Path(normal_dir).glob("*.jpg"))
    pneumonia_images = list(Path(pneumonia_dir).glob("*.jpeg")) + list(Path(pneumonia_dir).glob("*.jpg"))
    
    # Shuffle
    random.seed(42)
    random.shuffle(normal_images)
    random.shuffle(pneumonia_images)
    
    # Split
    normal_split = int(len(normal_images) * train_ratio)
    pneumonia_split = int(len(pneumonia_images) * train_ratio)
    
    splits = [
        (normal_images[:normal_split], train_normal, "Normal (train)"),
        (normal_images[normal_split:], test_normal, "Normal (test)"),
        (pneumonia_images[:pneumonia_split], train_pneumonia, "Pneumonia (train)"),
        (pneumonia_images[pneumonia_split:], test_pneumonia, "Pneumonia (test)"),
    ]
    
    # Copy files
    for images, dest_dir, description in splits:
        print(f"\nCopying {len(images)} images to {dest_dir}...")
        for img in images:
            try:
                shutil.copy2(img, dest_dir / img.name)
            except Exception as e:
                print(f"✗ Failed to copy {img.name}: {e}")
        print(f"✓ {description}: {len(images)} images")
    
    print(f"\n✓ Train/test split created in {output_base}")
    print(f"  Train: {normal_split + pneumonia_split} images")
    print(f"  Test:  {len(normal_images) - normal_split + len(pneumonia_images) - pneumonia_split} images")


def main():
    """Main function."""
    print("\nX-Ray Dataset Preparation\n")
    
    # Check if directories exist
    if not Path("data/NORMAL").exists():
        print("✗ Error: data/NORMAL directory not found")
        return False
    
    if not Path("data/PNEUMONIA").exists():
        print("✗ Error: data/PNEUMONIA directory not found")
        return False
    
    # Organize sample images
    organize_xray_data()
    
    # Optionally create train/test split (can be run separately)
    print("\n" + "=" * 60)
    print("To create train/test split for model training, run:")
    print("  python scripts/prepare_data.py --split")
    
    import sys
    if '--split' in sys.argv:
        create_train_test_split()
    else:
        print("Skipping train/test split creation (use --split flag to create)")
    
    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review sample images in data/sample_xrays/")
    print("2. Run: python models/xray_model.py")
    print("3. Run: python scripts/test_inference.py")
    print("4. Run: streamlit run app/streamlit_app.py")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

