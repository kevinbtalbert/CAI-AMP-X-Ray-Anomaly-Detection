"""
Test inference against deployed X-ray anomaly detection model.
"""
import os
import sys
import json
import requests
import base64
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io


def test_local_inference():
    """Test model locally using MLflow."""
    print("Testing local inference...")
    
    import mlflow
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load model
    model_path = "xray_model"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Run: python models/xray_model.py to create the model first")
        return False
    
    print(f"Loading model from {model_path}...")
    model = mlflow.pyfunc.load_model(model_path)
    
    # Test with sample image
    sample_dir = Path("data/sample_xrays")
    if not sample_dir.exists():
        print("Sample images not found. Run: python scripts/download_samples.py")
        return False
    
    sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not sample_images:
        print("No sample images found")
        return False
    
    test_image = str(sample_images[0])
    print(f"Testing with image: {test_image}")
    
    # Run inference
    result = model.predict({"image": test_image})
    
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    
    if result and len(result) > 0:
        res = result[0]
        
        print(f"\nAnomaly Score: {res['anomaly_score']:.3f}")
        
        print("\nTop Findings:")
        for finding in res['top_findings'][:3]:
            print(f"  - {finding['finding']}: {finding['confidence']:.3f}")
        
        print(f"\nBounding Boxes Detected: {len(res['bounding_boxes'])}")
        for i, box in enumerate(res['bounding_boxes'][:3]):
            print(f"  Box {i+1}: x={box['x']}, y={box['y']}, w={box['width']}, h={box['height']}")
        
        # Save heatmap
        if 'heatmap_base64' in res:
            heatmap_data = base64.b64decode(res['heatmap_base64'])
            output_path = "test_heatmap_output.png"
            with open(output_path, 'wb') as f:
                f.write(heatmap_data)
            print(f"\nHeatmap saved to: {output_path}")
        
        print("\n✓ Local inference successful!")
        return True
    else:
        print("✗ No results returned")
        return False


def test_endpoint_inference(endpoint_url: str, access_key: str, image_path: str):
    """
    Test inference against deployed Cloudera ML endpoint.
    
    Args:
        endpoint_url: Model endpoint URL
        access_key: Model access key
        image_path: Path to test image
    """
    print(f"Testing endpoint: {endpoint_url}")
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare request
    headers = {
        "Content-Type": "application/json"
    }
    
    if access_key:
        headers["Authorization"] = f"Bearer {access_key}"
    
    payload = {
        "dataframe_split": {
            "columns": ["image"],
            "data": [[image_b64]]
        }
    }
    
    # Send request
    print("Sending inference request...")
    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n" + "=" * 60)
        print("Endpoint Inference Results")
        print("=" * 60)
        
        if "predictions" in result and len(result["predictions"]) > 0:
            res = result["predictions"][0]
            
            print(f"\nAnomaly Score: {res['anomaly_score']:.3f}")
            
            print("\nTop Findings:")
            for finding in res['top_findings'][:3]:
                print(f"  - {finding['finding']}: {finding['confidence']:.3f}")
            
            print(f"\nBounding Boxes: {len(res['bounding_boxes'])}")
            
            print("\n✓ Endpoint inference successful!")
            return True
        else:
            print("✗ Unexpected response format")
            print(json.dumps(result, indent=2))
            return False
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(response.text)
        return False


def main():
    """Main test function."""
    load_dotenv()
    
    print("=" * 60)
    print("X-Ray Anomaly Detection - Inference Test")
    print("=" * 60)
    
    # Test local inference first
    print("\n[1] Testing Local Inference")
    print("-" * 60)
    local_success = test_local_inference()
    
    # Test endpoint if configured
    endpoint_url = os.getenv("MODEL_ENDPOINT_URL")
    access_key = os.getenv("MODEL_ACCESS_KEY")
    
    if endpoint_url:
        print("\n[2] Testing Endpoint Inference")
        print("-" * 60)
        
        # Get test image
        sample_dir = Path("data/sample_xrays")
        sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        
        if sample_images:
            test_image = str(sample_images[0])
            endpoint_success = test_endpoint_inference(endpoint_url, access_key, test_image)
        else:
            print("No sample images found for endpoint testing")
            endpoint_success = False
    else:
        print("\n[2] Endpoint Testing Skipped")
        print("-" * 60)
        print("Set MODEL_ENDPOINT_URL in .env to test endpoint inference")
        endpoint_success = None
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Local Inference: {'✓ PASS' if local_success else '✗ FAIL'}")
    if endpoint_success is not None:
        print(f"Endpoint Inference: {'✓ PASS' if endpoint_success else '✗ FAIL'}")
    
    return local_success and (endpoint_success is None or endpoint_success)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

