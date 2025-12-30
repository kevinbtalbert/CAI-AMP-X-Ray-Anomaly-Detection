"""
Deploy X-ray anomaly detection model to Cloudera ML using the Models API.
Reference: https://docs.cloudera.com/machine-learning/1.5.4/rest-api-reference/index.html
"""
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import mlflow

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xray_model import save_model


class ClouderaMLClient:
    """Client for Cloudera ML REST API v2."""
    
    def __init__(self, api_url: str, api_key: str, project_id: str):
        """
        Initialize Cloudera ML client.
        
        Args:
            api_url: Base URL for CML API (e.g., https://ml.example.com)
            api_key: API key for authentication
            project_id: Project ID where model will be deployed
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.project_id = project_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_model(
        self,
        name: str,
        description: str = "",
        disable_authentication: bool = False
    ) -> dict:
        """
        Create a new model in Cloudera ML.
        
        Args:
            name: Model name
            description: Model description
            disable_authentication: Whether to disable auth for the endpoint
            
        Returns:
            Model creation response
        """
        url = f"{self.api_url}/api/v2/projects/{self.project_id}/models"
        
        payload = {
            "project_id": self.project_id,
            "name": name,
            "description": description,
            "disable_authentication": disable_authentication
        }
        
        print(f"Creating model: {name}")
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 201:
            print(f"✓ Model created successfully")
            return response.json()
        else:
            print(f"✗ Failed to create model: {response.status_code}")
            print(f"Response: {response.text}")
            response.raise_for_status()
    
    def create_model_build(
        self,
        model_id: str,
        file_path: str,
        function_name: str = "predict",
        kernel: str = "python3",
        runtime_identifier: str = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-standard:2023.05.2-b7"
    ) -> dict:
        """
        Create a model build (register model version).
        
        Args:
            model_id: Model ID
            file_path: Path to model file/directory
            function_name: Prediction function name
            kernel: Kernel type
            runtime_identifier: Runtime image identifier
            
        Returns:
            Build creation response
        """
        url = f"{self.api_url}/api/v2/projects/{self.project_id}/models/{model_id}/builds"
        
        payload = {
            "project_id": self.project_id,
            "model_id": model_id,
            "file_path": file_path,
            "function_name": function_name,
            "kernel": kernel,
            "runtime_identifier": runtime_identifier
        }
        
        print(f"Creating model build...")
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 201:
            print(f"✓ Model build created successfully")
            return response.json()
        else:
            print(f"✗ Failed to create model build: {response.status_code}")
            print(f"Response: {response.text}")
            response.raise_for_status()
    
    def create_model_deployment(
        self,
        model_id: str,
        build_id: str,
        cpu: float = 2.0,
        memory: float = 4.0,
        gpu: int = 1,
        replicas: int = 1
    ) -> dict:
        """
        Deploy a model build.
        
        Args:
            model_id: Model ID
            build_id: Build ID to deploy
            cpu: CPU cores
            memory: Memory in GB
            gpu: Number of GPUs
            replicas: Number of replicas
            
        Returns:
            Deployment response
        """
        url = f"{self.api_url}/api/v2/projects/{self.project_id}/models/{model_id}/builds/{build_id}/deployments"
        
        payload = {
            "project_id": self.project_id,
            "model_id": model_id,
            "build_id": build_id,
            "cpu": cpu,
            "memory": memory,
            "gpu": gpu,
            "replicas": replicas
        }
        
        print(f"Deploying model...")
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 201:
            print(f"✓ Model deployed successfully")
            return response.json()
        else:
            print(f"✗ Failed to deploy model: {response.status_code}")
            print(f"Response: {response.text}")
            response.raise_for_status()
    
    def list_models(self) -> dict:
        """List all models in the project."""
        url = f"{self.api_url}/api/v2/projects/{self.project_id}/models"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to list models: {response.status_code}")
            response.raise_for_status()
    
    def get_model(self, model_id: str) -> dict:
        """Get model details."""
        url = f"{self.api_url}/api/v2/projects/{self.project_id}/models/{model_id}"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get model: {response.status_code}")
            response.raise_for_status()


def deploy_xray_model():
    """
    Main deployment function.
    Saves MLflow model and deploys to Cloudera ML.
    """
    # Load environment variables
    load_dotenv()
    
    api_url = os.getenv("CML_API_URL")
    api_key = os.getenv("CML_API_KEY")
    project_id = os.getenv("PROJECT_ID")
    model_name = os.getenv("MODEL_NAME", "xray-anomaly-detector")
    model_description = os.getenv(
        "MODEL_DESCRIPTION",
        "X-ray anomaly detection with visual localization using Grad-CAM"
    )
    
    # Validate environment
    if not all([api_url, api_key, project_id]):
        print("Error: Missing required environment variables")
        print("Please set: CML_API_URL, CML_API_KEY, PROJECT_ID")
        print("Copy .env.example to .env and fill in your values")
        return False
    
    print("=" * 60)
    print("X-Ray Anomaly Detection Model Deployment")
    print("=" * 60)
    
    # Step 1: Save MLflow model
    print("\n[1/4] Saving MLflow model...")
    model_path = "xray_model"
    save_model(model_path)
    
    # Step 2: Initialize Cloudera ML client
    print("\n[2/4] Connecting to Cloudera ML...")
    client = ClouderaMLClient(api_url, api_key, project_id)
    
    # Check if model already exists
    print("\nChecking for existing models...")
    try:
        models = client.list_models()
        existing_model = None
        
        if "models" in models:
            for model in models["models"]:
                if model.get("name") == model_name:
                    existing_model = model
                    print(f"Found existing model: {model_name} (ID: {model['id']})")
                    break
        
        # Step 3: Create model if it doesn't exist
        if not existing_model:
            print("\n[3/4] Creating new model...")
            model_response = client.create_model(
                name=model_name,
                description=model_description,
                disable_authentication=False
            )
            model_id = model_response.get("id")
        else:
            print("\n[3/4] Using existing model...")
            model_id = existing_model.get("id")
        
        print(f"Model ID: {model_id}")
        
        # Step 4: Create model build
        print("\n[4/4] Creating model build...")
        build_response = client.create_model_build(
            model_id=model_id,
            file_path=model_path,
            function_name="predict"
        )
        build_id = build_response.get("id")
        print(f"Build ID: {build_id}")
        
        # Step 5: Deploy model
        print("\nDeploying model...")
        deployment_response = client.create_model_deployment(
            model_id=model_id,
            build_id=build_id,
            cpu=8.0,
            memory=32.0,
            gpu=1,
            replicas=1
        )
        
        print("\n" + "=" * 60)
        print("✓ Deployment Complete!")
        print("=" * 60)
        print(f"\nModel Name: {model_name}")
        print(f"Model ID: {model_id}")
        print(f"Build ID: {build_id}")
        
        if "access_key" in deployment_response:
            print(f"Access Key: {deployment_response['access_key']}")
        
        print("\nYou can now use the model endpoint for inference.")
        print("See scripts/test_inference.py for example usage.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = deploy_xray_model()
    sys.exit(0 if success else 1)

