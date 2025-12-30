"""
Setup script for X-ray anomaly detection project.
Installs dependencies and prepares the environment.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(e.stderr)
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("X-Ray Anomaly Detection - Setup")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    # Install dependencies
    if not run_command(
        "pip install -r requirements.txt",
        "Installing dependencies"
    ):
        return False
    
    # Download sample images
    if not run_command(
        "python scripts/download_samples.py",
        "Downloading sample X-ray images"
    ):
        print("⚠️  Sample download failed, but continuing...")
    
    # Create model
    if not run_command(
        "python models/xray_model.py",
        "Creating MLflow model"
    ):
        return False
    
    # Create .env if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("\n✓ Created .env file from template")
            print("⚠️  Please edit .env with your Cloudera ML credentials")
    
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env with your Cloudera ML credentials")
    print("2. Run: streamlit run app/streamlit_app.py")
    print("3. Or deploy: python scripts/deploy_model.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

