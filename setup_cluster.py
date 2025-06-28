#!/usr/bin/env python3
"""
UC Merced Cluster Setup Script for VideoLLaMA3 Blind Navigation
Automatically configures the environment and validates setup
"""

import os
import sys
import subprocess
import torch
import pkg_resources
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return None, e.stderr

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'torch', 'transformers', 'accelerate', 'peft', 
        'opencv-python', 'datasets', 'tensorboard'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ {package}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    return missing_packages

def check_gpu():
    """Check GPU availability and CUDA setup"""
    print("🖥️ Checking GPU setup...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {gpu_count} GPU(s)")
        print(f"✅ Primary GPU: {gpu_name}")
        
        # Check VRAM
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Memory: {total_mem:.1f} GB")
        
        if total_mem < 8:
            print("⚠️ Warning: GPU has less than 8GB VRAM. Consider using smaller batch sizes.")
        
        return True
    else:
        print("❌ CUDA not available. Training will be very slow on CPU.")
        return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        'logs',
        'trained_models', 
        'data/videos',
        'data/annotations',
        'cache'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def validate_cluster_modules():
    """Check if cluster modules are loaded"""
    print("🔧 Checking cluster modules...")
    
    # Check if we're on a SLURM cluster
    if 'SLURM_JOB_ID' in os.environ:
        print(f"✅ Running on SLURM cluster (Job ID: {os.environ['SLURM_JOB_ID']})")
    else:
        print("ℹ️ Not running under SLURM (development/interactive mode)")
    
    # Check CUDA module
    cuda_version = os.environ.get('CUDA_VERSION', 'Not found')
    print(f"CUDA Version: {cuda_version}")

def test_model_loading():
    """Test if we can load the base model"""
    print("🧪 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        print("✅ Tokenizer loaded successfully")
        
        # Test model loading (CPU only for validation)
        print("Testing model loading (CPU)...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        print("✅ Base model loads successfully")
        
        # Clean up memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def create_test_script():
    """Create a test script for quick validation"""
    test_script = """#!/usr/bin/env python3
# Quick test script for VideoLLaMA3 setup

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print("✅ Transformers working correctly")
except Exception as e:
    print(f"❌ Transformers issue: {e}")

try:
    import cv2
    print("✅ OpenCV available")
except Exception as e:
    print(f"❌ OpenCV issue: {e}")

print("Setup validation complete!")
"""
    
    with open('test_setup.py', 'w') as f:
        f.write(test_script)
    
    os.chmod('test_setup.py', 0o755)
    print("✅ Created test_setup.py")

def main():
    """Main setup function"""
    print("🚀 VideoLLaMA3 Blind Navigation - UC Merced Cluster Setup")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Requires Python 3.8+")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Run all checks
    missing_packages = check_requirements()
    gpu_available = check_gpu()
    validate_cluster_modules()
    setup_directories()
    create_test_script()
    
    print("\n" + "=" * 60)
    print("📋 Setup Summary:")
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements_cluster.txt")
    else:
        print("✅ All required packages installed")
    
    if gpu_available:
        print("✅ GPU setup verified")
    else:
        print("⚠️ GPU not available - training will be slow")
    
    # Test model loading
    model_test = test_model_loading()
    if model_test:
        print("✅ Model loading test passed")
    else:
        print("❌ Model loading test failed")
    
    print("\n🎯 Next Steps:")
    print("1. Run: python test_setup.py (to validate)")
    print("2. Submit training job: sbatch scripts/train_cluster.sh")
    print("3. Monitor: squeue -u $USER")
    print("4. Check logs: tail -f logs/train_*.out")
    
    print("\n✅ Setup complete! Ready for training on UC Merced cluster.")

if __name__ == "__main__":
    main()