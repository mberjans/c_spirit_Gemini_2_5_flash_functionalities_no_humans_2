# Installation Guide

This guide provides comprehensive installation instructions for the C-Spirit project, an AI-driven ontology development and information extraction system for plant metabolomics research.

## Table of Contents

- [System Requirements](#system-requirements)
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Development Environment Setup](#development-environment-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Optional Dependencies](#optional-dependencies)
- [Additional Non-Python Dependencies](#additional-non-python-dependencies)
- [Environment Configuration](#environment-configuration)

## System Requirements

### Operating System
- **Linux**: Ubuntu 20.04+ or equivalent
- **macOS**: 10.15+ (Catalina or later)
- **Windows**: Windows 10+ with WSL2 recommended

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+ (for LLM processing)
- **Storage**: Minimum 10GB free space
- **CPU**: Multi-core processor recommended for ontology processing

## Prerequisites

### 1. Python Environment
- **Python Version**: 3.9.x (strictly required, not compatible with 3.10+)
- **Package Manager**: Poetry (recommended) or pip

### 2. System Dependencies

#### For macOS:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.9 git curl
```

#### For Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install python3.9 python3.9-dev python3.9-venv python3-pip git curl build-essential

# Install additional system libraries for PDF processing
sudo apt install libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig
```

#### For Windows (WSL2 recommended):
1. Install WSL2 with Ubuntu 20.04+
2. Follow Ubuntu installation steps above

### 3. Java Development Kit (JDK) - Required for Owlready2 Reasoners

The C-Spirit project uses Owlready2 for ontology management and reasoning. Owlready2's reasoning capabilities rely on Java-based reasoners like **HermiT** and **Pellet**, which require a Java Development Kit (JDK) to function properly.

#### Version Requirements
- **Minimum**: Java 5 (JDK 1.5) or higher
- **Recommended**: Java 8 (JDK 1.8) minimum
- **Production**: Java 11+ recommended for optimal performance and security
- **Maximum Tested**: Java 21 (latest LTS version)

#### Recommended JDK Distributions
- **Eclipse Adoptium** (formerly AdoptOpenJDK) - Recommended
- **Oracle OpenJDK** - Free and open source
- **Amazon Corretto** - Long-term support with performance optimizations
- **Azul Zulu** - Enterprise-grade with extensive testing

#### Platform-Specific Installation

##### macOS
```bash
# Option 1: Homebrew (Recommended - Eclipse Adoptium)
brew install --cask temurin

# Option 2: Homebrew OpenJDK
brew install openjdk@11

# Option 3: Oracle JDK (manual download from Oracle website required)
# Download from: https://www.oracle.com/java/technologies/downloads/

# Link the JDK for system-wide access (Homebrew OpenJDK only)
sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk
```

##### Linux (Ubuntu/Debian)
```bash
# Option 1: Eclipse Adoptium (Recommended)
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | sudo apt-key add -
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | sudo tee /etc/apt/sources.list.d/adoptium.list
sudo apt update
sudo apt install temurin-11-jdk

# Option 2: OpenJDK from default repository
sudo apt update
sudo apt install openjdk-11-jdk

# Option 3: Amazon Corretto
wget -O- https://apt.corretto.aws/corretto.key | sudo apt-key add -
echo "deb https://apt.corretto.aws stable main" | sudo tee /etc/apt/sources.list.d/corretto.list
sudo apt update
sudo apt install java-11-amazon-corretto-jdk
```

##### Linux (RHEL/CentOS/Fedora)
```bash
# RHEL/CentOS (yum)
sudo yum install java-11-openjdk-devel

# Fedora (dnf)
sudo dnf install java-11-openjdk-devel

# Amazon Corretto
sudo rpm --import https://yum.corretto.aws/corretto.key
sudo curl -L -o /etc/yum.repos.d/corretto.repo https://yum.corretto.aws/corretto.repo
sudo yum install java-11-amazon-corretto-devel
```

##### Windows
```powershell
# Option 1: Chocolatey (Recommended)
choco install adoptopenjdk11

# Option 2: Scoop
scoop bucket add java
scoop install adopt11-hotspot

# Option 3: Manual Installation
# Download from Eclipse Adoptium: https://adoptium.net/releases.html
# Or Oracle: https://www.oracle.com/java/technologies/downloads/
```

#### Verification and Configuration

##### Verify Installation
```bash
# Check Java version
java -version

# Check Java compiler (should show same version)
javac -version

# Check JAVA_HOME environment variable
echo $JAVA_HOME

# List all installed Java versions (Linux/macOS)
# Ubuntu/Debian
update-java-alternatives --list

# macOS
/usr/libexec/java_home -V
```

##### Expected Output
```
java version "11.0.19" 2023-04-18 LTS
Java(TM) SE Runtime Environment (build 11.0.19+7-LTS)
Java HotSpot(TM) 64-Bit Server VM (build 11.0.19+7-LTS, mixed mode)
```

#### Owlready2 Configuration

##### Automatic Java Detection
Owlready2 typically auto-detects Java installations on Linux and macOS. For most users, no additional configuration is needed after JDK installation.

##### Manual Java Path Configuration (if needed)
If Owlready2 cannot find Java automatically, configure it manually:

```python
# In your Python code, before importing Owlready2
import os
import owlready2

# Set Java path explicitly (adjust path as needed)
# Windows
# owlready2.JAVA_EXE = r"C:\Program Files\Java\jdk-11.0.19\bin\java.exe"

# macOS (Homebrew)
# owlready2.JAVA_EXE = "/opt/homebrew/bin/java"

# Linux
# owlready2.JAVA_EXE = "/usr/bin/java"

# Alternative: Set JAVA_HOME environment variable
# os.environ['JAVA_HOME'] = '/path/to/your/jdk'
```

##### Environment Variables
Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Linux/macOS
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"  # Adjust path
export PATH="$JAVA_HOME/bin:$PATH"

# macOS (Homebrew)
export JAVA_HOME="/opt/homebrew/opt/openjdk@11"
export PATH="$JAVA_HOME/bin:$PATH"
```

Windows (System Environment Variables):
```
JAVA_HOME=C:\Program Files\Java\jdk-11.0.19
PATH=%JAVA_HOME%\bin;%PATH%
```

#### Testing Owlready2 Reasoners

Verify that Owlready2 can use Java-based reasoners:

```python
# Test script - save as test_reasoners.py
"""Test Owlready2 reasoners with Java."""

import owlready2 as owl2

def test_hermit_reasoner():
    """Test HermiT reasoner."""
    try:
        # Create a simple ontology
        onto = owl2.get_ontology("http://test.org/onto.owl")
        
        with onto:
            class Person(owl2.Thing): pass
            class hasAge(owl2.DataProperty):
                domain = [Person]
                range = [int]
        
        # Try to sync with HermiT reasoner
        with onto:
            owl2.sync_reasoner_hermit([onto])
        
        print("✅ HermiT reasoner working correctly")
        return True
        
    except Exception as e:
        print(f"❌ HermiT reasoner failed: {e}")
        return False

def test_pellet_reasoner():
    """Test Pellet reasoner."""
    try:
        # Create a simple ontology
        onto = owl2.get_ontology("http://test.org/onto2.owl")
        
        with onto:
            class Animal(owl2.Thing): pass
            class Dog(Animal): pass
        
        # Try to sync with Pellet reasoner
        with onto:
            owl2.sync_reasoner_pellet([onto])
        
        print("✅ Pellet reasoner working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Pellet reasoner failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Owlready2 reasoners...")
    hermit_ok = test_hermit_reasoner()
    pellet_ok = test_pellet_reasoner()
    
    if hermit_ok and pellet_ok:
        print("🎉 All reasoners working correctly!")
    else:
        print("⚠️  Some reasoners failed. Check Java installation.")
```

Run the test:
```bash
python test_reasoners.py
```

#### Troubleshooting Java/Owlready2 Issues

##### Common Problems and Solutions

**Problem**: `java.exe not found` or `Java not found`
```bash
# Solution 1: Verify Java is in PATH
which java  # Linux/macOS
where java  # Windows

# Solution 2: Set JAVA_HOME explicitly
export JAVA_HOME="/path/to/your/jdk"
export PATH="$JAVA_HOME/bin:$PATH"

# Solution 3: Configure Owlready2 directly
# In Python:
import owlready2
owlready2.JAVA_EXE = "/path/to/java"
```

**Problem**: `OutofMemoryError` during reasoning
```python
# Solution: Increase Java heap size
import owlready2

# Set JVM arguments before first use
owlready2.JAVA_ARGS.extend(["-Xmx4g", "-Xms1g"])  # 4GB max, 1GB initial
```

**Problem**: `ClassNotFoundException` for reasoners
```bash
# Solution: Verify Owlready2 installation includes reasoner JARs
pip install --force-reinstall owlready2

# Check if JAR files exist
python -c "import owlready2; print(owlready2.__file__)"
# Look for .jar files in the Owlready2 installation directory
```

**Problem**: Permission errors on macOS Catalina+
```bash
# Solution: Grant permission to Java executable
# System Preferences > Security & Privacy > Privacy > Developer Tools
# Add Terminal or your IDE to the list

# Or use signed JDK distributions like Eclipse Adoptium
brew install --cask temurin
```

**Problem**: Reasoner hangs or takes too long
```python
# Solution: Set timeout for reasoning operations
import owlready2

# Configure reasoning timeout (in seconds)
owlready2.JAVA_ARGS.extend(["-Dnet.sourceforge.owlapi.util.TimerUtils.timeout=30"])
```

##### Performance Optimization

For better reasoning performance:

```python
# Optimize JVM settings for reasoning
import owlready2

# Set optimal JVM arguments
owlready2.JAVA_ARGS.extend([
    "-Xmx8g",           # Maximum heap size (adjust based on available RAM)
    "-Xms2g",           # Initial heap size
    "-XX:+UseG1GC",     # Use G1 garbage collector
    "-XX:+UseStringDeduplication",  # Reduce memory usage
    "-server"           # Server mode for better performance
])
```

#### Version Management

If you need multiple Java versions:

```bash
# Linux: Use update-alternatives
sudo update-alternatives --config java

# macOS: Use jenv
brew install jenv
jenv add /Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home
jenv global 11.0

# Windows: Use multiple installations and update PATH as needed
```

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/c-spirit.git
cd c-spirit
```

### Step 2: Set Up Python Virtual Environment

#### Option A: Using venv (Built-in Python)
```bash
# Create virtual environment with Python 3.9
python3.9 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (WSL):
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Option B: Using Poetry (Recommended)
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (follow the instructions from the installer)
export PATH="$HOME/.local/bin:$PATH"

# Configure Poetry to use Python 3.9
poetry env use python3.9

# Install dependencies
poetry install

# Activate Poetry shell
poetry shell
```

### Step 3: Install Core Dependencies

#### With Poetry (Recommended):
```bash
# Install production dependencies
poetry install

# Install with development dependencies
poetry install --with dev
```

#### With pip and venv:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install from pyproject.toml
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Step 4: Install Additional System-Specific Dependencies

#### For PDF Processing (PyMuPDF):
```bash
# This should be installed automatically, but if issues occur:
# macOS
brew install mupdf

# Ubuntu/Debian
sudo apt install libmupdf-dev

# Then reinstall PyMuPDF
pip install --force-reinstall PyMuPDF
```

#### For Fuzzy String Matching:
```bash
# Install python-Levenshtein for better performance
# This should be automatic, but if compilation fails:

# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake build-essential

# Reinstall if needed
pip install --force-reinstall python-Levenshtein
```

## Development Environment Setup

### 1. Install Development Tools

```bash
# With Poetry
poetry install --with dev

# With pip
pip install -e ".[dev]"
```

### 2. Set Up Pre-commit Hooks (Optional but Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Test the hooks
pre-commit run --all-files
```

### 3. Configure Code Quality Tools

The project uses Ruff for linting and Black for formatting. Configuration is in `pyproject.toml`.

```bash
# Run linting
ruff check src/

# Run formatting
black src/

# Run type checking (if mypy is installed)
mypy src/
```

## Verification

### 1. Test Basic Installation

```bash
# Run basic dependency test
python -c "import owlready2, Bio, fitz, fuzzywuzzy; print('Core dependencies installed successfully')"
```

### 2. Run Test Suite

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest tests/test_dependencies.py -v
pytest tests/test_code_quality.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### 3. Verify Key Functionality

```python
# Test script - save as test_installation.py
"""Test basic C-Spirit functionality."""

def test_imports():
    """Test that all key modules can be imported."""
    try:
        import owlready2
        import Bio
        import fitz  # PyMuPDF
        import fuzzywuzzy
        from Levenshtein import distance
        print("✅ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_versions():
    """Check versions of key dependencies."""
    import owlready2
    import Bio
    import fitz
    import fuzzywuzzy
    
    print(f"Owlready2: {owlready2.__version__}")
    print(f"Biopython: {Bio.__version__}")
    print(f"PyMuPDF: {fitz.__version__}")
    print(f"FuzzyWuzzy: {fuzzywuzzy.__version__}")

if __name__ == "__main__":
    if test_imports():
        test_versions()
        print("🎉 Installation verification successful!")
    else:
        print("⚠️  Installation verification failed!")
```

Run the verification:
```bash
python test_installation.py
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Conflicts
**Problem**: Wrong Python version or conflicts between versions
```bash
# Solution: Explicitly use Python 3.9
python3.9 -m venv venv
# or with Poetry
poetry env use python3.9
```

#### 2. PyMuPDF Installation Issues
**Problem**: Compilation errors when installing PyMuPDF
```bash
# Solution: Install system dependencies first
# macOS
brew install mupdf cmake

# Ubuntu/Debian
sudo apt install libmupdf-dev cmake build-essential

# Then reinstall
pip install --no-cache-dir --force-reinstall PyMuPDF
```

#### 3. Levenshtein Compilation Issues
**Problem**: C extension compilation fails
```bash
# Solution: Install development tools
# macOS
xcode-select --install
brew install cmake

# Ubuntu/Debian
sudo apt install build-essential cmake python3.9-dev

# Use alternative if compilation still fails
pip install fuzzywuzzy[speedup]
```

#### 4. Permission Errors
**Problem**: Permission denied during installation
```bash
# Solution: Use virtual environment (don't use sudo with pip)
python3.9 -m venv venv
source venv/bin/activate
pip install -e .
```

#### 5. Poetry Not Found
**Problem**: Poetry command not found after installation
```bash
# Solution: Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Or use the full path temporarily
$HOME/.local/bin/poetry install
```

#### 6. Virtual Environment Issues
**Problem**: Virtual environment activation fails
```bash
# Solution: Recreate virtual environment
rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Memory Issues

If you encounter memory issues during installation or testing:

```bash
# Increase pip's memory limit
pip install --no-cache-dir -e .

# Install dependencies one by one if needed
pip install owlready2==0.36
pip install "biopython>=1.84,<2.0"
pip install "PyMuPDF>=1.26.0,<2.0"
```

### Testing Installation Issues

```bash
# Clear pytest cache if tests fail
pytest --cache-clear

# Run tests in isolation
pytest -x  # Stop on first failure
pytest --tb=short  # Shorter tracebacks
```

## Optional Dependencies

### 1. Ollama (For Local LLM Support)

The C-Spirit project leverages Large Language Models (LLMs) for various AI-driven tasks including named entity recognition, relationship extraction, and text processing. While cloud-based models provide excellent capabilities, **Ollama** enables you to run powerful LLMs locally, offering several advantages:

- **Complete Privacy**: All data processing happens locally, ensuring sensitive research data never leaves your system
- **No Usage Limits**: Run unlimited inference without API costs or rate limits
- **Offline Functionality**: Continue working even without internet connectivity
- **Custom Models**: Use specialized or fine-tuned models for plant metabolomics research
- **Faster Inference**: Eliminate network latency for real-time processing

#### What is Ollama?

Ollama is an open-source platform that makes it easy to run large language models locally on your machine. It provides a simple command-line interface for downloading, managing, and running various LLM models including Llama, Gemma, DeepSeek, and many others.

#### System Requirements

##### Minimum Hardware Requirements
- **RAM**: 8GB minimum (for 7B parameter models)
- **Storage**: 10GB+ free space per model
- **CPU**: Modern multi-core processor (Intel Core i5 or equivalent)
- **Network**: Internet connection for initial model downloads

##### Recommended Hardware Requirements
- **RAM**: 16GB+ (for 13B parameter models), 32GB+ (for 33B parameter models)
- **Storage**: SSD with 50GB+ free space for multiple models
- **CPU**: High-performance multi-core processor (Intel Core i7/i9 or AMD Ryzen 7/9)
- **GPU**: Optional but highly recommended for faster inference

##### GPU Acceleration (Optional but Recommended)
- **NVIDIA**: RTX 3060 or newer with 8GB+ VRAM, CUDA 11.8+ drivers
- **AMD**: RX 6600 XT or newer (Linux only, experimental support)
- **Apple Silicon**: M1/M2/M3/M4 chips with 16GB+ unified memory (automatic acceleration)

#### Operating System Support

##### macOS Requirements
- **Version**: macOS 12 Monterey or later
- **Architecture**: Intel x86_64 or Apple Silicon (M1/M2/M3/M4)
- **Memory**: 8GB RAM minimum, 16GB+ recommended

##### Linux Requirements  
- **Distributions**: Ubuntu 20.04+, Debian 11+, CentOS 8+, Fedora 35+, or equivalent
- **Kernel**: Linux kernel 4.18+ with glibc 2.17+
- **Architecture**: x86_64 or ARM64

##### Windows Requirements
- **Version**: Windows 10 (64-bit) or Windows 11
- **Architecture**: x86_64 (Intel/AMD 64-bit processors)
- **WSL**: Windows Subsystem for Linux (optional but recommended for development)

#### Platform-Specific Installation

##### macOS Installation

**Option 1: Direct Download (Recommended)**
```bash
# Download the official installer
curl -L https://ollama.com/download/Ollama.dmg -o Ollama.dmg

# Open the DMG file and drag Ollama to Applications
open Ollama.dmg

# After installation, verify the installation
ollama --version
```

**Option 2: Homebrew**
```bash
# Install using Homebrew
brew install ollama

# Start the Ollama service
brew services start ollama

# Verify installation
ollama --version
```

##### Linux Installation

**Option 1: Universal Install Script (Recommended)**
```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Start Ollama service (if systemd is available)
sudo systemctl enable ollama
sudo systemctl start ollama
```

**Option 2: Manual Installation**
```bash
# Download the binary
sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama

# Make it executable
sudo chmod +x /usr/bin/ollama

# Create a system user for Ollama (optional but recommended)
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
```

**Option 3: Docker Installation**
```bash
# Pull the official Ollama Docker image
docker pull ollama/ollama

# Run Ollama in a container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# For GPU support (NVIDIA)
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

##### Windows Installation

**Option 1: Official Installer (Recommended)**
```powershell
# Download the installer
Invoke-WebRequest -Uri https://ollama.com/download/OllamaSetup.exe -OutFile OllamaSetup.exe

# Run the installer
.\OllamaSetup.exe

# Verify installation (after restart)
ollama --version
```

**Option 2: Package Managers**
```powershell
# Using winget
winget install Ollama.Ollama

# Using Chocolatey
choco install ollama

# Using Scoop
scoop install ollama
```

**Option 3: WSL2 (For Development)**
```bash
# Inside WSL2 Ubuntu/Debian environment
curl -fsSL https://ollama.com/install.sh | sh
```

#### Initial Configuration and Setup

##### 1. Start Ollama Service

**macOS/Linux:**
```bash
# Start Ollama server (runs in background)
ollama serve

# Or use system service (Linux with systemd)
sudo systemctl start ollama
sudo systemctl enable ollama  # Auto-start on boot
```

**Windows:**
```powershell
# Ollama starts automatically after installation
# Check if service is running
Get-Service -Name "Ollama*"

# Manually start if needed
Start-Service -Name "OllamaService"
```

##### 2. Configure Environment Variables (Optional)

```bash
# Set custom model storage location (optional)
export OLLAMA_MODELS="/path/to/your/models/directory"

# Set custom host/port (default: localhost:11434)
export OLLAMA_HOST="0.0.0.0:11434"

# Configure GPU settings (if applicable)
export CUDA_VISIBLE_DEVICES="0"  # Use first GPU only

# Add to your shell profile for persistence
echo 'export OLLAMA_MODELS="$HOME/.ollama/models"' >> ~/.bashrc
echo 'export OLLAMA_HOST="localhost:11434"' >> ~/.bashrc
```

#### Model Management

##### Downloading Models

C-Spirit works well with various model sizes depending on your hardware:

```bash
# Small models (good for 8GB RAM systems)
ollama pull llama3.2:3b          # 3 billion parameters, ~2GB
ollama pull gemma2:2b            # 2 billion parameters, ~1.4GB
ollama pull qwen2.5:3b           # 3 billion parameters, ~2GB

# Medium models (good for 16GB RAM systems)  
ollama pull llama3.2:7b          # 7 billion parameters, ~4.7GB
ollama pull gemma2:7b            # 7 billion parameters, ~4.7GB
ollama pull qwen2.5:7b           # 7 billion parameters, ~4.7GB

# Large models (requires 32GB+ RAM)
ollama pull llama3.1:13b         # 13 billion parameters, ~7.4GB
ollama pull llama3.1:70b         # 70 billion parameters, ~40GB

# Specialized models for code and reasoning
ollama pull deepseek-coder:6.7b  # Code generation and analysis
ollama pull codellama:7b         # Meta's code-focused model
```

##### Model Management Commands

```bash
# List installed models
ollama list

# Show model information
ollama show llama3.2:7b

# Remove a model to free space
ollama rm llama3.2:3b

# Update a model to latest version
ollama pull llama3.2:7b

# Copy a model with different name
ollama cp llama3.2:7b my-custom-model
```

#### Verification and Testing

##### 1. Basic Functionality Test

```bash
# Test basic model interaction
ollama run llama3.2:3b "Hello, can you help me with plant metabolomics research?"

# Test with a simple scientific query
ollama run llama3.2:3b "What are the main classes of plant secondary metabolites?"

# Exit the interactive session
# Type /bye or press Ctrl+D
```

##### 2. API Endpoint Test

```bash
# Test HTTP API (Ollama server must be running)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "List three common plant alkaloids:",
  "stream": false
}'

# Test model list endpoint
curl http://localhost:11434/api/tags
```

##### 3. Python Integration Test

Create a test script for C-Spirit integration:

```python
# save as test_ollama_integration.py
"""Test Ollama integration for C-Spirit project."""

import json
import requests
from typing import Optional

def test_ollama_connection() -> bool:
    """Test if Ollama server is accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def test_model_inference(model: str = "llama3.2:3b") -> Optional[str]:
    """Test model inference with a plant metabolomics query."""
    if not test_ollama_connection():
        print("❌ Ollama server not accessible")
        return None
    
    try:
        payload = {
            "model": model,
            "prompt": "What is the molecular formula of caffeine, a common plant alkaloid?",
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            print(f"✅ Model {model} responded successfully")
            print(f"Response: {answer[:100]}...")
            return answer
        else:
            print(f"❌ API request failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return None

def list_available_models() -> list:
    """List all available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

if __name__ == "__main__":
    print("Testing Ollama integration for C-Spirit...")
    
    # Test connection
    if test_ollama_connection():
        print("✅ Ollama server is running")
        
        # List models
        models = list_available_models()
        if models:
            print(f"✅ Available models: {', '.join(models)}")
            
            # Test inference with first available model
            test_model_inference(models[0])
        else:
            print("❌ No models found. Please pull a model first:")
            print("   ollama pull llama3.2:3b")
    else:
        print("❌ Ollama server not running. Please start it:")
        print("   ollama serve")
```

Run the test:
```bash
python test_ollama_integration.py
```

#### Usage Examples for C-Spirit

##### 1. Named Entity Recognition

```bash
# Extract metabolite names from text
ollama run llama3.2:7b "Extract all plant metabolite names from this text: 'The study found high levels of quercetin, kaempferol, and chlorogenic acid in the leaf extracts, along with trace amounts of caffeine and theobromine.'"
```

##### 2. Relationship Extraction

```bash
# Identify relationships between compounds and plants
ollama run llama3.2:7b "What is the relationship between salicin and willow bark? Explain the biosynthetic pathway."
```

##### 3. Text Classification

```bash
# Classify research abstracts
ollama run llama3.2:7b "Classify this abstract by research area: 'This study investigates the antimicrobial properties of essential oils extracted from Lavandula angustifolia against various bacterial strains.'"
```

#### Integration with C-Spirit

To use Ollama in your C-Spirit code:

```python
# Example integration in C-Spirit project
import requests
import json
from typing import Dict, Any, Optional

class OllamaLLMProvider:
    """Ollama LLM provider for C-Spirit."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:7b"):
        self.base_url = base_url
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using Ollama model."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get("timeout", 60)
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return None
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
    
    def extract_entities(self, text: str) -> list:
        """Extract named entities from text."""
        prompt = f"""Extract all plant metabolite names from the following text. Return only the names, one per line:

Text: {text}

Metabolite names:"""
        
        response = self.generate(prompt)
        if response:
            return [line.strip() for line in response.split('\n') if line.strip()]
        return []

# Usage example
if __name__ == "__main__":
    llm = OllamaLLMProvider(model="llama3.2:7b")
    
    text = "The plant contains quercetin, kaempferol, and chlorogenic acid."
    entities = llm.extract_entities(text)
    print(f"Extracted entities: {entities}")
```

#### Performance Optimization

##### 1. Model Selection Guidelines

For C-Spirit workloads, choose models based on your hardware:

| Hardware | Recommended Models | Use Cases |
|----------|-------------------|-----------|
| 8GB RAM | llama3.2:3b, gemma2:2b | Basic NER, simple classification |
| 16GB RAM | llama3.2:7b, qwen2.5:7b | Complex NER, relationship extraction |
| 32GB+ RAM | llama3.1:13b, deepseek-coder:6.7b | Advanced reasoning, code generation |

##### 2. GPU Acceleration

If you have a compatible GPU:

```bash
# Check GPU usage
nvidia-smi  # For NVIDIA GPUs

# Monitor GPU usage while running models
watch -n 1 nvidia-smi

# For Apple Silicon, GPU acceleration is automatic
```

##### 3. Memory Management

```bash
# Monitor system memory usage
htop  # or 'top' on macOS

# Clear model from memory when not in use
ollama stop llama3.2:7b

# Restart Ollama service to clear all models from memory
# macOS/Linux
pkill ollama && ollama serve

# Windows
Restart-Service -Name "OllamaService"
```

#### Troubleshooting

##### Common Issues and Solutions

**Problem**: `ollama: command not found`
```bash
# Solution 1: Verify installation path
which ollama

# Solution 2: Add to PATH (if installed manually)
export PATH="/usr/local/bin:$PATH"
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc

# Solution 3: Reinstall using package manager
# macOS
brew reinstall ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Problem**: `connection refused` or `server not running`
```bash
# Solution 1: Start Ollama server
ollama serve

# Solution 2: Check if port is in use
netstat -an | grep 11434
lsof -i :11434

# Solution 3: Use different port
export OLLAMA_HOST="localhost:11435"
ollama serve
```

**Problem**: `model not found` or download failures
```bash
# Solution 1: Verify model name
ollama list

# Solution 2: Re-download model
ollama rm llama3.2:7b
ollama pull llama3.2:7b

# Solution 3: Check disk space
df -h  # Linux/macOS
dir   # Windows

# Solution 4: Clear model cache
rm -rf ~/.ollama/models/*  # Use with caution
```

**Problem**: Out of memory errors
```bash
# Solution 1: Use smaller model
ollama pull llama3.2:3b  # Instead of 7b or 13b

# Solution 2: Stop other models
ollama stop llama3.2:7b

# Solution 3: Restart Ollama service
pkill ollama && ollama serve
```

**Problem**: Slow inference performance
```bash
# Solution 1: Check available RAM
free -h  # Linux
vm_stat  # macOS

# Solution 2: Use quantized models
ollama pull llama3.2:7b-q4_0  # 4-bit quantized version

# Solution 3: Enable GPU acceleration (if available)
# Ensure NVIDIA drivers and CUDA are properly installed
nvidia-smi
```

**Problem**: GPU not being utilized
```bash
# Solution 1: Verify GPU drivers (NVIDIA)
nvidia-smi

# Solution 2: Check CUDA installation
nvcc --version

# Solution 3: Restart Ollama after driver installation
pkill ollama && ollama serve

# Solution 4: For Apple Silicon, check Activity Monitor
# GPU acceleration is automatic but can be monitored
```

**Problem**: Permission denied errors (Linux)
```bash
# Solution 1: Fix ownership
sudo chown -R $USER:$USER ~/.ollama

# Solution 2: Run with proper permissions
sudo ollama serve

# Solution 3: Add user to ollama group (if created)
sudo usermod -a -G ollama $USER
```

##### Advanced Troubleshooting

**Enable Debug Logging:**
```bash
# Set debug environment variable
export OLLAMA_DEBUG=1
ollama serve

# Check logs location
# Linux: /var/log/ollama.log or journalctl -u ollama
# macOS: ~/Library/Logs/Ollama/
# Windows: Check Event Viewer
```

**Check System Resources:**
```bash
# Monitor resource usage during model operation
htop  # or top

# Check disk I/O
iotop  # Linux only

# Monitor network usage (for model downloads)
nethogs  # Linux
nettop   # macOS
```

**Reset Ollama Configuration:**
```bash
# Remove all models and configuration (nuclear option)
rm -rf ~/.ollama

# Reinstall and reconfigure
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

#### Security Considerations

##### 1. Network Security

```bash
# By default, Ollama only listens on localhost
# To allow external connections (use with caution):
export OLLAMA_HOST="0.0.0.0:11434"

# For production, use reverse proxy with authentication
# nginx, Apache, or similar
```

##### 2. Model Verification

```bash
# Verify model checksums when possible
ollama show llama3.2:7b --verbose

# Only download models from trusted sources
# Official Ollama model library: https://ollama.com/library
```

##### 3. Data Privacy

- All processing happens locally - no data sent to external servers
- Model files are stored locally in `~/.ollama/models/`
- Consider encrypting the models directory for sensitive deployments

#### Model Recommendations for C-Spirit

Based on C-Spirit's use cases in plant metabolomics research:

##### For Named Entity Recognition:
- **llama3.2:7b** - Best balance of accuracy and speed
- **qwen2.5:7b** - Excellent for scientific text
- **gemma2:7b** - Good general-purpose model

##### For Relationship Extraction:
- **llama3.1:13b** - Superior reasoning capabilities (if you have enough RAM)
- **deepseek-coder:6.7b** - Good for structured data extraction

##### For Text Classification:
- **llama3.2:3b** - Fast and sufficient for most classification tasks
- **gemma2:2b** - Minimal resource usage

##### Development and Testing:
- **llama3.2:3b** - Quick iteration and testing
- **qwen2.5:3b** - Good for prototyping scientific applications

#### Integration with Development Workflow

```bash
# Add to your development environment
# Add these to your .env file or shell profile:

# Ollama configuration
export OLLAMA_HOST="localhost:11434"
export OLLAMA_MODELS="$HOME/.ollama/models"

# C-Spirit LLM configuration  
export C_SPIRIT_LLM_PROVIDER="ollama"
export C_SPIRIT_LLM_MODEL="llama3.2:7b"
export C_SPIRIT_LLM_ENDPOINT="http://localhost:11434"

# Performance tuning
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
```

### 2. Additional NLP Tools
```bash
# For advanced text processing (optional)
pip install spacy nltk

# Download spaCy models if needed
python -m spacy download en_core_web_sm
```

### 3. Graph Visualization (Optional)
```bash
# For ontology visualization
pip install networkx matplotlib graphviz

# System graphviz (required for Python graphviz)
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt install graphviz
```

## Additional Non-Python Dependencies

While the core C-Spirit functionality relies primarily on Python libraries, several non-Python dependencies may be required for advanced features, performance optimization, or specific use cases. This section outlines potential additional dependencies that might arise during development or deployment.

### 1. Graph Visualization and Analysis

#### Graphviz (For Ontology and Network Visualization)
**Purpose**: Rendering complex ontology structures, metabolic networks, and relationship graphs
**When needed**: When generating publication-quality diagrams, debugging ontology structures, or creating visual reports

##### Installation:

**macOS:**
```bash
# Using Homebrew (recommended)
brew install graphviz

# Using MacPorts
sudo port install graphviz
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install graphviz graphviz-dev

# Additional fonts for better diagram rendering
sudo apt install fonts-liberation fonts-dejavu
```

**RHEL/CentOS/Fedora:**
```bash
# RHEL/CentOS
sudo yum install graphviz graphviz-devel

# Fedora
sudo dnf install graphviz graphviz-devel
```

**Windows:**
```powershell
# Using Chocolatey
choco install graphviz

# Using Scoop
scoop install graphviz

# Manual installation: Download from https://graphviz.org/download/
```

##### Verification:
```bash
# Test Graphviz installation
dot -V
neato -V
fdp -V

# Test with simple graph
echo 'digraph G { A -> B -> C }' | dot -Tpng -o test.png
```

#### Neo4j (Optional - For Large-Scale Graph Databases)
**Purpose**: Storing and querying large metabolic networks and complex ontology relationships
**When needed**: For projects with >100k entities or complex graph traversals

```bash
# Installation varies by system - see Neo4j documentation
# Typically requires Java 11+ (already covered in Java section)

# Community Edition (free)
# Download from: https://neo4j.com/download-center/
```

### 2. Document Processing and OCR

#### Tesseract OCR (For Scanned Document Processing)
**Purpose**: Extracting text from scanned PDFs and images in scientific literature
**When needed**: Processing older papers or image-based chemical structure diagrams

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # For additional languages
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
# Additional language packs if needed
sudo apt install tesseract-ocr-deu tesseract-ocr-fra
```

**Windows:**
```powershell
choco install tesseract
# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### Poppler Utils (For Advanced PDF Processing)
**Purpose**: Enhanced PDF text extraction and manipulation
**When needed**: Processing complex PDFs with embedded fonts or unusual layouts

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt install poppler-utils
```

### 3. Database Systems

#### PostgreSQL (Optional - For Large-Scale Data Storage)
**Purpose**: Storing extracted entities, relationships, and intermediate processing results
**When needed**: Large-scale deployments or when SQLite performance becomes insufficient

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### Redis (Optional - For Caching and Job Queues)
**Purpose**: Caching LLM responses, intermediate results, and managing background tasks
**When needed**: Multi-user deployments or batch processing workflows

**macOS:**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 4. Scientific Computing Libraries

#### BLAST+ (For Sequence Analysis)
**Purpose**: Protein and nucleotide sequence similarity searches
**When needed**: If extending C-Spirit to include genomic or proteomic data

**macOS:**
```bash
brew install blast
```

**Ubuntu/Debian:**
```bash
sudo apt install ncbi-blast+
```

#### RDKit Dependencies (For Chemical Structure Analysis)
**Purpose**: Chemical structure manipulation, similarity calculations, and molecular property prediction
**When needed**: Advanced chemical structure analysis beyond basic Python libraries

**System Libraries:**
```bash
# macOS
brew install boost cmake eigen

# Ubuntu/Debian
sudo apt install libboost-all-dev cmake libeigen3-dev

# Note: RDKit itself is available as a Python package,
# but system libraries may improve performance
```

### 5. Machine Learning and AI Infrastructure

#### CUDA Toolkit (For GPU-Accelerated Computing)
**Purpose**: Accelerating large language model inference and machine learning computations
**When needed**: Using local GPU resources for LLM processing or large-scale entity extraction

**Requirements**: NVIDIA GPU with compute capability 3.5 or higher

**Installation**: 
- Download from NVIDIA Developer website
- Follow platform-specific installation guides
- Verify with `nvidia-smi` and `nvcc --version`

#### Docker (For Containerized Deployments)
**Purpose**: Consistent deployment environments and service orchestration
**When needed**: Production deployments or complex multi-service architectures

**macOS:**
```bash
brew install --cask docker
# Or download Docker Desktop from docker.com
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER  # Add current user to docker group
```

### 6. Version Control and Large File Storage

#### Git LFS (Large File Storage)
**Purpose**: Managing large ontology files, model weights, and dataset files in version control
**When needed**: Storing files >100MB in Git repositories

**Installation:**
```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows
choco install git-lfs

# Initialize in repository
git lfs install
git lfs track "*.owl"
git lfs track "*.rdf"
git lfs track "*.model"
```

### 7. Web Services and APIs

#### Apache HTTP Server or Nginx (Optional)
**Purpose**: Serving API endpoints or web interfaces for ontology browsing
**When needed**: Deploying C-Spirit as a web service or creating public APIs

**macOS:**
```bash
brew install nginx
brew services start nginx
```

**Ubuntu/Debian:**
```bash
sudo apt install nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

### 8. System Monitoring and Debugging

#### htop/btop (Enhanced System Monitoring)
**Purpose**: Monitoring resource usage during intensive processing tasks
**When needed**: Performance optimization and debugging

**macOS:**
```bash
brew install htop btop
```

**Ubuntu/Debian:**
```bash
sudo apt install htop
# btop installation varies - check GitHub releases
```

#### valgrind (Memory Debugging - Linux only)
**Purpose**: Debugging memory issues in compiled extensions
**When needed**: Troubleshooting memory leaks in native Python extensions

```bash
# Ubuntu/Debian
sudo apt install valgrind
```

### 9. Internationalization and Text Processing

#### ICU (International Components for Unicode)
**Purpose**: Advanced text processing, normalization, and internationalization
**When needed**: Processing non-English scientific literature or special characters

**macOS:**
```bash
brew install icu4c
```

**Ubuntu/Debian:**
```bash
sudo apt install libicu-dev
```

### 10. Verification and Testing

To verify optional dependencies, use these test commands:

```bash
# Graph visualization
dot -V && echo "✅ Graphviz available" || echo "❌ Graphviz not installed"

# OCR capabilities
tesseract --version && echo "✅ Tesseract available" || echo "❌ Tesseract not installed"

# PDF processing
pdfinfo -v && echo "✅ Poppler available" || echo "❌ Poppler not installed"

# Database systems
psql --version && echo "✅ PostgreSQL available" || echo "❌ PostgreSQL not installed"
redis-cli --version && echo "✅ Redis available" || echo "❌ Redis not installed"

# BLAST tools
blastn -version && echo "✅ BLAST+ available" || echo "❌ BLAST+ not installed"

# Docker
docker --version && echo "✅ Docker available" || echo "❌ Docker not installed"

# Git LFS
git lfs version && echo "✅ Git LFS available" || echo "❌ Git LFS not installed"
```

### Installation Priority

Dependencies are categorized by priority:

1. **High Priority** (likely needed for most deployments):
   - Graphviz (for visualization)
   - Git LFS (for large file management)
   - Enhanced monitoring tools (htop/btop)

2. **Medium Priority** (needed for specific features):
   - Tesseract OCR (for scanned documents)
   - PostgreSQL (for large-scale storage)
   - Docker (for containerized deployment)

3. **Low Priority** (specialized use cases):
   - CUDA Toolkit (GPU acceleration)
   - Neo4j (complex graph queries)
   - BLAST+ (sequence analysis)
   - Redis (caching and queues)

### Performance Considerations

- **Graphviz**: Large ontologies may require increased memory allocation
- **Database Systems**: Configure appropriate connection pools and memory settings
- **CUDA**: Ensure sufficient GPU memory for chosen LLM models
- **Docker**: Monitor container resource usage in production environments

## Environment Configuration

### 1. Environment Variables

Create a `.env` file in the project root (optional):

```bash
# API Keys (if using external services)
NCBI_API_KEY=your_ncbi_api_key_here
PUBMED_EMAIL=your_email@domain.com

# LLM Configuration
C_SPIRIT_LLM_PROVIDER=openai
C_SPIRIT_LLM_MODEL=gpt-4

# Data directories
C_SPIRIT_DATA_DIR=./data
C_SPIRIT_ONTOLOGY_DIR=./data/ontologies
C_SPIRIT_OUTPUT_DIR=./data/outputs

# Logging
C_SPIRIT_LOG_LEVEL=INFO
```

### 2. Configure Git (Recommended)

```bash
# Set up git configuration for development
git config --local user.name "Your Name"
git config --local user.email "your.email@domain.com"

# Set up git hooks (if using pre-commit)
pre-commit install
```

## Next Steps

After successful installation:

1. **Read the Documentation**: Check `docs/plan.md` and `docs/tickets.md` for project details
2. **Explore Examples**: Review the usage examples in `README.md`
3. **Run Tests**: Execute the full test suite to ensure everything works
4. **Configure Environment**: Set up any required API keys and configuration
5. **Start Development**: Begin with the basic ontology development examples

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the Issues**: [GitHub Issues](https://github.com/your-org/c-spirit/issues)
2. **Review Documentation**: See the `docs/` directory for additional information
3. **Run Diagnostics**: Use the verification scripts above to identify specific problems
4. **Community Support**: Join discussions at [GitHub Discussions](https://github.com/your-org/c-spirit/discussions)

---

**Last Updated**: August 2025  
**Compatible with**: C-Spirit v0.1.0+  
**Python Version**: 3.9.x required