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
```bash
# Install Ollama for local LLM inference
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (optional)
ollama pull llama2
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