# test_setup.py - Verify all libraries are installed correctly

print("Testing library imports...")

# Track import success
imports_successful = []

try:
    import transformers
    print("✓ transformers installed")
    imports_successful.append(True)
except ImportError:
    print("✗ transformers NOT installed")
    imports_successful.append(False)

try:
    import torch
    print("✓ torch installed")
    imports_successful.append(True)
except ImportError:
    print("✗ torch NOT installed")
    imports_successful.append(False)

try:
    import pandas
    print("✓ pandas installed")
    imports_successful.append(True)
except ImportError:
    print("✗ pandas NOT installed")
    imports_successful.append(False)

try:
    import matplotlib
    print("✓ matplotlib installed")
    imports_successful.append(True)
except ImportError:
    print("✗ matplotlib NOT installed")
    imports_successful.append(False)

try:
    import seaborn
    print("✓ seaborn installed")
    imports_successful.append(True)
except ImportError:
    print("✗ seaborn NOT installed")
    imports_successful.append(False)

try:
    import numpy
    print("✓ numpy installed")
    imports_successful.append(True)
except ImportError:
    print("✗ numpy NOT installed")
    imports_successful.append(False)

try:
    import sklearn
    print("✓ scikit-learn installed")
    imports_successful.append(True)
except ImportError:
    print("✗ scikit-learn NOT installed")
    imports_successful.append(False)

# Check if all imports were successful
if all(imports_successful):
    print("\n✓ All required libraries are ready!")
else:
    print("\n✗ Some libraries are missing. Please install them using:")
    print("  pip install -r requirements.txt")
