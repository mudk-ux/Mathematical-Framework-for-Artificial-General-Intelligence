import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.abspath('.'))

# Try to import the extensions
try:
    from extensions.infinite_population import MeanFieldApproximation
    print("Successfully imported MeanFieldApproximation")
except ImportError as e:
    print(f"Failed to import MeanFieldApproximation: {e}")

# Print directory structure
print("\nDirectory structure:")
for root, dirs, files in os.walk("extensions"):
    print(f"{root}/")
    for file in files:
        if file.endswith(".py"):
            print(f"  - {file}")
