#!/usr/bin/env python3
"""
Test script to verify file saving functionality
"""

import os
import json
import sys

def main():
    # Create output directory
    output_dir = "./results/save_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple data structure for testing
    test_data = {
        "test": "data",
        "numbers": [1, 2, 3, 4, 5]
    }
    
    # Save raw data
    raw_data_path = os.path.join(output_dir, "raw_data.json")
    print(f"Attempting to save raw data to {raw_data_path}")
    
    try:
        with open(raw_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Verify the file was created
        if os.path.exists(raw_data_path):
            print(f"Raw data successfully exported to {raw_data_path}")
            print(f"File size: {os.path.getsize(raw_data_path)} bytes")
        else:
            print(f"Failed to create raw data file at {raw_data_path}")
    except Exception as e:
        print(f"Error exporting raw data: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
