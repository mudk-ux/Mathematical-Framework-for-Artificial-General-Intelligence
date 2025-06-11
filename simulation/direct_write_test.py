#!/usr/bin/env python3
"""
Test script to verify file writing in the optimization experiment
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def run_test():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/direct_write_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "test.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting direct write test")
    
    # Create some test data
    test_data = {
        "test": "data",
        "numbers": [1, 2, 3, 4, 5],
        "array": np.array([1, 2, 3, 4, 5]).tolist()
    }
    
    # Try to save the data
    try:
        # Create a simple test file first
        test_file_path = os.path.join(output_dir, "test_write.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test write permissions")
        print(f"Successfully created test file at {test_file_path}")
        
        # Now try to save the raw data
        raw_data_path = os.path.join(output_dir, "raw_data.json")
        print(f"Attempting to save raw data to {raw_data_path}")
        
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
    
    logger.info(f"Test completed. Results saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    output_dir = run_test()
    print(f"Test output directory: {output_dir}")
    print(f"Files in output directory:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
