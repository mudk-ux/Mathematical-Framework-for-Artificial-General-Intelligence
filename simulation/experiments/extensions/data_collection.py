#!/usr/bin/env python3
"""
Data Collection Module for MMAI System Extensions

This module provides functions for collecting and saving experimental data.
"""

import os
import json
import numpy as np
import logging

def save_raw_data(data, output_dir, filename="raw_data.json"):
    """
    Save raw data to a JSON file
    
    Parameters:
    - data: Data to save
    - output_dir: Directory to save the data
    - filename: Name of the file to save the data
    
    Returns:
    - success: True if the data was saved successfully, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full path
        raw_data_path = os.path.join(output_dir, filename)
        logger.info(f"Attempting to save raw data to {raw_data_path}")
        
        # Convert data to JSON-serializable format
        serializable_data = convert_to_serializable(data)
        
        # Save data
        with open(raw_data_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Verify the file was created
        if os.path.exists(raw_data_path):
            logger.info(f"Raw data successfully exported to {raw_data_path}")
            return True
        else:
            logger.error(f"Failed to create raw data file at {raw_data_path}")
            return False
    except Exception as e:
        logger.error(f"Error exporting raw data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def convert_to_serializable(data):
    """
    Convert data to JSON-serializable format
    
    Parameters:
    - data: Data to convert
    
    Returns:
    - serializable_data: JSON-serializable data
    """
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data
