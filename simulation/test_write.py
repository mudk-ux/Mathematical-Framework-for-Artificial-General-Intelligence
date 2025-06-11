#!/usr/bin/env python3
import os
import json

output_dir = "./results/test_output"
os.makedirs(output_dir, exist_ok=True)

data = {"test": "data"}
file_path = os.path.join(output_dir, "test_data.json")

with open(file_path, "w") as f:
    json.dump(data, f)

print(f"File created at {file_path}")
print(f"File exists: {os.path.exists(file_path)}")
print(f"File size: {os.path.getsize(file_path)} bytes")
