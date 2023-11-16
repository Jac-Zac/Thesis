#!/usr/bin/env python3
import json

# Load your JSON data
with open("../output.json") as f:
    data = json.load(f)

# List of required keys
required_keys = ["Nome_verbatim", "Locality", "Elevation", "Day", "Month", "Year"]

# Iterate over the data
for item in data:
    # Check if all keys are present in the 'prediction' dictionary
    for key in required_keys:
        value = item["prediction"].get(key)
        if value is None:
            print(
                f"Key '{key}' is missing in the prediction for filename '{item['filename']}'"
            )
            # Add the missing key with a default value
            item["prediction"].setdefault(key, "")
        elif isinstance(value, dict):
            print(
                f"Encountered a dictionary for key '{key}' in the prediction for filename '{item['filename']}'"
            )
            item["prediction"][key] = ""
            # Handle the dictionary value as needed

# Save the updated data to a new JSON file
with open("../output_clean.json", "w") as f:
    json.dump(data, f)

# Command to print the missing keys
# cat output.json | jq | grep -B 10 -A 10 "zbf_0012.jpg"
