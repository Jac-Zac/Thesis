#!/usr/bin/env python3
import json
from collections import OrderedDict

from pyexcel_ods3 import save_data

# Load your JSON data
with open("../output.json") as f:
    data = json.load(f)

# Prepare data for ODS
ods_data = OrderedDict()
sheet_data = [["filename"] + list(data[0]["prediction"].keys())]

for item in data:
    filename = item["filename"]
    prediction = item["prediction"]
    row_data = [filename] + list(prediction.values())
    sheet_data.append(row_data)

ods_data["Sheet1"] = sheet_data

# Save data to an ODS file
save_data("../output.ods", ods_data)
