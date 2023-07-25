#!/usr/bin/env python3
import json
from io import BytesIO

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import ExifTags
from PIL import Image

def preprocess_image(image, size):
    # Resize the image to a specific size
    image = image.resize(size)

    # Check if the image has orientation metadata and rotate it if necessary
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            if hasattr(image, "_getexif"):
                exif = dict(image._getexif().items())
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            break
    return image


st.title("Herbarium Image Text Extractor")

uploaded_file = st.file_uploader("Choose an herbarium image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = preprocess_image(image, (1600, 1200))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Extract Information"):
        # Call the LangChain agent to extract information
        pass

def extract_information(image):
    url = "https://api.langchain.com/v1/agent"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_LANGCHAIN_API_KEY"
    }
    # Convert the image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    data = {
        "image": img_base64,
        "task": "extract_herbarium_information"
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


if st.button("Extract Information"):
    extracted_info = extract_information(image)
    st.write(extracted_info)
