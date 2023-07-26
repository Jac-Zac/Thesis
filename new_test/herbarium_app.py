#!/opt/homebrew/anaconda3/bin/streamlit run
import base64
import json
from io import BytesIO

import cv2
import numpy as np
import openai
import requests
import streamlit as st
from PIL import Image
from PIL import ImageOps
from pytesseract import image_to_string
from transformers import pipeline

# Set up the OpenAI API key and model
model = "gpt-3.5-trivial- examples"

# Define a function to preprocess the image
def preprocess_image(image, size):
    # Resize the image to a specific size
    image = image.resize(size)

    # Automatically rotate the image based on its EXIF orientation metadata
    image = ImageOps.exif_transpose(image)

    return image

# Define a function to extract text from the image using OCR
def extract_text(image):
    # Perform OCR on the image using PyTesseract
    text = image_to_string(image)
    return text

# Define a function to extract information from the text using a language model
def extract_information(text):
    # Create a request to the OpenAI completion endpoint
    headers = {"Content-Type": "application/json"}
    payload = {
        "engine": "text-davinci-002",
        "prompt": f"Name of the species: \\nDate when it was found: \\nLocation where it was found: \\n\\nImage text: {text}",
        "max_tokens": 1024,
        "n": 1,
        "stop": None,
        "temperature": 0.7,
    }
    response = requests.post("https://api.openai.com/v1/engines/text-davinci-002/completions", headers=headers, data=json.dumps(payload))

    # Parse the response JSON
    response_json = json.loads(response.content)

    st.write(response_json)
    # Extract the first completion from the response
    #completion = response_json["choices"][0]

    # Return the extracted information in JSON
    # return json.loads(completion["text"])
    return json.loads(response_json)

# Set up the Streamlit app
st.title("Herbarium Image Text Extractor")

# Upload the image file
uploaded_file = st.file_uploader("Choose an herbarium image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a PIL Image object
    image = Image.open(uploaded_file)

    # Preprocess the image
    image = preprocess_image(image, (1600, 1200))

    # Extract text from the image using OCR
    text = extract_text(image)

    # Extract information from the text using the language model
    information = extract_information(text)

    # Display the extracted information
    st.write(information)
else:
    st.write("Please upload an image.")
