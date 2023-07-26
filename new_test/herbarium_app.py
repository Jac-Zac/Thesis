#!/opt/homebrew/bin/streamlit run
import base64
import json
from io import BytesIO

import cv2
import numpy as np
import requests
import streamlit as st
from langchain import HuggingFaceHub
from langchain import LLMChain
from langchain import PromptTemplate
from PIL import Image
from PIL import ImageOps
from pytesseract import image_to_string
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline


# Define a function to pre process the image
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

def extract_information(text):
    model = HuggingFaceHub(repo_id='gpt2', huggingfacehub_api_token='your_token')

    # Create a prompt with the template
    template = '''
    You have to give me only what I want so that I can save it as a json.
    Based on the text I give you return:

    - Name of the species:

    - Date when it was found:

    - Location where it was found:

    IMAGE TEXT: {text}
    '''

    prompt = PromptTemplate(
            input_variables=["text"],
            template=template
            )


    llm_chain = LLMChain(prompt=prompt, llm=model, verbose=True)

    informations = llm_chain.predict(text=text)

    # Parse the response
    response_json = json.loads(informations)

    st.write(response_json)

    return response_json

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
