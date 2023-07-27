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
from langchain import OpenAI
from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from PIL import Image
from PIL import ImageOps
from pytesseract import image_to_string

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
    ocr_text = image_to_string(image)
    return ocr_text


# https://www.youtube.com/watch?v=2xxziIWmaSA minute 19 help formatting

# Retrive and vector Stores
def extract_information(ocr_text):

    repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    # repo_id = "databricks/dolly-v2-3b"

    model = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64},
    )

    template = """
    OCR TEXT: {ocr_text}

    Please extract the following information from the OCR text:
    1. Species Name (put the name of the spieces in the correct standard):
    2. Author:
    3. Date (format the date in a standard way):
    4. Location:
    5. Altitude:

    Respond only with the json output.
    JSON OUTPUT:
    """

    # You have to give me only what I want so that I can save it as a json.

    prompt = PromptTemplate(input_variables=["ocr_text"], template=template)

    llm_chain = LLMChain(prompt=prompt, llm=model, verbose=True)

    informations = llm_chain.predict(ocr_text=ocr_text)

    return informations


# Set up the Streamlit app
st.title("Herbarium Image Text Extractor")

# Upload the image file
uploaded_file = st.file_uploader(
    "Choose an herbarium image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Convert the uploaded file to a PIL Image object
    image = Image.open(uploaded_file)

    # Preprocess the image
    image = preprocess_image(image, (1600, 1200))

    # Extract text from the image using OCR
    ocr_text = extract_text(image)

    # Extract information from the text using the language model
    information = extract_information(ocr_text)

    # Display the extracted information
    st.write("Model Response:")
    st.write(information)
    print(information)

else:
    st.write("Please upload an image.")
