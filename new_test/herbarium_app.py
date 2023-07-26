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
def extract_information(ocr_text):

    model = OpenAI(model_name = "text-davinci-003", openai_api_key=openai_api_key)

    template = """
    OCR TEXT: {ocr_text}

    Please extract the following information from the OCR text:

    1. Species Name: `species_name`
    2. Author: `author`
    3. Date: `date`
    4. Locatio: `location`

    Model Output§:
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
