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

# How to use PALM: https://www.youtube.com/watch?v=orPwLibLqm4

def preprocess_image(image, size):
    image = image.resize(size)
    image = ImageOps.exif_transpose(image)
    return image

def extract_text(image):
    ocr_text = image_to_string(image)
    return ocr_text

def extract_information(ocr_text):
    repo_id = "google/flan-t5-xxl"
    model = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64},
        huggingfacehub_api_token="your-token"
    )

    # Also search for response schema and structured output parser... I WANT JSON
    # better prompt https://www.youtube.com/watch?v=jhI1z38Kj4I
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

    prompt = PromptTemplate(input_variables=["ocr_text"], template=template)
    llm_chain = LLMChain(prompt=prompt, llm=model, verbose=True)
    informations = llm_chain.predict(ocr_text=ocr_text)

    return informations

st.set_page_config(
    page_title="Herbarium Image Text Extractor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Herbarium Image Text Extractor Herbarium 🌿")
st.sidebar.title("TrOCR Image Text Extractor")
st.sidebar.write("Created by Jacopo Zacchigna")
st.sidebar.markdown("[GitHub](https://github.com/Jac-Zac)")

uploaded_file = st.sidebar.file_uploader(
    "Choose an herbarium image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = preprocess_image(image, (1600, 1200))
    ocr_text = extract_text(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Extracted Information")
        if st.button("Run Extraction"):
            information = extract_information(ocr_text)
            st.write(information)
else:
    st.sidebar.write("Please upload an image.")
