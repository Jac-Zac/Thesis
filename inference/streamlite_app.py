#!/usr/bin/env python3
import streamlit as st
import torch
from PIL import ExifTags
from PIL import Image
from transformers import DonutProcessor
from transformers import VisionEncoderDecoderConfig
from transformers import VisionEncoderDecoderModel


def run_prediction(sample):
    global pretrained_model, processor, task_prompt
    if isinstance(sample, dict):
        # prepare inputs
        pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0)
    else:  # sample is an image
        # prepare encoder inputs
        pixel_values = processor(image, return_tensors="pt").pixel_values

    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    # run inference
    outputs = pretrained_model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=pretrained_model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]

    # post-processing
    if "cord" in task_prompt:
        prediction = prediction.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        # prediction = re.sub(r"<.*?>", "", prediction, count=1).strip()  # remove first task start token
    prediction = processor.token2json(prediction)

    # load reference target
    if isinstance(sample, dict):
        target = processor.token2json(sample["target_sequence"])
    else:
        target = "<not_provided>"

    return prediction, target


# Image processing change the orientation if needed and the size accordingly to the model we use
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


# What does this model do
task_prompt = "<s_herbarium>>"
st.markdown(
    """
### Donut Herbarium Testing
Experimental OCR-free Document Understanding Vision Transformer, fine-tuned with an herbarium dataset of around 1400 images.
"""
)

with st.sidebar:
    information = st.radio(
        "Choose one predictor:",
        ("Low Res (1200 * 900) 5 epochs", "Mid res (1600 ^ 1200) 10 epochs"),
    )
    image_choice = st.selectbox("Pick one 📑", ["1", "2", "3"], index=1)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

st.text(
    f"{information} mode is ON!\nTarget 📑: {image_choice}"
)  # \n(opening image @:./img/receipt-{receipt}.png)')

col1, col2 = st.columns(2)

# Chose image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    pass
    # image_choice_map = {
    #    '1': '../donut_example/copy/img_resized/test/00021.jpg',
    #    '2': '../donut_example/copy/img_resized/test/00031.jpg',
    #    '3': '../donut_example/copy/img_resized/test/00050.jpg',
    # }
    # image = Image.open(image_choice_map[image_choice])


if information == "Low Res (1200 * 900) 5 epochs":
    image = preprocess_image(image, (1200, 900))
else:
    image = preprocess_image(image, (1600, 1200))

with col1:
    st.image(image, caption="Your target sample")

# Run the model
if st.button("Parse sample! 🐍"):
    image = image.convert("RGB")

    # Choose which version to run base on the selected box
    with st.spinner(f"Running the model on the target..."):
        if information == "Low Res (1200 * 900) 5 epochs":
            processor = DonutProcessor.from_pretrained(
                "Jac-Zac/thesis_test_donut",
                revision="12900abc6fb551a0ea339950462a6a0462820b75",
            )
            pretrained_model = VisionEncoderDecoderModel.from_pretrained(
                "Jac-Zac/thesis_test_donut",
                revision="12900abc6fb551a0ea339950462a6a0462820b75",
            )

        elif information == "Mid res (1600 ^ 1200) 10 epochs":
            processor = DonutProcessor.from_pretrained(
                "Jac-Zac/thesis_test_donut",
                revision="8c5467cb66685e801ec6ff8de7e7fdd247274ed0",
            )
            pretrained_model = VisionEncoderDecoderModel.from_pretrained(
                "Jac-Zac/thesis_test_donut",
                revision="8c5467cb66685e801ec6ff8de7e7fdd247274ed0",
            )

        # this is the same for both models
        task_prompt = f"<s_herbarium>"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_model.to(device)

    with col2:
        st.info(f"Parsing 📑...")
        parsed_info, _ = run_prediction(image)
        st.text(f"\n{information}")
        st.json(parsed_info)
