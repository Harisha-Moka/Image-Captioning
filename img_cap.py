import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
import requests

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_input):
    # Check if the input is a URL (string)
    if isinstance(image_input, str) and image_input.startswith("http"):
        # Load image from URL
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif hasattr(image_input, 'read'):  # Check if the input is an UploadedFile object
        # Load image from UploadedFile object (file-like object)
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError("Invalid image input")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate captions
    outputs = model.generate(**inputs)

    # Decode the generated captions
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption, image

def get_image_download_link(image, filename="image.png"):
    """Generate a link to download the image."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return st.download_button(label="Download Image", data=buffered, file_name=filename, mime="image/png")

# Streamlit UI
st.title("Image Captioning with BLIP")
st.write("Provide an image URL or upload an image to generate a caption.")

# Option to provide a URL or upload an image
image_source = st.radio("Select Image Source", ("URL", "Upload"))

if image_source == "URL":
    image_url = st.text_input("Enter Image URL")
    if image_url:
        try:
            with st.spinner("Generating caption..."):
                caption, image = generate_caption(image_url)
            st.image(image, caption="Input Image", use_column_width=True)
            st.success(f"Generated Caption: {caption}")
            get_image_download_link(image)  # Add download button for the image
        except Exception as e:
            st.error(f"Error: {e}")

elif image_source == "Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            with st.spinner("Generating caption..."):
                caption, image = generate_caption(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success(f"Generated Caption: {caption}")
            get_image_download_link(image)  # Add download button for the image
        except Exception as e:
            st.error(f"Error: {e}")