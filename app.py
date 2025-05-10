import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import random

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logo.jpeg", width=100)  # Adjust size as needed
with col_title:
    st.markdown("<h1 style='margin-bottom: 0; margin-left:0px'>ESRGAN-CBAM Super Resolution</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.saved_model.load("esrgan_model")
    return model

model = load_model()

def pil_image(obj) :
    obj= tf.cast(obj, tf.uint8).numpy()
    return Image.fromarray(obj)

def load_image(uploaded_file):
    image_data = uploaded_file.getvalue()
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    return image

def logs(msg) :
    print("#####################################")
    print(msg)
    print("#####################################")

def preprocess_image(image):
    hr_size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    image = tf.image.crop_to_bounding_box(image, 0, 0, hr_size[0], hr_size[1])
    image = tf.cast(image, tf.float32)
    batch_image = tf.expand_dims(image, 0)
    return batch_image

def get_random_crop(image, crop_size=32):
    h, w, _ = image.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    crop = tf.image.crop_to_bounding_box(image, top, left, crop_size, crop_size)
    return tf.expand_dims(crop, 0), crop.numpy().astype(np.uint8)

def super_resolve(image):
    sr = model(image)
    sr = sr.numpy().squeeze() 
    sr_image = tf.cast(sr, tf.uint8).numpy()
    return Image.fromarray(sr_image)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)

    if st.button("Generate SR"):
        with st.spinner("Processing..."):
            try:
                preprocessed = preprocess_image(image)
                sr_image = super_resolve(preprocessed)
                org_image = pil_image(image)

                # Display Side-by-Side
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(org_image, use_container_width=True)

                with col2:
                    st.subheader("Super-Resolved Image")
                    st.image(sr_image, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                
    if st.button("Zoom"):
        with st.spinner("Processing..."):
            try:
                crop_tensor, crop_np = get_random_crop(image)
                sr_np = super_resolve(crop_tensor)

                # Display Side-by-Side
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Cropped Part (32x32)")
                    st.image(pil_image(crop_np), use_container_width=True)

                with col2:
                    st.subheader("Super-Resolved Zoom")
                    st.image(sr_np, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")