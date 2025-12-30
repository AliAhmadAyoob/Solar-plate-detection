import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO('helmet-model.pt')
model = load_model()

st.set_page_config(page_title="Helmet Detection Model", page_icon="🪖")

st.title("🪖 License Plate Recognition")
st.write("Upload an image of helmet to recognize detect helmet.")
uploaded = st.file_uploader('Choose an image...',type=['jpg','jpeg','png'])

if uploaded is not None:
    if uploaded.type.startswith('image'):
        img = Image.open(uploaded).convet("RGB")
        st.image(img,caption='uploaded image',use_column_width=True)

        with spinner('Detecting....'):
            pred = model.predict(img)
            result_img = pred[0].plot()[ :,:,::-1]
            st.image(result_img,caption='Detected image',use_container_width=True)
        