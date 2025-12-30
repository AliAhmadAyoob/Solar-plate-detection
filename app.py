import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    # return YOLO('helmet-model.pt')
    pass
model = load_model()

st.set_page_config(page_title="Solar Plate Detection Model", page_icon="🛰️")

st.title("🛰️ Solar Plate Detection")
st.write("Upload image of solar panel to detect whether it is **clear/not clear**.")
uploaded = st.file_uploader('Choose an image...',type=['jpg','jpeg','png'])

if uploaded is not None:
    if uploaded.type.startswith('image'):
        img = Image.open(uploaded).convert("RGB")
        # st.image(img,caption='uploaded image',use_column_width=True)

        with st.spinner('Detecting....'):
            pred = model.predict(img)
            result_img = pred[0].plot()[ :,:,::-1]
            st.image(result_img,caption='Detected image',use_container_width=True)

        

