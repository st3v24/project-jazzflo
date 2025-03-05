import streamlit as st
import pandas as pd
import torch
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    weights = "trained_weights/yolov7_best_v3.pt"
    model = torch.hub.load("yolov7","custom",f"{weights}",source="local",trust_repo=True)
    return model

def load_image(img_file_or_path):
    actual_image = Image.open(img_file_or_path).resize((640,640))
    img = np.array(actual_image)
    return img

@st.cache_data
def run_inference(_model,img):
    results = _model(img)
    df = results.pandas().xyxy[0]
    result_df = df[['class','xmin', 'ymin', 'xmax', 'ymax', 'confidence']]
    result_img = results.render()
    return result_df, result_img


def main():
    st.title('Jasmine Flower Detection')
    # left_column, middle_column, right_column = st.columns(3)
    st.session_state['running'] = False
    # model = load_model()
    
    # with left_column:
    upload = st.file_uploader('Upload Image')
    if upload is not None:
        img = load_image(upload)
        st.session_state['img'] = img
        st.session_state['running'] = st.button('Run Inference')    
    
    # with middle_column:
    if st.session_state['running']:
        model = load_model()
        result_df, result_img = run_inference(model,img)
        st.session_state['result_img'] = result_img
        st.dataframe(result_df)
    
    # with right_column:
    if st.session_state.get('result_img') is not None:
        result_img = st.session_state.get('result_img')
        if result_img is not None:
            st.image(result_img, caption='Result Image', use_container_width=True)

if __name__ == '__main__':
    main()