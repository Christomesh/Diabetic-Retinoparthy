import streamlit as st
from PIL import Image
import pathlib
import cv2
import numpy as np
from fastai.vision.all import *

pathlib.PosixPath = pathlib.WindowsPath
path = Path('models')

st.title('Diabetic Retinoparthy DashBoard')


def main():
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")

        st.write('Using ResNet 152 ...')
        learner = load_learner(path/'resnet18_deployed.pkl')
        preds = learner.predict(opencv_image)[0][::-1][:-1]
        st.write("Features present")
        st.write(preds)

main()

