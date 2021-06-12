import streamlit as st
import pathlib
from fastai.vision.all import *

pathlib.PosixPath = pathlib.WindowsPath
path = Path('models')

st.title('Diabetic Retinoparthy DashBoard')


def main():
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        st.image(img.to_thumb(500,500), caption='Uploaded Image')

        st.write('Using ResNet 152 ...')
        learner = load_learner(path/'resnet18_deployed.pkl')
        preds = learner.predict(img)[0][::-1][:-1]
        st.write("Features present")
        st.write(preds)

main()

