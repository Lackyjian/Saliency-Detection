import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from PIL import Image

gen_256 = tf.keras.models.load_model('generator_256_256.h5')
gen_512 = tf.keras.models.load_model('generator_512_512.h5')


def pred(img, model, add_to_channel = False, channel = 0):
    try:
        if model == gen_256:
            size = (256, 256)
        else:
            size = (512, 512)
        img = np.array(img)
        img = tf.image.resize(img, size)
        img = (img-127.5)/127.5
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        pred = (pred+1)/2
        pred = np.squeeze(pred)
        img = np.squeeze(img)
        img = (img+1)/2

        st.image(img, caption = 'Original')
        st.image(pred, caption = 'Saliency Map')
        if add_to_channel:
            img[:,:,channel] += pred*6
        else:
            img[:,:,channel] = pred*7
        #clipping
        img = np.clip(img, 0, 1)
        st.image(img, caption = 'Combined')
    except ValueError:
        st.warning('There is some problem with image, try uploading another one!')


st.title('Image Saliency Predictor')
img_up = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
st.text('Or')
img_cam = st.camera_input('Take a photo')

if img_cam is not None:
    img = Image.open(img_cam)
elif img_up is not None:
    img = Image.open(img_up)

st.divider()

channel = st.selectbox('Select Highlighting channel', ['red', 'green', 'blue'])
channel_dict = {'red':0, 'green':1, 'blue':2}

model = st.radio('Select model', ['256x256', '512x512'])
model_dict = {'256x256':gen_256, '512x512':gen_512}

add_to_channel = st.checkbox('Add mask to channel')
if st.button('Predict'):
    pred(img, model_dict[model], add_to_channel, channel_dict[channel])