import streamlit as st
from PIL import Image

image = Image.open('images\image_1.jpg')
st.image(image, caption='Suasana hutan yang sunyi')

audio_file = open('audio\\audio_1.wav', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/wav')

video_file = open('video\\nature.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)