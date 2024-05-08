import streamlit as st
from icecream import ic


st.write('Hello *World*! :sunglasses:')

files = st.file_uploader("Upload Research Paper","pdf",True,)



for file in files:
    st.write("filename:", file.name)
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())