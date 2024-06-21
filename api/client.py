import requests
import streamlit as st


def get_qween_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke", 
                             json={"input": {"topic": input_text}})
    return response.json()['output']


def get_deepseek_response(input_text):
    response = requests.post("http://localhost:8000/code/invoke", 
                             json={"input": {"topic": input_text}})
    return response.json()['output']


st.title("Demo Qween2 and DeepSeekCoder")

input_text  = st.text_input("Write an poem topic here...")
input_text2 = st.text_input("Write a model name here...")

if input_text:
    st.write(get_qween_response(input_text))

if input_text2:
    st.write(get_deepseek_response(input_text2))