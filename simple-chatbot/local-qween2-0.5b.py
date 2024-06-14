from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st

import os
import getpass
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# promt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a helpful assistant. Respond to the user query"),
        ("user", "Question: {question}")
    ]
)

# streamlit ui
st.title("Langchain Demo Qween2")
input_text = st.text_input("Write somthing here...")

# llm 
llm = Ollama(model="qwen2:0.5b")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))