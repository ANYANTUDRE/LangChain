from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn
import os


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Langchain server",
    version="1.0",
    description="API server"
)

# we're using 2 llms: qween2 and deepseek-coder:1.3b
qween_model = Ollama(model="qwen2:0.5b")
deepseek_model = Ollama(model="deepseek-coder:1.3b")

# prompts templates
prompt1 = ChatPromptTemplate.from_template("Write me a love poem about {topic} with 50 words")
prompt2 = ChatPromptTemplate.from_template("Write me a python code to train a simple {topic} machine learning model")

add_routes(
    app,
    prompt1 | qween_model,
    path="/poem"
)

add_routes(
    app,
    prompt2 | deepseek_model,
    path="/code"
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)