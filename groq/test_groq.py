from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY
)

system = "Tu es un assistant, expert en agriculture et en agronomie qui doit communiquer et aider un agriculteur dans le besoin. Réponds à ses questions de facon détaillée et structurée."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
res = chain.invoke({"text": "La culture du maïs nécessite une pluviométrie supérieure à"})
print(res)