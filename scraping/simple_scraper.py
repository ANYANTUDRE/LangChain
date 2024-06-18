from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.chains import create_extraction_chain
from langchain_community.llms import Ollama

import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST")

llm = Ollama(model="mistral") 
schema = {
    "properties":{"Mooré": {"type": "string"},
                  "francais": {"type": "string"},
                },
    "required": ["Mooré", "francais"],
}

def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).invoke(content)


def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["h1", "p"]
    )
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content


urls = ["https://raamde-bf.net/minist%c9%9b%c9%9br-ning-se%cc%83n-get-laaf%c9%a9-wa%cc%83-ya%cc%83ka-yam-n-na-n-zab-ne-pali-de%cc%83nga%cc%83/"]
extracted_content = scrape_with_playwright(urls, schema=schema)
