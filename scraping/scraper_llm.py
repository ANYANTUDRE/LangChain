from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains import create_extraction_chain
from langchain_community.llms import Ollama

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

import os
from dotenv import load_dotenv
import asyncio

import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OLLAMA_HOST"] = "https://051f-34-85-147-59.ngrok-free.app"

"""async def run_playwright(site):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(site)

        page_source = await page.content()
        soup = BeautifulSoup(page_source, "html.parser")
        
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        data = '\n'.join(chunk for chunk in chunks if chunk)
        await browser.close()
    return data

async def main():
    output = await run_playwright("https://raamde-bf.net/a-kemi-seba-bee-burki%cc%83na-faso/")
    return output"""

# Run the asynchronous function and handle the result
#output = asyncio.run(main())

llm = Ollama(model="qwen2:0.5b")
schema = {
    "properties":{"Mooré": {"type": "string"},
                  "francais": {"type": "string"},
                },
    "required": ["Mooré", "francais"],
}

def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter


def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span"]
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


urls = ["https://www.wsj.com"]
extracted_content = scrape_with_playwright(urls, schema=schema)




# Create the extraction chain
#extraction_chain = create_extraction_chain(schema=structured_schema, llm=llm)

# Run the extraction chain
#result = extraction_chain.invoke({"input": output})

