from scrapegraphai.graphs import SmartScraperGraph

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST")


graph_config = {
    "llm": {
        "model": "ollama/mistral",
        "temperature": 0,
        "format": "json",  # Ollama needs the format to be specified explicitly
        "base_url": "https://2ca7-34-168-189-244.ngrok-free.app",  # set Ollama URL
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "https://2ca7-34-168-189-244.ngrok-free.app",  # set Ollama URL
    },
    "verbose": True,
}

smart_scraper_graph = SmartScraperGraph(
    prompt="Dans le texte, il y a une parti en en langue Mooré puis une autre en langue Francais. Récupère moi chaque paragraphe dans les 2 langues",
    # also accepts a string with the already downloaded HTML code
    source="https://raamde-bf.net/minist%c9%9b%c9%9br-ning-se%cc%83n-get-laaf%c9%a9-wa%cc%83-ya%cc%83ka-yam-n-na-n-zab-ne-pali-de%cc%83nga%cc%83",
    config=graph_config
)

result = smart_scraper_graph.run()
print(result)


# Prettify the result and display the JSON
import json

output = json.dumps(result, indent=2)  # Convert result to JSON format with indentation

line_list = output.split("\n")  # Split the JSON string into lines

# Print each line of the JSON separately
for line in line_list:
    print(line)