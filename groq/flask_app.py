from flask import Flask, request, jsonify, render_template
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")


app = Flask(__name__)


"""------------------------------------------ChatBot------------------------------------"""
### load mistral model
llm = Ollama(model="mistral", base_url=OLLAMA_HOST)
print("LLM Initialized....")

prompt_template = """<s>[INST] Tu es un assistant, expert en agriculture et en agronomie qui doit communiquer et aider un agriculteur dans le besoin. Réponds à ses questions de facon détaillée et structurée.
                            Contexte: {context}
                            Agriculteur: {question}
                            IA: [/INST]"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
load_vector_store = Chroma(persist_directory="stores/koobo_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})


@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/get_chat_response', methods=['POST'])
def get_chat_response():
    query = request.form.get('query')
    # Your logic to handle the query
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    answer = response['result']
    #source_document = response['source_documents'][0].page_content
    source_documents = response['source_documents']
    source_documents_list = []
    page_number_list = []
    for doc in source_documents:
        source_doc = doc.metadata['source']
        page_number = doc.metadata['page']
        if source_doc not in source_documents_list:
            source_documents_list.append(source_doc)
            page_number_list.append(page_number)

    #doc = response['source_documents'][0].metadata['source']
    #response_data = {"answer": answer, "source_document": source_document, "doc": doc}
    response_data = {"answer": answer, "source_documents_list": source_documents_list, "page_number_list": page_number_list}
    return jsonify(response_data)




"""------------------------------------Détection Maladies----------------------------"""
### load LLAVA model
llava = Ollama(model="llava-phi3", base_url=OLLAMA_HOST)
print("LLAVA Initialized....")

def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/get_detect_response', methods=['POST', 'GET'])
def get_detect_response():
    if 'image' in request.files:
        #print(request.files)
        image_file = request.files['image']
        pil_image = Image.open(image_file.stream)
        pil_image = pil_image.convert('RGB')
        image_b64 = convert_to_base64(pil_image)

        llm_with_image_context = llava.bind(images=[image_b64])
        result = llm_with_image_context.invoke(request.form.get('query'))
        response_data = {"result": result}
        return jsonify(response_data)
    else:
        return jsonify({"error": "No image uploaded"}), 400



"""------------------------------------Détection Maladies avec Ollama----------------------------"""
### load LLAVA model
llava = Ollama(model="llava", base_url=OLLAMA_HOST)
print("LLAVA Initialized....")

def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/get_detect_response', methods=['POST', 'GET'])
def get_detect_response():
    if 'image' in request.files:
        #print(request.files)
        image_file = request.files['image']
        pil_image = Image.open(image_file.stream)
        pil_image = pil_image.convert('RGB')
        image_b64 = convert_to_base64(pil_image)

        llm_with_image_context = llava.bind(images=[image_b64])
        result = llm_with_image_context.invoke(request.form.get('query'))
        response_data = {"result": result}
        return jsonify(response_data)
    else:
        return jsonify({"error": "No image uploaded"}), 400




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
















if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)