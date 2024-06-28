from gradio_client import Client
import json

prompt = [
    {'image': 'frigo2.jpg'}, # Either a local path or an url
    {'text': 'Describe this image'},
]

# Save the prompt to a JSON file
with open('prompt.json', 'w') as f:
    json.dump(prompt, f)


client = Client("https://qwen-qwen-vl-max.hf.space/--replicas/ctmnh/")
result = client.predict(
		prompt,	# str (filepath to JSON file) in 'Qwen-VL-Max' Chatbot component
		"Describe this image",	# str  in 'Input' Textbox component
		fn_index=0
)
print(result)