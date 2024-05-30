#Imports for flask and connection to javascirpt
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
#import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import faiss
import requests
import pandas as pd
import torch
import datasets
from sentence_transformers.util import semantic_search

#Code for Model Starts here
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

#Following can be used if the dataset is online
#faqs_embeddings = datasets.load_dataset('Zarcend/testDataSet')

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]

output = query(texts)
embeddings = pd.DataFrame(output)

dataset_embeddings = torch.from_numpy(embeddings.to_numpy()).to(torch.float)


data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
question = "how many movies does Leonardo Di Caprio have?"

tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")

print(tqa(table=table, query=question)['cells'][0])

def getchat(input):
    question = [input]
    output = query(question)

    query_embeddings = torch.FloatTensor(output)

    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=1)
    top_k_texts = [texts[hit['corpus_id']] for hit in hits[0]]
    #first_result = top_k_texts[0]
    return top_k_texts





#embeddings.to_csv("embeddings.csv", index=False)
#print(embeddings)

#Connecting to Front end
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/process_data', methods=['POST'])
# def process_data():
#     data = request.json  # Extract JSON data sent from JS
#     input = data['message']
#     result = {"response": getchat(input)}
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)  # Run the Flask app
