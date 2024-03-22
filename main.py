#import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import faiss
import requests
import pandas as pd
import torch
import datasets
from sentence_transformers.util import semantic_search


model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

faqs_embeddings = datasets.load_dataset('Zarcend/testDataSet')

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

dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

question = ["How can Medicare"]
output = query(question)

query_embeddings = torch.FloatTensor(output)

hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

print([texts[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])



#embeddings.to_csv("embeddings.csv", index=False)
#print(embeddings)

