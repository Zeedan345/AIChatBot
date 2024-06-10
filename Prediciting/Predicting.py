# Databricks notebook source

pip install pinecone-client

# COMMAND ----------

pip install sentence_transformers

# COMMAND ----------

import torch
from sentence_transformers import SentenceTransformer

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the table embedding model from huggingface models hub
retriever = SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)
retriever

# COMMAND ----------

import pandas as pd

# Read the CSV file
data = sqlContext.sql('select * from `data`')
df = data.toPandas()

def process_value(value):
    if value == 0:
        return True
    elif value == 1:
        return False
    else:
        return value

# Apply the function to each element in the DataFrame
df = df.applymap(process_value)

tables = []
# Iterate through the DataFrame in chunks of 10 rows
chunk_size = 5
for start in range(0, len(df), chunk_size):
    # Select 10 rows starting from the current index
    chunk = df.iloc[start:start+chunk_size]
    tables.append(chunk)


# COMMAND ----------

#Converting the table to string
def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed

# COMMAND ----------

# format all the dataframes in the tables list
processed_tables = _preprocess_tables(tables)


# COMMAND ----------

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key='6315210d-75de-4bdd-b3c1-ffcb5ca3b35c')


# COMMAND ----------

# you can choose any name for the index
index_name = "table-qa"

# Check if the table-qa index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# COMMAND ----------


from tqdm.auto import tqdm

# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(processed_tables), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(processed_tables))
    # extract batch
    batch = processed_tables[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch).tolist()
    # create unique IDs ranging from zero to the total number of tables in the dataset
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

# check that we have all vectors in index
index.describe_index_stats()
     

# COMMAND ----------

query = "Give me 5 providers in Tupelo who offer medicaid"
# generate embedding for the query
xq = retriever.encode([query]).tolist()
# query pinecone index to find the table containing answer to the query
result = index.query(vector=xq, top_k=3)
result

# COMMAND ----------

# Get the ids for indices 0, 1, and 2
ids = [int(result["matches"][i]["id"]) for i in range(3)]

# Extract the head (first few rows) of each table corresponding to ids 0, 1, and 2
heads = [tables[id].head() for id in ids]

# Concatenate the heads into a single DataFrame
batch = pd.concat(heads, ignore_index=True)




# COMMAND ----------

# Load model directly
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")
#model.max_seq_length = 8192

# COMMAND ----------


batch.fillna('', inplace=True)
batch = batch.sample(frac=1.0, random_state=42, replace=False).reset_index(drop=True).astype(str)
batch


# COMMAND ----------

# Encode the table and query
inputs = tokenizer(table=batch, queries=query, padding="max_length", return_tensors="pt")

# Pass the inputs to the model
outputs = model(**inputs)

# Interpret the logits to get the answer
logits = outputs.logits
logits_agg = outputs.logits_aggregation

# Get the answer from the model's output
predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs,
    logits.detach(),
    logits_agg.detach()
)

answers = []
for coordinates in predicted_answer_coordinates:
    cell_values = []
    for coord in coordinates:
        try:
            cell_values.append(batch.iat[coord])
        except IndexError:
            print(f"Index {coord} is out of bounds for the DataFrame with shape {batch.shape}")
            continue
    if cell_values:
        answers.append(", ".join(cell_values))

print("Answer:", answers)

