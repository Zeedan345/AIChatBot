# Databricks notebook source
#Installing the reqiured external libariries

%pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22
dbutils.library.restartPython()

# COMMAND ----------

#Converting Text to Embedings

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

#Embeddings endpoints convert text into a vector (array of float). Here is an example using BGE:
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# MAGIC

# COMMAND ----------

# MAGIC     %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `TRAPS_MAP_DATA_csv`

# COMMAND ----------

dataFrame = sqlContext.sql('select * from `TRAPS_MAP_DATA_csv`')

# COMMAND ----------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd


tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

table = sqlContext.sql('select * from `mac_data_1_edit_2_csv`')

table_df = table.toPandas()
table_df.fillna('', inplace=True)

# Truncate the table to a manageable size (e.g., 100 rows and 10 columns)
max_rows = 100
max_cols = 10
truncated_table_df = table_df.iloc[:max_rows, :max_cols]

query = "Most useful provider in Jackson"
encoding = tokenizer(table=truncated_table_df, query=query, return_tensors="pt", truncation=True, max_length=1024)


outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))



# f.close()



# COMMAND ----------


