from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering
from transformers import pipeline
import pandas as pd
import json

f = open('mac_json.json', encoding ="utf8")
data = json.load(f)
for i in range(len(data)):  # Iterate over the length of the list
    for key, value in data[i].items():  # Iterate through key-value pairs in each dictionary
        if isinstance(value, (int, float)):
            data[i][key] = str(value)
        elif value == None:
            data[i][key] = "NUll"



tokenizer = AutoTokenizer.from_pretrained("google/tapas-large-finetuned-wtq", drop_rows_to_fit=True)
model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")




#prepare table + question
    
table = pd.DataFrame.from_dict(data)
question = "Which county has the most clinics"
# pipeline model
# Note: you must to install torch-scatter first.
tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")


inputs = tokenizer(table=table, queries=question, truncation=True, max_length= 256,  return_tensors="pt")
# result

outputs = model(**inputs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
#53

f.close()