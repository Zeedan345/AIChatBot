from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

import json

f = open('mac_v4.json', encoding = "utf8")
data = json.load(f)
# for i in range(len(data)):  # Iterate over the length of the list
#     for key, value in data[i].items():  # Iterate through key-value pairs in each dictionary
#         if value ==0:
#             data[i][key] = "False"
#         elif value ==1:
#             data[i][key] = "True"
#         elif isinstance(value, (int, float)):
#             data[i][key] = str(value)
#         elif value == None:
#             data[i][key] = "NUll"


tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

table = pd.DataFrame.from_dict(data)
table.to_excel('output.xlsx', index=False)

query = "Which city are the most amount of providers located in"
encoding = tokenizer(table=table, query=query, return_tensors="pt", truncation = True, max_length = 1024)


outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))



# f.close()

