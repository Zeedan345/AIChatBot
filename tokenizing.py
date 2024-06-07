from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json
import torch

# Load JSON data
with open('data_compact.json', encoding="utf8") as f:
    data = json.load(f)

# Process the JSON data
# for entry in data:
#     for key, value in entry.items():
#         if value == 0:
#             entry[key] = "False"
#         elif value == 1:
#             entry[key] = "True"
#         elif isinstance(value, (int, float)):
#             entry[key] = str(value)
#         elif value is None:
#             entry[key] = "Null"


def process_value(value):
    if value == 0:
        return "False"
    elif value == 1:
        return "True"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return value

# Apply the function to each element in the DataFrame
df_processed = df.applymap(process_value)



# Concatenate context string
context = ''
context += ' '.join(json.dumps(entry) for entry in data)

# Define the model name
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the context
tokenized_context = tokenizer(context, return_tensors='pt')

# Save the tokenized data
torch.save(tokenized_context, 'tokenized_context.pt')
print("Tokenized context saved.")
