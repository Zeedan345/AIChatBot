from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json

# Load JSON data efficiently
with open('data_compact.json', encoding="utf8") as f:
    data = json.load(f)

# Process the JSON data using list comprehensions
# for entry in data:
#     for key, value in entry.items():
#         if value == 0:
#             entry[key] = "False"
#         elif value == 1:
#             entry[key] = "True"
#         elif value is None:
#             entry[key] = "Null"

# Define the model name
model_name = "deepset/roberta-base-squad2"

# Load the model and tokenizer once
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the pipeline+
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Efficiently concatenate context string
string = ''
string += ' '.join(json.dumps(entry) for entry in data)

# Define the QA input
QA_input = {
    'question': 'Who is the most useful provider in Starkville',
    'context': string
}

# Get predictions
res = nlp(QA_input)

# Print the result
print(res)

# No need to explicitly close the file when using 'with' statement
