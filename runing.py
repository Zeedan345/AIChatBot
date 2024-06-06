from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch

# Define the model name
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load the tokenized data
tokenized_context = torch.load('tokenized_context.pt')

# Define the model name
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name, device_map = 'cuda')

# Initialize the pipeline with the loaded model and tokenizer
nlp = pipeline('question-answering', model=model, tokenizer=AutoTokenizer.from_pretrained(model_name))

# Define the QA input with the question and tokenized context
QA_input = {
    'question': 'What is the best provider in Starkville',
    'context': tokenized_context
}

# Use the pipeline to get predictions
# Here we need to convert tokenized inputs back to text for the pipeline to understand
inputs = tokenizer.decode(tokenized_context['input_ids'][0])
QA_input['context'] = inputs

res = nlp(QA_input)

# Print the result
print(res)
