import torch
import json
from sentence_transformers import SentenceTransformer

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd

df = pd.read_csv('train.csv', encoding='unicode_escape')

def process_value(value):
    if value == 0:
        return 0.0
    elif value == 1:
        return 1.0
    else:
        return value

# Apply the function to each element in the DataFrame
df = df.applymap(process_value)
df = df.astype(str)

from transformers import TapasTokenizer
df = df[:25]
# Sample questions generation
questions = []
answers = []
coordinates = []
float_answers = []

# Define sample questions and their corresponding columns
sample_questions = [
    ("What is the phone number of the provider \"{}\"?", "Phone1_Number", 7),
    ("Which provider is located in {}?", "Provider City", 3),
    ("What is the website URL for \"{}\"?", "WebsiteUrl", 10),
    ("What is the postal code for \"{}\"?", "PostalCode", 5),
    ("Which provider in {} offers services?", "Provider City", 3),
    ("What type of phone is listed as the main contact for \"{}\"?", "Phone1_Type", 6),
]

# Generate questions for each provider in the table
for index, row in df.iterrows():
    for question_template, answer_col, answer_coord in sample_questions:
        if pd.notna(row[answer_col]) and row[answer_col] != '':
            question = question_template.format(row['Provider Name'])
            answer = row[answer_col]
            questions.append(question)
            answers.append(answer)
            coordinates.append([(index, df.columns.get_loc(answer_col))])
            float_answers.append([float('nan')])
# Create a DataFrame for the generated QA pairs
qa_df = pd.DataFrame({
    'question': questions,
    'answer_text': answers,
    'answer_coordinates': coordinates,
    'float_answer': float_answers
})
qa_df = qa_df[5:15]
# Shuffle and reset index of the DataFrame
train_df = qa_df.sample(frac=1.0, random_state=42, replace=False).reset_index(drop=True)
# Add a 'position' column to the train_df DataFrame
train_df['position'] = range(len(train_df))

batch = df.fillna('').astype(str)
train_df

# Initialize the tokenizer
tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-sqa")

# Tokenize the input table and questions
encoding = tokenizer(
    table=batch,
    queries=train_df['question'].tolist(),
    answer_coordinates=train_df['answer_coordinates'].tolist(),
    answer_text=train_df['answer_text'].tolist(),
    float_answer=train_df['float_answer'].tolist(),
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)
encoding.keys()

from torch.utils.data import DataLoader, TensorDataset

# Create a TensorDataset and DataLoader
dataset = TensorDataset(
    encoding['input_ids'],
    encoding['attention_mask'],
    encoding['token_type_ids'],
    encoding['labels'],
)
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        table = pd.read_csv('train.csv', encoding =  'unicode_escape').astype(str) # TapasTokenizer expects the table data to be text only
        table = table[:25]
        if item.position != 0:
          # use the previous table-question pair to correctly set the prev_labels token type ids
          previous_item = self.df.iloc[idx-1]
          encoding = self.tokenizer(table=table,
                                    queries=[previous_item.question, item.question],
                                    answer_coordinates=[previous_item.answer_coordinates, item.answer_coordinates],
                                    answer_text=[previous_item.answer_text, item.answer_text],
                                    float_answer=[previous_item.float_answer, item.float_answer],
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt"
          )
          # use encodings of second table-question pair in the batch
          encoding = {key: val[-1] for key, val in encoding.items()}
        else:
          # this means it's the first table-question pair in a sequence
          encoding = self.tokenizer(table=table,
                                    queries=item.question,
                                    answer_coordinates=item.answer_coordinates,
                                    answer_text=item.answer_text,
                                    float_answer= item.float_answer,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt"
          )
          # remove the batch dimension which the tokenizer adds
          encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        return encoding

    def __len__(self):
        return len(self.df)

train_dataset = TableDataset(df=train_df, tokenizer=tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2)

train_dataset[0]["token_type_ids"].shape

train_dataset[1]["input_ids"].shape

batch = next(iter(train_dataloader))

batch["input_ids"].shape

batch["token_type_ids"].shape

tokenizer.decode(batch["input_ids"][0])

#first example should not have any prev_labels set
assert batch["token_type_ids"][0][:,3].sum() == 0

tokenizer.decode(batch["input_ids"][1])

assert batch["labels"][0].sum() == batch["token_type_ids"][1][:,3].sum()
print(batch["token_type_ids"][1][:,3].sum())

from transformers import TapasForQuestionAnswering

model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-sqa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.max_seq_length = 8192
model.to(device)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):  # loop over the dataset multiple times
   print("Epoch:", epoch)
   for idx, batch in enumerate(train_dataloader):
        # get the inputs;
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                       labels=labels)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()




model.save_pretrained("AFFIRM-fine-tuned-tapas-25", safe_serialization=False)
tokenizer.save_pretrained("AFFIRM-fine-tuned-tapas-25", safe_serialization=False)