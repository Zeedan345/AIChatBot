import torch
import json
import ast
from sentence_transformers import SentenceTransformer
from transformers import TapasTokenizer

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd

df = pd.read_csv('train100.csv', encoding='unicode_escape')
df = df.astype(str)


questions = pd.read_csv('TrainingQuestions.csv', encoding='unicode_escape')

#train_df = questions.astype(str)
train_df = questions.sample(frac=1.0, random_state=42, replace=False).reset_index(drop=True)
train_df['position'] = range(len(train_df))
train_df['answer_coordinates'] = train_df['answer_coordinates'].apply(ast.literal_eval)
batch = df
train_df


# Initialize the tokenizer
tokenizer = TapasTokenizer.from_pretrained("Zarcend/AFFIRM-FineTune-25")

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        table = pd.read_csv('train100.csv', encoding =  'unicode_escape').astype(str) # TapasTokenizer expects the table data to be text only
        table = table.sample(frac=1.0, random_state=42, replace=False).reset_index(drop=True)
        if item.position != 0:
          # use the previous table-question pair to correctly set the prev_labels token type ids
          previous_item = self.df.iloc[idx-1]
          encoding = self.tokenizer(table=table,
                                    queries=[previous_item.question, item.question],
                                    answer_coordinates=[previous_item.answer_coordinates, item.answer_coordinates],
                                    answer_text=[previous_item.answer_text, item.answer_text],
                                    float_answer=[previous_item.float_answer, item.float_answer],
                                    padding=True,
                                    #truncation=True,
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
                                    padding=True,
                                    #truncation=True,
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

model = TapasForQuestionAnswering.from_pretrained("Zarcend/AFFIRM-FineTune-25")
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


model.save_pretrained("AFFIRM-fine-tuned-tapas-25")
tokenizer.save_pretrained("AFFIRM-fine-tuned-tapas-25")
