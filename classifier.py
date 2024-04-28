from transformers import RobertaModel, AutoTokenizer, AdamW
import numpy as np
import torch 
from torch import nn

import matplotlib.pyplot as plt

from tqdm import tqdm

# for test
from merge_dataset import merge_dataset
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_list = [
    'general mathematics',
    'computer science',
    'general physics',
    'electrical engineering and systems science',
    'theoretical physics'
]

# make label to answer
def make_classes(train_set):
    answer = []
    for data in train_set:
        arr = []
        for i in range(5):
            arr.append(1 if i == class_list.index(data['label']) else 0)
        answer.append(arr)
    return answer

# convert output of model into label
def convert_classes_to_label(classes):
    idx = 0
    max_conf = 0
    for i in range(len(classes)):
        if classes[i] > max_conf:
            max_conf = classes[i]
            idx = i
    return class_list[idx]

def train_one(model, tokenizer, batched_train_set, answer_tenser, optimizer, loss_func):

    model.train().to(device)
    losses = []

    index = 0
    for data in tqdm(batched_train_set, desc = "Progress of train_one"):
        inputs = tokenizer(data, return_tensors='pt', truncation=True, padding=True, max_length=500).to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        pred = torch.clamp(outputs.pooler_output, min=1e-6, max=1)
        # print(answer_tenser[index])
        loss = loss_func(pred, answer_tenser[index].to(device))
        # backward: /(applies gradient descent)
        loss.backward()
        optimizer.step()
        index += 1

        losses.append(loss.item())
    return np.mean(losses)

def train_classifier(model, tokenizer, train_set, batch = 1, epochs = 5, lr = 1e-5, weight_decay = 0.01):

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    string_train_set = []
    for data in train_set:
        string_train_set.append(data['text'])
    answers = make_classes(train_set)

    batched_train_set = []
    bacthed_answers = []
    for i in range(0, len(string_train_set), batch):
        temp_data = []
        temp_answer = []
        for j in range(batch):
            if i + j < len(string_train_set):
                temp_data.append(string_train_set[i+j])
                temp_answer.append(answers[i+j])
        batched_train_set.append(temp_data)
        bacthed_answers.append(torch.Tensor(temp_answer))


    optimizer = AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
    loss_func = torch.nn.BCELoss()

    loss_values = []
    for epoch in tqdm(range(epochs), desc = "Progress of train_classifier"):
        loss = train_one(model, tokenizer, batched_train_set, bacthed_answers, optimizer, loss_func)
        loss_values.append(loss)

    # Generate an image of the loss curves
    plt.clf()
    plt.plot(range(epochs), loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('loss curve of classifier.png')

if __name__ == '__main__':
    model = RobertaModel.from_pretrained('distilroberta-base')
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", use_fast=True)

    dataset = merge_dataset(load_dataset("knowledgator/Scientific-text-classification", split='train'))
    dataset = dataset.select(range(len(dataset)//4))

    # change the last layer to create 5 output
    model.pooler.dense = nn.Linear(768, 5)


    train_classifier(model, tokenizer, dataset, batch = 1, epochs = 5, lr = 1e-5, weight_decay = 0.01)