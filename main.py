from time import sleep
import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from merge_dataset import merge_dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading dataset knowledgator/Scientific-text-classification")
dataset = merge_dataset(load_dataset("knowledgator/Scientific-text-classification", split='train'))
print(dataset[0])

print("Loading tokenizer from openai-community/gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", use_fast=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

print("Loading model from openai-community/gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium").to(device)

label_counts = Counter(dataset['label'][:50000]) # Past 50k samples, the dataset labels have only 1 sample each.

def getResponse(history, model, tokenizer):
    input_ids = tokenizer.encode(history, return_tensors='pt').to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float().to(device)
    output = model.generate(input_ids, 
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=attention_mask, 
                            #max_length=500, 
                            max_new_tokens=500,
                            do_sample=True, 
                            top_k=50, top_p=0.95, temperature=0.8, num_beams=5,
                            num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

conversationHistory = ""
print("Bot: Hi, how can I help you today?")
while True:
    # Handle user's input
    userInput = input("\nYou: ")
    if userInput == "exit": break
    conversationHistory += f"User: {userInput}\nBot: "
    response = getResponse(conversationHistory, model, tokenizer)[len(conversationHistory):]
    
    # Truncate the response to remove the duplication of the conversation
    if response.find('User:') != -1: response = response[:response.find('User:')]
    elif response.find('Bot:') != -1: response = response[:response.find('Bot:')]

    # Print the response word by word
    print("\nBot: ", end='', flush=True)
    for word in response.split():
        print(word, end=' ', flush=True)
        sleep(0.05)
    print()

    conversationHistory += f"{response}\n"
