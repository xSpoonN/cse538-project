from time import sleep
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading tokenizer from openai-community/gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", use_fast=True, padding_side='left')
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

print("Loading tokenizer from distilbert/distilroberta-base")
classTokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", use_fast=True)
classTokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Loading classifier from checkpoint")
classifier = AutoModelForSequenceClassification.from_pretrained("classifier", num_labels = 5).to(device)

print("Loading experts from checkpoints")
experts = {
    "computer science": AutoModelForCausalLM.from_pretrained("expert_CS").to(device),
    "electrical engineering and systems science": AutoModelForCausalLM.from_pretrained("expert_EE").to(device),
    "general physics": AutoModelForCausalLM.from_pretrained("expert_GP").to(device),
    "theoretical physics": AutoModelForCausalLM.from_pretrained("expert_TP").to(device),
    "general mathematics": AutoModelForCausalLM.from_pretrained("expert_GM").to(device)
}
class_list = [
    'general mathematics',
    'computer science',
    'general physics',
    'electrical engineering and systems science',
    'theoretical physics'
]
print("Loaded all models\n")

def convert_classes_to_label(classes):
    idx = 0
    max_conf = 0
    for i in range(len(classes)):
        if classes[i] > max_conf:
            max_conf = classes[i]
            idx = i
    return class_list[idx]

def getResponse(history, tokenizer):
    inputs = classTokenizer(history, return_tensors='pt', truncation=True, padding=True, max_length=500).to(device)
    classifier.eval()
    outputs = classifier(**inputs)
    pred = convert_classes_to_label(torch.clamp(outputs.logits, min=1e-6, max=1).tolist()[0])

    input_ids = tokenizer.encode(history, return_tensors='pt', truncation=True).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float().to(device)
    model = experts[pred]
    #model = AutoModelForCausalLM.from_pretrained("expert_CS").to(device) # For testing purposes
    output = model.generate(input_ids,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=attention_mask, 
                            #max_length=500, 
                            max_new_tokens=500,
                            do_sample=True, 
                            top_k=50, top_p=0.95, temperature=0.8, num_beams=5,
                            num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)
    return response

conversationHistory = ""
print("Bot: Hi, how can I help you today?")
while True:
    # Handle user's input
    userInput = input("\nYou: ")
    if userInput == "exit": break
    conversationHistory += f"User: {userInput}\nBot: "
    response = getResponse(conversationHistory, tokenizer)[len(conversationHistory):]
    
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
