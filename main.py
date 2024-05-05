from time import sleep
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\r[. . . .] Loading tokenizer from openai-community/gpt2-medium ", end='')
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", use_fast=True, padding_side='left')
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

print("\r[✔ . . .] Loading tokenizer from distilbert/distilroberta-base", end='')
classTokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", use_fast=True)
classTokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("\r[✔ ✔ . .] Loading classifier from checkpoint                  ", end='')
classifier = AutoModelForSequenceClassification.from_pretrained("classifier", num_labels = 5).to(device)

print("\r[✔ ✔ ✔ .] Loading experts from checkpoints                    ", end='')
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
print("\r[✔ ✔ ✔ ✔] Loaded all models                                   \n")

def convert_classes_to_label(classes):
    idx = 0
    max_conf = 0
    for i in range(len(classes)):
        if classes[i] > max_conf:
            max_conf = classes[i]
            idx = i
    return class_list[idx]

def getResponse(history, tokenizer):
    # Classify the topic of the conversation
    inputs = classTokenizer(history, return_tensors='pt', truncation=True, padding=True, max_length=500).to(device)
    classifier.eval()
    outputs = classifier(**inputs)
    pred = convert_classes_to_label(torch.clamp(outputs.logits, min=1e-6, max=1).tolist()[0])

    # Generate a response based on the topic
    input_ids = tokenizer.encode(history, return_tensors='pt', truncation=True).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float().to(device)
    model = experts[pred]
    # model = AutoModelForCausalLM.from_pretrained("mastermind").to(device) # For testing against the base GPT2 Model.
    output = model.generate(input_ids,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=attention_mask, 
                            #max_length=500, 
                            max_new_tokens=500, # Generate up to 500 tokens
                            do_sample=True, # Sample from the distribution
                            top_k=50, top_p=0.95, temperature=1, num_beams=5, # Take the top 50 tokens with at least 95% probability and a temperature of 0.8. Beam width of 5.
                            num_return_sequences=1) # Generate 1 sequence
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-process the response
    response = response.strip() # Remove leading and trailing whitespaces
    response = response[len(history):] # Remove the duplicated conversation
    if response.find('User:') != -1: response = response[:response.find('User:')] # Truncate the response to remove the duplication of the conversation
    elif response.find('Bot:') != -1: response = response[:response.find('Bot:')]
    response = ". ".join(list(OrderedDict.fromkeys(response.split(". ")))) # Remove duplicate sentences
    response = response[:response.rfind(".")+1] # Delete incomplete sentences

    print("Classified as:", pred)
    return response

conversationHistory = ""
print("Bot: Hi, how can I help you today?")
while True:
    userInput = input("\nYou: ")
    if userInput == "exit": break
    conversationHistory += f"User: {userInput}\nBot: "
    response = getResponse(conversationHistory, tokenizer)

    # Print the response word by word
    print("\nBot: ", end='', flush=True)
    for word in response.split():
        print(word, end=' ', flush=True)
        sleep(0.05)
    print()

    conversationHistory += f"{response}\n"
    conversationHistory = " ".join(conversationHistory.split()[-400:]) # Keep the last 400 words of the conversation
