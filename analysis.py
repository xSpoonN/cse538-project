# Main file for the chatbot
# Team members:
# - Kevin Tao
# - Haneul Lee

# Description:
# A classifier is used to determine the topic of the conversation, and the appropriate expert model is used to generate a response.
# The classifier is trained on the full dataset, while the expert models are trained on the filtered datasets.
# The classifier is trained as a sequence classification model, while the expert models are trained as causal language models in a self-supervised manner.
# Input is appended to an existing conversation history, with 'User:' and 'Bot:' prefixes to differentiate between user input and bot responses.
# If the conversation length exceeds 400 words, the conversation history is truncated to the last 400 words to prevent max model input length errors.
# Responses are post-processed to remove duplicated conversations, incomplete sentences, and duplicate sentences.

# For the RAG model, uncomment the line in getResponse() and comment the line below it.
# RAG is done by encoding the user input and comparing it with the dataset embeddings using cosine similarity.
# The top k closest text entries are then concatenated and used to provide context to the system.

# NLP class concepts used:
# - [I. Tokenization] Tokenizing the input text and generating tokens for the model.
# - [I. Classification] Classifying the topic of the conversation using a sequence classification model.
# - [I. Gradient Descent] Optimizing the model using the AdamW optimizer.
# - [II. Vector Semantics] Generating embeddings for the dataset entries and user input.
# - [III. Transformers (autoencoding)] Training the classifier as a RoBERTa based sequence classification model.
# - [III. Transformers (autoregressive)] Training the expert models as causal language models.
# - [III. Fine-tuning] Fine-tuning the classifier and expert models on the filtered datasets.
# - [III. Zero/few-shot Generation] Generating responses using the models based on the conversation history, initially nothing.
# - [III. Beam Search] Applying beam search to generate multiple responses and select the best one.
# - [IV. Dialog] Creating a conversation history and generating responses based on the history.
# - [IV. Question Answering] Generating responses based on the user input, conversation history, and using RAG to retrieve context.

# System specifications during testing:
# Windows 11 Home 23H2 OS Build 22631.3447
# AMD Ryzen 7 5700 @ 3.7 GHz, 32 GB RAM DDR4 @ 3200 MHz CL16
# NVIDIA GeForce RTX 4060 Ti 8 GB GDDR6 DX12

from time import sleep
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from collections import OrderedDict
from rag import vectorize_text, get_closest_text
from sentence_transformers import SentenceTransformer
import pickle
from merge_dataset import merge_dataset
from datasets import load_dataset
import time
from tqdm import tqdm


def convert_classes_to_label(classes):
    idx = 0
    max_conf = 0
    for i in range(len(classes)):
        if classes[i] > max_conf:
            max_conf = classes[i]
            idx = i
    return class_list[idx]

def getResponse(history: str, tokenizer, topEntries) -> str:
    # Classify the topic of the conversation
    inputs = classTokenizer(history, return_tensors='pt', truncation=True, padding=True, max_length=500).to(device)
    classifier.eval()
    outputs = classifier(**inputs)
    pred = convert_classes_to_label(torch.clamp(outputs.logits, min=1e-6, max=1).tolist()[0])

    # Generate a response based on the topic
    input = history if topEntries is None else ' '.join([entry[:150] for entry in topEntries]) + '\n' + history
    input_ids = tokenizer.encode(input, return_tensors='pt', truncation=True, max_length=512, padding='max_length').to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float().to(device)
    model = experts[pred]
    # model = AutoModelForCausalLM.from_pretrained("mastermind").to(device) # For testing against the base GPT2 Model.
    output = model.generate(input_ids,
                            labels=input_ids,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=attention_mask, 
                            # max_length=512, 
                            max_new_tokens=500, # Generate up to 500 tokens
                            do_sample=True, # Sample from the distribution
                            top_k=50, top_p=0.95, temperature=1, num_beams=5, # Take the top 50 tokens with at least 95% probability and a temperature of 0.8. Beam width of 5.
                            num_return_sequences=1) # Generate 1 sequence
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # # Post-process the response
    # response = response.strip() # Remove leading and trailing whitespaces
    # response = response[len(history):] # Remove the duplicated conversation
    # if response.find('User:') != -1: response = response[:response.find('User:')] # Truncate the response to remove the duplication of the conversation
    # elif response.find('Bot:') != -1: response = response[:response.find('Bot:')]
    # response = ". ".join(list(OrderedDict.fromkeys(response.split(". ")))) # Remove duplicate sentences
    # response = response[:response.rfind(".")+1] # Delete incomplete sentences
    
    inputs_generated = tokenizer(response, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        outputs_generated = model(**inputs_generated, labels=inputs_generated["input_ids"])
        loss = outputs_generated.loss
    perplexity = torch.exp(loss).item()

    
    return response, perplexity

def get_perplexity(dataset):
    
    response_time_sum = 0
    perplexity_sum = 0
    for data in tqdm(dataset, desc = "Progress of get_perplexity"):
        start_time = time.time() * 1000.0 # get cur time in ms
        response, perplexity = getResponse(data['text'], tokenizer, None) # Disable RAG
        perplexity_sum += perplexity
        response_time_sum += (time.time() * 1000.0) - start_time

    avg_response_time = response_time_sum / len(dataset)
    avg_perplexity = perplexity_sum / len(dataset)
    print("avg_response_time: ", avg_response_time, "ms")
    print("avg_perplexity: ", avg_perplexity)




if __name__ == '__main__':
    # Initialize the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\r[. . . . . .] Loading tokenizer from openai-community/gpt2-medium ", end='')
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", use_fast=True, padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    print("\r[✔ . . . . .] Loading tokenizer from distilbert/distilroberta-base", end='')
    classTokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", use_fast=True)
    classTokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("\r[✔ ✔ . . . .] Loading classifier from checkpoint                  ", end='')
    classifier = AutoModelForSequenceClassification.from_pretrained("classifier", num_labels = 5).to(device)

    print("\r[✔ ✔ ✔ . . .] Loading experts from checkpoints                    ", end='')
    experts = {
        "computer science": AutoModelForCausalLM.from_pretrained("expert_CS").to(device),
        "electrical engineering and systems science": AutoModelForCausalLM.from_pretrained("expert_EE").to(device),
        "general physics": AutoModelForCausalLM.from_pretrained("expert_GP").to(device),
        "theoretical physics": AutoModelForCausalLM.from_pretrained("expert_TP").to(device),
        "general mathematics": AutoModelForCausalLM.from_pretrained("expert_GM").to(device)
    }
    class_list = [ 'general mathematics', 'computer science', 'general physics', 'electrical engineering and systems science', 'theoretical physics' ]

    print("\r[✔ ✔ ✔ ✔ . .] Loading data embeddings                             ", end='')
    with open('embeddings.pkl', 'rb') as f:
        data_dict = pickle.load(f)
        data_entries = list(data_dict.keys())
        data_embeddings = np.array(list(data_dict.values()))

    print("\r[✔ ✔ ✔ ✔ ✔ .] Loading sentence vectorizer                         ", end='')
    vectorizer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    print("\r[✔ ✔ ✔ ✔ ✔ ✔] Loaded all models                                   \n")

    # Start analysys
    dataset = merge_dataset(load_dataset("knowledgator/Scientific-text-classification", split='train'))
    get_perplexity(dataset)
