# File to generate vector embeddings for each entry in the dataset
# Team members:
# - Kevin Tao
# - Haneul Lee

# Description:
# multi-qa-MiniLM-L6-cos-v1 model is used to generate embeddings for each entry in the dataset.
# The embeddings are then normalized and saved to a file for later use in the chatbot.
# This file also contains functions used by the chatbot to vectorize text and find the k closest text entries.

# System specifications during generation:
# Windows 11 Home 23H2 OS Build 22631.3447
# AMD Ryzen 7 5700 @ 3.7 GHz, 32 GB RAM DDR4 @ 3200 MHz CL16
# NVIDIA GeForce RTX 4060 Ti 8 GB GDDR6 DX12

from merge_dataset import merge_dataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

def vectorize_text(text, model):
    return model.encode(text, convert_to_tensor=True)

def get_closest_text(text, embeddings, entries, model, k=3):
    query_embedding = vectorize_text(text, model)
    query_embedding = query_embedding.cpu().numpy() 
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=0, keepdims=True)
    cos_scores = np.dot(embeddings, query_embedding)
    top_results_indices = np.argsort(cos_scores, axis=0)[-k:][::-1]
    top_results = [entries[i] for i in top_results_indices]
    return [text for text in top_results]
    
if __name__ == '__main__':
    print("Loading dataset knowledgator/Scientific-text-classification")
    dataset = merge_dataset(load_dataset("knowledgator/Scientific-text-classification", split='train'))

    print("Generating embeddings")
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    embeddings = vectorize_text(dataset['text'], model)

    print("Normalizing embeddings")
    embeddings = embeddings.cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print("Saving embeddings")
    with open('embeddings.pkl', 'wb') as f:
        data_dict = {entry: embeddings for entry, embeddings in zip(dataset['text'], embeddings)}
        pickle.dump(data_dict, f)