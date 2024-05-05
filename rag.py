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