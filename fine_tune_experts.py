import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from merge_dataset import merge_dataset

def fine_tune_expert(dataset, tokenizer, model, device, epochs=1, batch_size=4, learning_rate=1e-6):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    model.to(device)
    torch.cuda.empty_cache()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = [f"{text}" for text in zip(batch['text'])]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)["input_ids"]
            with torch.cuda.amp.autocast():
                outputs = model(model_inputs, labels=model_inputs)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {train_loss/(i+1):.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}")

    return model

if __name__ == '__main__':
    print("Loading dataset knowledgator/Scientific-text-classification")
    dataset = merge_dataset(load_dataset("knowledgator/Scientific-text-classification", split='train'))
    dataset_CS = dataset.filter(lambda x: x['label'] == 'computer science')
    dataset_Math = dataset.filter(lambda x: x['label'] == 'general mathematics')
    dataset_TPhysics = dataset.filter(lambda x: x['label'] == 'theoretical physics')
    dataset_Physics = dataset.filter(lambda x: x['label'] == 'general physics')
    dataset_EE = dataset.filter(lambda x: x['label'] == 'electrical engineering and systems science')

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    model_CS = AutoModelForCausalLM.from_pretrained("fine_tuned_dialogue") # Load the model checkpoint
    model_Math = AutoModelForCausalLM.from_pretrained("fine_tuned_dialogue")
    model_TPhysics = AutoModelForCausalLM.from_pretrained("fine_tuned_dialogue")
    model_Physics = AutoModelForCausalLM.from_pretrained("fine_tuned_dialogue")
    model_EE = AutoModelForCausalLM.from_pretrained("fine_tuned_dialogue")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset, model, name in zip([dataset_CS, dataset_Math, dataset_TPhysics, dataset_Physics, dataset_EE], 
                              [model_CS, model_Math, model_TPhysics, model_Physics, model_EE],
                              ["CS", "GM", "TP", "GP", "EE"]):
        model = fine_tune_expert(dataset, tokenizer, model, device)
        model.save_pretrained(f"expert_{name}")