# This file trains the mastermind model on all entries in the dataset.
# Team members:
# - Kevin Tao
# - Haneul Lee

# Description:
# All entries in the dataset are used to train the mastermind model, from a GPT2 base model.
# The model is trained as a causal language model in a self-supervised manner.

# System specifications during training:
# Windows 11 Home 23H2 OS Build 22631.3447
# AMD Ryzen 7 5700 @ 3.7 GHz, 32 GB RAM DDR4 @ 3200 MHz CL16
# NVIDIA GeForce RTX 4060 Ti 8 GB GDDR6 DX12

import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from merge_dataset import merge_dataset

def fine_tune_mastermind(dataset, tokenizer, model, device, epochs=1, batch_size=4, learning_rate=1e-6):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    model.to(device)
    torch.cuda.empty_cache()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)["input_ids"]
            with torch.cuda.amp.autocast():
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 1 == 0:
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / (i+1)
                remaining_time = avg_time * (len(train_loader) - i)
                print(f"\rEpoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {train_loss/(i+1):.4f}, ETA: {remaining_time:.2f}s", end="")
    print(f"\rTrained model in {time.time() - start_time:.2f}s over {epochs} epoch{'s' if epochs != 1 else ''}. Final loss: {train_loss/len(train_loader):.4f}              ")
    return model

if __name__ == '__main__':
    print("Loading dataset knowledgator/Scientific-text-classification")
    dataset = merge_dataset(load_dataset("knowledgator/Scientific-text-classification", split='train'))

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2") # Load the model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mastermind = fine_tune_mastermind(dataset, tokenizer, model, device)
    mastermind.save_pretrained("mastermind")