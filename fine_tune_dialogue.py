import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def fine_tune_dialogue(dataset, tokenizer, model, device, epochs=1, batch_size=4, learning_rate=1e-5):
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
            inputs = [f"Question: {q} Answer: {a}" for q, a in zip(batch['question'], batch['correct_answer'])]
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
if __name__ == '__main__':
    sciq = load_dataset("allenai/sciq", split='train')

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fine_tune_dialogue(sciq, tokenizer, model, device)
    model.save_pretrained("fine_tuned_dialogue")