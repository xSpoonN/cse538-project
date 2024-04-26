import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading tokenizer from openai-community/gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", use_fast=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

print("Loading model from openai-community/gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium").to(device)

def chatbot(question, max_length=200, temperature=0.7):
    inputs = tokenizer.encode(question, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id, 
                             temperature=temperature, do_sample=True, top_k=50, top_p=0.95, 
                             num_return_sequences=5)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Output2: ", tokenizer.decode(outputs[1], skip_special_tokens=True))
    print("Output3: ", tokenizer.decode(outputs[2], skip_special_tokens=True))
    print("Output4: ", tokenizer.decode(outputs[3], skip_special_tokens=True))
    print("Output5: ", tokenizer.decode(outputs[4], skip_special_tokens=True))

    return response

# Example usage:
question = "A user's question: Explain Big-O Notation [END TURN]\nThe answer is:"
response = chatbot(question)
print(response)