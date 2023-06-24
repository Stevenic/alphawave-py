from transformers import AutoModelForCausalLM, AutoTokenizer
from alphawave_pyexts import serverUtils as sv
import torch

if __name__ == '__main__':
    model_name = "/home/bruce/Downloads/FastChat/Falcon-40B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    model.tie_weights()
    print(f"Successfully loaded the model {model_name} into memory")
    print('**** ready to serve on port 5004')

    sv.server_program(model=model, tokenizer=tokenizer)
