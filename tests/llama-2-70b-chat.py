import torch
import transformers
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from alphawave_pyexts import serverUtils as sv


model_name = '/home/bruce/Downloads/llama/llama-2-70b-chat'
print(f"Loading {model_name}")

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Hello my name is"
device = "cuda:0"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)
print(f"server started {model_name} into memory, use alpaca")

if __name__ == '__main__':
    prompt = ''
    question = "What is a cool fact about ducks?"
    model_inputs = tokenizer(prompt + question, return_tensors="pt").to(model.device)
    print(model_inputs.input_ids.shape)  # 6782 tokens, needs a GPU with >24GB 
    
    # Let's use it!
    generate_kwargs = {"max_new_tokens": 40, "do_sample": False}
    gen_out = model.generate(**model_inputs, **generate_kwargs)
    print(tokenizer.decode(gen_out[0], skip_special_tokens=True))

    sv.server(model=None, tokenizer=tokenizer, pipeline = pipeline, stop_str=['User:'])
    #sv.server(model=model, tokenizer=tokenizer, pipeline = None, stop_str=['User:'])
