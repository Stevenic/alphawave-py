from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
from alphawave_pyexts import serverUtils as sv
import inspect

"""
if __name__ == '__main__':
    model_name = "/home/bruce/Downloads/FastChat/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.tie_weights()
    print(f"Successfully loaded the model {model_name} into memory")
    print('**** ready to serve on port 5004')

    sv.server_program(model, tokenizer)
"""
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=True, spaces_between_special_tokens=False)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)
sv.server(model=None, tokenizer=tokenizer, pipeline = pipeline, stop_str=['User:'])
#sv.server(model=model, tokenizer=tokenizer, pipeline = None, stop_str=['User:'])
