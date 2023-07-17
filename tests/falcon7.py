from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
from alphawave_pyexts import serverUtils as sv
import inspect


model_name = "tiiuae/falcon-7b-instruct"
print(f"Loading {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=True, spaces_between_special_tokens=False)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)
print(f"server started {model_name} into memory, use falcon_instruct")
sv.server(model=None, tokenizer=tokenizer, pipeline = pipeline, stop_str=['User:'])
#sv.server(model=model, tokenizer=tokenizer, pipeline = None, stop_str=['User:'])
