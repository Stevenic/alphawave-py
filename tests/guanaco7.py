from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, TextIteratorStreamer
import transformers
import accelerate
import torch
from accelerate import load_checkpoint_and_dispatch
import datetime
import os
import traceback
from alphawave_pyexts import serverUtils as sv

# Load the model.
# You can also use the 13B model or 7gb
model_name = "JosephusCheung/Guanaco"

print('devices', torch.cuda.device_count(), torch.cuda.current_device())
print(f"Starting to load the model {model_name} into memory")

tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=True, spaces_between_special_tokens=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
model.tie_weights()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)
#sv.server(model=model, tokenizer=tokenizer, pipeline = None, stop_str=['H:', 'Human:', 'User:'])
sv.server(model=None, tokenizer=tokenizer, pipeline = pipeline, stop_str=['H:', 'Human:', 'User:'])

print(f"Successfully loaded the model {model_name} into memory")
