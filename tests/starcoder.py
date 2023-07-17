from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import accelerate
import torch
from accelerate import load_checkpoint_and_dispatch
import datetime
import os
import traceback
from alphawave_pyexts import serverUtils as sv

# Load the model.
model_name = "bigcode/starcoder"
print(f"Loading {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
model.tie_weights()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

print('devices', torch.cuda.device_count(), torch.cuda.current_device())

print(f"server started {model_name} into memory, ")

if __name__ == '__main__':
    sv.server(model, tokenizer)
