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
# You can also use the 13B model or 7gb
model_name = "TheBloke/guanaco-13B-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
model.tie_weights()

print('devices', torch.cuda.device_count(), torch.cuda.current_device())
print(f"Starting to load the model {model_name} into memory")

DEV=torch.cuda.current_device()

print(f"Successfully loaded the model {model_name} into memory")

if __name__ == '__main__':
    sv.server_program(model, tokenizer)
