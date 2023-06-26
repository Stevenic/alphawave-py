from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
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
model_name = 'huggyllama/llama-30b'
adapters_name = "timdettmers/guanaco-33b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)
model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


model.tie_weights()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)

print('devices', torch.cuda.device_count(), torch.cuda.current_device())
print(f"Starting to load the model {model_name} into memory")


print(f"Successfully loaded the model {model_name} into memory")

if __name__ == '__main__':
    sv.server(tokenizer=tokenizer, pipeline=pipeline, stop_str=['Human:', 'Assistant:'])
