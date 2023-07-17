from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import transformers
import accelerate
import torch
from accelerate import load_checkpoint_and_dispatch
import datetime
import os
import traceback
from alphawave_pyexts import serverUtils as sv

# Load the model.
print('devices', torch.cuda.device_count(), torch.cuda.current_device())
model_name = "THUDM/chatglm2-6b"
print(f"Loading {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
model.tie_weights()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)


print(f"server started {model_name} into memory, use chatglm2 or maybe claude")

if __name__ == '__main__':
    #sv.server(tokenizer=tokenizer, pipeline=pipeline, stop_str=['Human:', 'Assistant:'])
    sv.server(model=model, tokenizer=tokenizer, stop_str=['###'])#, 'Human:'])#, 'Assistant:'])
