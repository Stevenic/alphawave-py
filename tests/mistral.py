import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes
from alphawave_pyexts import serverUtils as sv
from accelerate import load_checkpoint_and_dispatch
import datetime
import os
import requests, socket
from queue import Queue
import json
import sys
import time
import traceback


#model_name = "/home/bruce/Downloads/models/mistral-7b-instruct"
model_name = "/home/bruce/Downloads/models/mistral-orca"
print(f'**** Loading {model_name}')

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, skip_special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
    use_safetensors=True,
    device_map="auto",
)
model.tie_weights()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

print(f'server started {model_name} on port 5004, use openorca')
sv.server(model=model, tokenizer=tokenizer, stop_str=['<|im_end|>'])
#sv.server(tokenizer=tokenizer, pipeline=pipeline, stop_str=['### Instruction', '### User'])
