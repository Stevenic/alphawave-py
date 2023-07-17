import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


model_name = "lmsys/vicuna-7b-v1.3"
print(f"Loading {model_name}")

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        #trust_remote_code=True
    )
    model.tie_weights()
    print(f'server started {model_name} on port 5004, use vicuna_v1.1')
    sv.server(model=model, tokenizer=tokenizer)
