import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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


model_name = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"

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
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    print('**** ready to serve on port 5004')
    #sv.server(model=model, tokenizer=tokenizer)
    sv.server(tokenizer=tokenizer, pipeline=pipeline, stop_str=['###', '### Input', '### Response'])
