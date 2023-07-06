import torch
import transformers
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


model_name = "TheBloke/wizardLM-7B-HF"

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, skip_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_8bit=True,
        device_map="auto",
    )
    model.tie_weights()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    print('**** ready to serve on port 5004')
    #sv.server(model=model, tokenizer=tokenizer, stop_str=['###', '### Input', '### Response'])
    sv.server(tokenizer=tokenizer, pipeline=pipeline, stop_str=['###', '### Input', '### Response'])
