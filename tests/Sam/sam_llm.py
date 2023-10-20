import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Any, Dict

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and cache
models_dir = "/home/bruce/Downloads/models/"

models = ["Mistral-7B-OpenOrca-exl2",
          "mistral-airoboros-7b-GPTQ",
          "Nous-Hermes-70b-GPTQ",
          "platypus2-70b-instruct-gptq",
          "Spicyboros-70b-22-GPTQ",
          ]

model_number = -1
while model_number < 0 or model_number > len(models) -1:
    print(f'Available models:')
    for i in range(len(models)):
        print(f'{i}. {models[i]}')
    number = input('input model # to load: ')
    try:
        model_number = int(number)
    except:
        print(f'Enter a number between 0 and {len(models)-1}')


config = ExLlamaV2Config()
config.model_dir = models_dir+models[model_number]
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + models[model_number])
model.load([19, 24])

tokenizer = ExLlamaV2Tokenizer(config)

cache = ExLlamaV2Cache(model)

# Initialize generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

# Settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.1
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.15
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
max_new_tokens = 250

# Make sure CUDA is initialized so we can measure performance
generator.warmup()


host = socket.gethostname()
host = ''
port = 5004  # initiate port no above 1024

app = FastAPI()
print(f"starting server")

async def stream_data(query: Dict[Any, Any], max_new_tokens):
    generated_tokens = 0
    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        print (chunk, end = "")
        yield chunk
        if eos or generated_tokens == max_new_tokens:
            print('\n')
            break

@app.post("/")
async def get_stream(request: Request):
    global generator, settings, tokenizer
    query = await request.json()
    print(f'request: {query}')
    message_j = query

    temp = 0.1
    if 'temp' in message_j.keys():
        temp = message_j['temp']
    settings.temperature = temp

    top_p = 1.0
    if 'top_p' in message_j.keys():
        top_p = message_j['top_p']
    settings.top_p = top_p

    max_tokens = 100
    if 'max_tokens' in message_j.keys():
        max_tokens = message_j['max_tokens']
    stop_conditions = ['###','<|endoftext|>']
    if 'eos' in message_j.keys():
        stop_conditions = message_j['eos']

    prompt = message_j['prompt']
    input_ids = tokenizer.encode(prompt)
    generator.set_stop_conditions(stop_conditions)
    generator.begin_stream(input_ids, settings)
    
    return StreamingResponse(stream_data(query, max_new_tokens = max_tokens))

