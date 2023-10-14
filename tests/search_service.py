import json
import sys
import os
import time
import traceback
import readline as rl
from fastapi import FastAPI
from alphawave.MemoryFork import MemoryFork
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
import alphawave_pyexts.llmsearch.google_search_concurrent as gs
import alphawave_pyexts.utilityV2 as ut
history = {}


#client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)
client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)
model = 'gpt-3.5-turbo'
memory = VolatileMemory()
functions = None
tokenizer = GPT3Tokenizer()

app = FastAPI()


@app.get("/search/")
async def search(query: str, max_chars: int = 1000):
  global client, memory, functions, tokenizer
  response_text = ''
  storeInteraction = True
  try:
    query_phrase, keywords = ut.get_search_phrase_and_keywords(client, query, model, memory, functions, tokenizer)
    google_text= \
      gs.search_google(query, gs.QUICK_SEARCH, query_phrase, keywords, client, model, memory, functions, tokenizer, max_chars)
    return {"result":google_text}
  except Exception as e:
    traceback.print_exc()
    return {"result":str(e)}

