import json
import sys
import os
import time
import traceback
import readline as rl
from alphawave.MemoryFork import MemoryFork
import alphawave_pyexts.llmsearch.google_search_concurrent as gs
import alphawave_pyexts.utilityV2 as ut
import asyncio
history = {}


async def run_chat(client, query_string, model,  memory, functions, tokenizer, max_chars=1500, search_level=gs.QUICK_SEARCH):
  #tracemalloc.start()
  response_text = ''
  storeInteraction = True
  try:
    #
    fork = MemoryFork(memory)
    query_phrase, keywords = await ut.get_search_phrase_and_keywords(client, query_string, model, fork, functions, tokenizer)

    google_text=\
      await gs.search_google(query_string, gs.QUICK_SEARCH, query_phrase, keywords, client, model, fork, functions, tokenizer, max_chars)
    return google_text
  except:
    traceback.print_exc()
  return ''

if __name__ == '__main__' :
  while True:
    asyncio.run(run_chat(input('Yes?')))
