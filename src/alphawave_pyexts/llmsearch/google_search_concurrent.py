import concurrent.futures
import json
import sys
import os
import time
import string
from datetime import date
from datetime import datetime
import random
import traceback
import re
import requests
import copy
from itertools import zip_longest
#import urllib3
import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import wordfreq as wf
from unstructured.partition.html import partition_html
import nltk
import urllib.parse as en
import asyncio
import warnings
from promptrix.Prompt import Prompt
from alphawave_agents.PromptCommand import PromptCommand
from promptrix.UserMessage import UserMessage
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave_agents.PromptCommand import CommandSchema, PromptCommandOptions
from alphawave.AlphaWave import AlphaWave
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.MemoryFork import MemoryFork
import alphawave_pyexts.utilityV2 as ut

today = " as of "+date.today().strftime("%b-%d-%Y")+'\n\n'

suffix = "\nA: "
client = '\nQ: '
QUICK_SEARCH = 'quick'
NORMAL_SEARCH = 'moderate'
DEEP_SEARCH = 'deep'

system_prime = {"role": "system", "content":"You analyze Text with respect to Query and list any relevant information found, including direct quotes from the text, and detailed samples or examples in the text."}
priming_1 = {"role": "user", "content":"Query:\n"}
priming_2 = {"role": "user", "content":"List relevant information in the provided text, including direct quotes from the text. If none, respond 'no information'.\nText:\n"}


# Define a function to make a single URL request and process the response
def process_url(query_phrase, keywords, keyword_weights, url, timeout, client, model, memory, functions, tokenizer, max_chars):
    start_time = time.time()
    site = ut.extract_site(url)
    result = ''
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            options = Options()
            options.page_load_strategy = 'eager'
            options.add_argument("--headless")
            result = ''
            with webdriver.Chrome(options=options) as dr:
                #print(f'*****setting page load timeout {timeout} {url}')
                dr.set_page_load_timeout(timeout)
                try:
                    dr.get(url)
                    response = dr.page_source
                    result = response_text_extract(query_phrase, keywords, keyword_weights, url, response, int(time.time()-start_time),
                                                          client, model, memory, functions, tokenizer, max_chars)
                except selenium.common.exceptions.TimeoutException:
                    return '', url
    except Exception:
        traceback.print_exc()
        print(f"{site} err")
        pass
    #print(f"Processed {site}: {len(response)} / {len(result)} {int((time.time()-start_time)*1000)} ms")
    return result, url

async def process_urls(query_phrase, keywords, keyword_weights, urls, search_level, client, model, memory, functions, tokenizer, max_chars):
    response = []
    start_time = time.time()
    full_text = ''
    in_process = []
    max_w = 8
    # Create a ThreadPoolExecutor with 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
        # initialize scan of google urls
        try:
            while len(urls) > 0 or len(in_process) >0:
                # empty queue if out of time
                if (search_level==NORMAL_SEARCH and (len(full_text) > 6400 or time.time()-start_time > 24)
                    or (search_level==QUICK_SEARCH  and (len(full_text) > 4800 or time.time()-start_time > 18))):
                    urls = []
                elif len(urls) > 0:
                    url = urls[0]
                    urls = urls[1:]
                    # set timeout so we don't wait for a slow site forever
                    timeout = 24-int(time.time()-start_time)
                    if search_level==NORMAL_SEARCH:
                        timeout = timeout+8
                    future = executor.submit(process_url, query_phrase, keywords, keyword_weights, url, timeout, client, model, memory, functions, tokenizer, max_chars)
                    in_process.append(future)
                    #print(f'added one to in_process {len(urls)}, {len(in_process)}')
                # Process the responses as they arrive
                for future in in_process:
                    if future.done():
                        in_process.remove(future)
                        #print(f'removed one from in_process {len(urls)}, {len(in_process)}')
                        try:
                            result, url = future.result()
                        except asyncio.TimeoutError:
                            continue
                        if 'coroutine' in str(type(result)).lower():
                            result = await result
                        if len(result) > 0:
                            site = ut.extract_site(url)
                            domain = ut.extract_domain(url)
                            response.append({'source':ut.extract_domain(url), 'url':url, 'text':result})

                if len(in_process) < max_w:
                    time.sleep(.05)
                else:
                    time.sleep(.5)
                        
            executor.shutdown(wait=True)
            #print(f'process_urls returning')
            return response
        except:
            traceback.print_exc()
            executor.shutdown(wait=False)
        return response

def extract_subtext(text, query_phrase, keywords, keyword_weights):
    ###  maybe we should score based on paragraphs, not lines?
    sentences = ut.reform(text)
    sentence_weights = {}
    final_text = ''
    for sentence in sentences:
        sentence_weights[sentence] = 0
        for keyword in keywords:
            if keyword in sentence or keyword.lower() in sentence:
                if keyword in keyword_weights.keys():
                    sentence_weights[sentence]+=keyword_weights[keyword]
                    
    # now pick out sentences starting with those with the most keywords
    max_sentence_weight = 0
    for keyword in keyword_weights.keys():
        max_sentence_weight += keyword_weights[keyword]
    for i in range(max_sentence_weight,1,-1):
        if len(final_text)>3000 and i < max(1, int(max_sentence_weight/4)): # make sure we don't miss any super-important text
            return final_text
        for sentence in sentences:
            if len(final_text)+len(sentence)>3001 and i < max(1, int(max_sentence_weight/4)):
                continue
            if sentence_weights[sentence] == i:
                final_text += sentence

    return final_text


def search(query_phrase):
    sort = '&sort=date-sdate:d:w'
    if 'today' in query_phrase or 'latest' in query_phrase:
        sort = '&sort=date-sdate:d:s'
    try:
        google_query=en.quote(query_phrase)
    except:
        print(f'googleSearch pblm w query_phrase, ignoring {query_phrase}')
        return []
    response=[]
    try:
        start_wall_time = time.time()
        url="https://www.googleapis.com/customsearch/v1?key="+ut.google_key+'&cx='+ut.google_cx+'&num=10'+sort+'&q='+google_query
        response = requests.get(url)
        response_json = json.loads(response.text)
    except:
      traceback.print_exc()
      return []

    #see if we got anything useful from google
    if 'items' not in response_json.keys():
        return []

    # compile list of urls
    urls = []
    for i in range(len(response_json['items'])):
        url = response_json['items'][i]['link'].lstrip().rstrip()
        urls.append(url)
    return urls

def log_url_process(site, reason, raw_text, extract_text, gpt_text):
    return


async def llm_tldr (text, query, client, model, memory, functions, tokenizer, max_chars):
    text = text[:max_chars] # make sure we don't run over token limit
    prompt = Prompt([UserMessage('Analyze the following Text to identify if there is any content relevant to the query {{$query}}, Respond using this JSON template:\n\n{"relevant": "Yes" if there is any text found in the input that is relevant to the query, "No" otherwise>, "tldr": "<relevant content found in Text, rewritten coherence>"}\n\nText:\n{{$input}}\n. ')])
    response_text=''
    completion = None
    schema = {
        "schema_type":"object",
        "title":"tldr",
        "description":"extract query-relevant text",
        "properties":{
            "relevant": {
                "type": "string",
                "description": "Yes if any relevant text, otherwise No"
            },
            "tldr": {
                "type": "string",
                "description": "relevant extract from Text, or empty string"
            }
        },
        "required":["relevant", "tldr"],
        "returns":"extract"
    }
    
    options = PromptCompletionOptions(completion_type='chat', model=model)
    # don't clutter primary memory with tldr stuff
    fork = MemoryFork(memory)
    response = await ut.run_wave(client, {'input':text, 'query':query}, prompt, options, fork, functions, tokenizer, validator=JSONResponseValidator(schema))
    #print(f'\n***** google llm_tldr processing \n{response}')
    if type(response) == dict and 'status' in response and response['status'] == 'success'and 'message' in response:
        message = response['message']
        if type(message) != dict:
            try:
                message = json.loads(message)
            except Exception as e:
                if message.find('tldr')> 0:
                    return message[message.find('tldr'):]
                else:
                    return ''
        if 'content' in message:
            content = message['content']
            if type(content) !=  dict:
                try:
                    content = json.loads(content.strip())
                except Exception as e:
                    return ''
            if 'relevant' in content and 'yes' in str(content['relevant']).lower() and 'tldr' in content:
                #print(f"***** google llm_tldr\n{content['tldr']}\n")
                return content['tldr']
    return ''
    

def response_text_extract(query_phrase, keywords, keyword_weights, url, response, get_time, client, model, memory, functions, tokenizer, max_chars):
    curr=time.time()
    extract_text = ''
    site = ut.extract_site(url)
    if url.endswith('pdf'):
        pass
    else:
        elements = partition_html(text=response)
        str_elements = []
        for e in elements:
            stre = str(e).replace('  ', ' ')
            str_elements.append(stre)
        extract_text = extract_subtext(str_elements, query_phrase, keywords, keyword_weights)
    if len(extract_text.strip()) < 8:
        return ''

    # now ask openai to extract answer
    response = ''
    response = llm_tldr(extract_text, query_phrase, client, model, memory, functions, tokenizer, max_chars)
    return response

def extract_items_from_numbered_list(text):
    items = ''
    elements = text.split('\n')
    for candidate in elements:
        candidate = candidate.lstrip('. \t')
        if len(candidate) > 4 and candidate[0].isdigit():
            candidate = candidate[1:].lstrip('. ')
            if len(candidate) > 4 and candidate[0].isdigit(): # strip second digit if more than 10 items
                candidate = candidate[1:].lstrip('. ')
            items += candidate+' '
    return items

async def search_google(original_query, search_level, query_phrase, keywords, client, model, memory, functions, tokenizer, max_chars):
  start_time = time.time()
  all_urls=[]; urls_used=[]; urls_tried=[]
  index = 0; tried_index = 0
  full_text=''
  keyword_weights = {}
  for keyword in keywords:
      zipf = wf.zipf_frequency(keyword, 'en')
      weight = max(0, int((8-zipf)))
      if weight > 0:
          keyword_weights[keyword] = weight
          subwds = keyword.split(' ')
          if len(subwds) > 1:
              for subwd in subwds:
                  sub_z = wf.zipf_frequency(subwd, 'en')
                  sub_wgt = max(0, int((8-zipf)*1/2))
                  if sub_wgt > 0:
                      keyword_weights[subwd] = sub_wgt

                  
  try:  # query google for recent info
    sort = ''
    if 'today' in original_query or 'latest' in original_query:
        original_query = today.strip('\n')+' '+original_query
    extract_query = ''
    orig_phrase_urls = []
    if len(original_query) > 0:
        orig_phrase_urls = search(original_query[:min(len(original_query), 128)])
        extract_query = original_query[:min(len(original_query), 128)]
    gpt_phrase_urls = []
    if len(query_phrase) > 0:
        gpt_phrase_urls = search(query_phrase)
        extract_query = query_phrase # prefer more succint query phrase if available
    if len(orig_phrase_urls) == 0 and len(gpt_phrase_urls) == 0:
        return ''

    for url in orig_phrase_urls:
        if url in gpt_phrase_urls:
            gpt_phrase_urls.remove(url)

    # interleave both lists now that duplicates are removed
    urls = [val for tup in zip_longest(orig_phrase_urls, gpt_phrase_urls) for val in tup if val is not None]
    all_urls = copy.deepcopy(urls)
    full_text = \
        await process_urls(extract_query, keywords, keyword_weights, all_urls, search_level, client, model, memory, functions, tokenizer, max_chars)
  except:
      traceback.print_exc()
  return  full_text

