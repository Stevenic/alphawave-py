import openai
import requests
import time
import traceback
import json
import os
from typing import Any, Dict, List, Tuple
import alphawave.LLMClient as client

openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_35 = 'gpt-3.5-turbo'
GPT_4 = 'gpt-4'
FC_T5 = 'fs-t5-3b-v1.0'
VICUNA_7B = 'vicuna-7b'
VICUNA_13B = 'vicuna-13b'
WIZARD_13B = 'wizard-13b'
WIZARD_30B = 'wizard-30b'
GUANACO_33B = 'guanaco-33b'
MODEL = WIZARD_30B
#MODEL = GPT_35
#MODEL = GPT_4
#MODEL = GUANACO_33B

if not MODEL.startswith('gpt') and MODEL != WIZARD_30B and MODEL != GUANACO_33B:
  # this for connecting to fastchat server
  client.initialize(MODEL)

def ask_LLM(model, gpt_message, max_tokens=100, temp=0.7, top_p=1.0, tkroot = None, tkdisplay=None):
    completion = None
    response = ''
    try:
      if model == FC_T5 or model == VICUNA_7B or model == VICUNA_13B or model == WIZARD_13B:
        print(f'calling {model} via fastchat')
        completion = client.send_w_stream(messages=gpt_message, max_tokens = max_tokens, temperature=temp )
        if completion is None:
          print('None from '+model)
        else:
          #print(f'completion from t5: {completion.choices[0].message.content}')
          response = completion

      elif model == WIZARD_30B or model == GUANACO_33B:
        print('calling {model} via local server')
        completion = client.run_query(gpt_message, temp, top_p, max_tokens, tkroot, tkdisplay)
        if completion is None:
          print('None from '+model)
        else:
          #print(f'completion from t5: {completion.choices[0].message.content}')
          if tkdisplay is not None:
            tkdisplay.insert(tk.END, completion)
            if tkroot is not None:
              tkroot.update()
          response = completion

      else:
        print('calling chatGPT', gpt_message)
        stream= openai.ChatCompletion.create(
            model=model, messages=gpt_message, max_tokens=max_tokens, temperature=temp, top_p=1, stop='STOP', stream=True)
        response = ''
        if stream is None:
          print('None from GPT')
          return response
        for chunk in stream:
          item = chunk['choices'][0]['delta']
          if 'content' in item.keys():
            #print(f"completion from gpt: {chunk['choices'][0]['delta']['content']}")
            if tkdisplay is not None:
              tkdisplay.insert(tk.END, chunk['choices'][0]['delta']['content'])
              if tkroot is not None:
                tkroot.update()
              response += chunk['choices'][0]['delta']['content']
    except:
        traceback.print_exc()
        print(response)
    return response

if __name__ == '__main__':  
  print(ask_LLM(ut.MODEL, gpt_message, max_tokens=100, temp=.1, top_p=0.1))
