from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import accelerate
import torch
from accelerate import load_checkpoint_and_dispatch
import datetime
import os
from threading import Event, Thread
from uuid import uuid4
import requests, socket
from queue import Queue
import json
import sys
import time
import traceback
from collections.abc import Iterable

# Load the model.

print('devices', torch.cuda.device_count(), torch.cuda.current_device())
max_new_tokens = 1536
    
class MyStreamer(TextIteratorStreamer):
  def __init__(self, tokenizer, stop_event=None, skip_prompt=True):
    super(TextIteratorStreamer, self).__init__(tokenizer, True)
    self.text_queue = Queue()
    self.stop_signal = None
    self.timeout = None
    self.stop_event = stop_event
    
  def put(self, value):
    if len(value.shape) > 1 and value.shape[0] > 1:
      raise ValueError("TextStreamer only supports batch size 1")
    elif len(value.shape) > 1:
      value = value[0]
    idx = value.shape[0]
    kill = False
    value_list = value.tolist()
    super().put(value[:idx])
    if kill or self.stop_event.is_set():
      print("\n***** MyStreamer stopping stream kill {kill}, event {stop_event.is_set()}")
      super().end()
      time.sleep(0.5)
      if not self.stop_event.is_set():
        self.stop_event.set() # let main know we're done
        print("set stop in streamer put")
      raise SystemExit
    
    
def submit(message, model=None, tokenizer=None, pipeline=None, stop_event=None,conn=None, stop_str = None):
    stop_event.clear()
    streamer = MyStreamer(tokenizer, stop_event=stop_event)
    message_j = json.loads(message)
    temp = 0.7
    if 'temp' in message_j.keys():
        temp = message_j['temp']
    top_p = 1.0
    if 'top_p' in message_j.keys():
        top_p = message_j['top_p']
    max_tokens = 100
    if 'max_tokens' in message_j.keys():
        max_tokens = message_j['max_tokens']
    user_prompt = 'User'
    if 'user' in message_j.keys():
        user_prompt = message_j['user']
    asst_prompt = 'Assistant'
    if 'asst' in message_j.keys():
        asst_prompt = message_j['asst']
    eos = '<|endoftext|>'
    if 'eos' in message_j.keys():
        eos = message_j['eos']

    print(f'\n temp {temp}, top_p: {top_p}, max_tokens={max_tokens}\n')

    ### Pipeline
    if pipeline:
      result = pipeline(message_j['prompt'],
                        max_new_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temp,
                        #do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        return_full_text=False,
                        #early_stopping=True,
                        #attention_mask= attention_mask,
                        #eos_token_id=tokenizer.eos_token_id
                        )
      if result is not None and type(result[0]) == dict and 'generated_text' in result[0]:
        generated_text = result[0]['generated_text'] 
        #print(generated_text)
        conn.send(bytearray((generated_text).encode('utf8')))
        return generated_text
      else: return ''

    ### model and tokenizer
    else:
      encode_dict = tokenizer.encode_plus(message_j['prompt'], return_tensors="pt")# .to(DEV)
      input_ids = encode_dict['input_ids']
      attention_mask = encode_dict['attention_mask']
      input_ids = input_ids.to('cuda')
      generation_kwargs = dict(
        do_sample=True,
        max_new_tokens=max_tokens,
        top_p=top_p,
        temperature=temp,
        streamer = streamer,
        #do_sample=True,
        top_k=10,
        num_return_sequences=1,
        #early_stopping=True,
        attention_mask= attention_mask,
        eos_token_id=tokenizer.eos_token_id
      )
      
      generated_text = ""
      thread = Thread(target=model.generate, args=(input_ids,), kwargs=generation_kwargs)
      thread.start()
      for new_text in streamer:
        print(new_text)
        test_text = generated_text + new_text
        idx1 = test_text.find('<|endoftext|>')
        idx2 = test_text.find(user_prompt)
        idx3 = -1
        if stop_str is not None and isinstance(stop_str, Iterable):
          for stop in stop_str:
            idx3 = max(idx3, test_text.find(stop))
        if stop_event.is_set(): # flush generate
          continue
        if idx1>= 0 or idx2>=0 or idx3 >= 0:
          stop_event.set()
          idx = min(max(idx1, 0), max(idx2,0), max(idx3,0))
          if idx > 0:
            conn.send(bytearray((new_text[:idx]).encode('utf8')))
            print(f'setting stop event in submit {idx1}, {idx2}, {idx3}')
        else:
          generated_text = test_text
          conn.send(bytearray((new_text).encode('utf8')))  # send data to the client
      return generated_text

    
def server_program(model=None, tokenizer=None, pipeline=None, stop_str=None):
    # either model/tokenizer OR pipeline.
    # stop_str is a list of stop_strings
    # get the hostname
    host = socket.gethostname()
    host = ''
    port = 5004  # initiate port no above 1024
    stop_event = Event()
    print(f"starting server {stop_str}")
    while True:
      try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM)  as server_socket: # get instance
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))  # bind host address and port together
            server_socket.settimeout(3)

            while True:
                try:
                    conn = None
                    server_socket.listen(10)
                    #print("waiting...")
                    try:
                      conn, address = server_socket.accept()  # accept new connection
                    except Exception:
                      pass
                    if conn is None: continue
                    print("Connection from: " + str(address))
                    query_string = ''
                    while True:
                      s = conn.recv(1024)
                      if s is None or not s:
                        break
                      if (len(s) > 5 and s[-3:] == b'xff' and s[-6:-3] == b'x00'):
                        s = s[:-6]
                        query_string += s.decode('utf-8')
                        break
                      else:
                        query_string += s.decode('utf-8')
                    print("got string:", query_string)
                    if query_string is not None and len(query_string) > 0:
                        message_j = json.loads(query_string)
                        submit(query_string, model=model, tokenizer=tokenizer, pipeline=pipeline, stop_event=stop_event, conn=conn, stop_str=stop_str)
                        print('sending termination')
                        conn.sendall(b'x00xff')
                    if conn is not None: conn.close()
                except TimeoutError:
                  #traceback.print_exc()
                  if conn is not None: conn.close()
                except KeyboardInterrupt:
                  print("idle loop interrrupt")
                  traceback.print_exc()
                  sys.exit(0)
        print("resetting socket")
        server_socket.close()
      except BrokenPipeError:
        pass
      except KeyboardInterrupt:
        sys.exit(0)
      except Exception:
        traceback.print_exc()
