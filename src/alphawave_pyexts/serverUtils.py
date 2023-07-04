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
import gc
import time
import string
import traceback
from collections.abc import Iterable

# Load the model.

print('devices', torch.cuda.device_count(), torch.cuda.current_device())
max_new_tokens = 1536

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, prompt, stop_strs=[]):
      self.stop_ids = []
      self.stop_strs = stop_strs
      self.tokenizer=tokenizer
      self.prompt_len=len(prompt)
      self.prompt=True
      
      if stop_strs is not None and isinstance(stop_strs, Iterable):
        for stop_wd in stop_strs:
          #print(f'***** stop wd pre encode {stop_wd}')
          if stop_wd is not None:
            tokens = tokenizer.encode(stop_wd)
            self.stop_ids.append(tokens)
      print(f'***** stop_id tokens {self.prompt_len}, {self.stop_ids} ')
      
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
      # don't process prompt
      if self.prompt:
        self.prompt = False
        return False
      
      for stop_id in self.stop_ids:
        if len(input_ids[0]) >= len(stop_id):
          if input_ids[0][:-len(stop_id)] == stop_id:
            print(f'***** stopping on token {stop_id}')
            return True
      input_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
      for stop_str in self.stop_strs:
        if len(input_str)>self.prompt_len and stop_str in input_str[self.prompt_len:]:
          print(f'***** stopping on string {stop_str}, \n{input_str[self.prompt_len:]}')
          return True
      return False
    
class MyStreamer(TextIteratorStreamer):
  def __init__(self, tokenizer, stop_event=None, skip_prompt=True):
    super(TextIteratorStreamer, self).__init__(tokenizer, skip_prompt)
    self.stop_event = stop_event
    ### why do these have to be here? should be inherited from TextIteratorStreamer?
    self.text_queue = Queue()
    self.stop_signal = None
    self.timeout = None

  def put(self, value):
    """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
    """

    if len(value.shape) > 1 and value.shape[0] > 1:
      raise ValueError("TextStreamer only supports batch size 1")
    elif len(value.shape) > 1:
      value = value[0]

    if self.skip_prompt and self.next_tokens_are_prompt:
      self.next_tokens_are_prompt = False
      return

    # Add the new token to the cache and decodes the entire thing.
    self.token_cache.extend(value.tolist())
    text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
    
    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
      printable_text = text[self.print_len :]
      self.token_cache = []
      self.print_len = 0
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
      printable_text = text[self.print_len :]
      self.print_len += len(printable_text)
      # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
      # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
      printable_text = text[self.print_len : text.rfind(" ") + 1]
      self.print_len += len(printable_text)
    self.on_finalized_text(printable_text)
    if self.stop_event.is_set():
        self.on_finalized_text(" ", stream_end=True)
        print(f"\n***** MyStreamer exit stop_event is {self.stop_event.is_set()}")
        time.sleep(0.1)
        raise SystemExit

  def end(self):
    """Flushes any remaining cache and prints a newline to stdout."""
    # Flush the cache, if it exists
    print('received end from generate')
    if len(self.token_cache) > 0:
      text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
      printable_text = text[self.print_len :]
      self.token_cache = []
      self.print_len = 0
    else:
      printable_text = ""
    self.next_tokens_are_prompt = True
    self.on_finalized_text(printable_text, stream_end=True)

  def on_finalized_text(self, text: str, stream_end: bool = False):
      """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
      filtered_text = "".join(filter(lambda x: x in string.printable, text))
      self.text_queue.put(filtered_text, timeout=self.timeout)
      if stream_end:
        self.text_queue.put(self.stop_signal, timeout=self.timeout)

  def __iter__(self):
    return self

  def __next__(self):
    value = self.text_queue.get(timeout=self.timeout)
    if value == self.stop_signal:
      raise StopIteration()
    else:
      return value
    
def submit(message, model=None, tokenizer=None, pipeline=None, stop_event=None,conn=None, stop_str = None):
    stop_event.clear()
    message_j = json.loads(message)
    streamer = MyStreamer(tokenizer, stop_event=stop_event)
    stopping_criteria=StoppingCriteriaList([StopOnTokens(tokenizer, message_j['prompt'], stop_str )])
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
                        stopping_criteria = stopping_criteria,
                        #eos_token_id=tokenizer.eos_token_id
                        )
      if result is not None and type(result[0]) == dict and 'generated_text' in result[0]:
        generated_text = result[0]['generated_text'] 
        #print(generated_text)
        idx=100000; idx3 = -1; 
        if stop_str is not None and isinstance(stop_str, Iterable):
          for stop in stop_str:
            idx3 = generated_text.find(stop)
            if idx3 >0 and idx3 < idx : idx = idx3
        if idx > 0 and idx < 100000:
          print(f'Truncating at {idx}')
          generated_text = generated_text[:idx]
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
        top_k=20,
        num_return_sequences=1,

        attention_mask= attention_mask,
        eos_token_id=tokenizer.eos_token_id
      )
      
      generated_text = ""
      #thread = Thread(target=model.generate, args=(input_ids,), kwargs=generation_kwargs)
      thread = Thread(target=lambda: model.generate(input_ids,
                                                    do_sample=True,
                                                    max_new_tokens=max_tokens,
                                                    min_new_tokens=20,
                                                    top_p=top_p,
                                                    stopping_criteria = stopping_criteria,
                                                    temperature=temp,
                                                    streamer=streamer,
                                                    num_return_sequences=1,
                                                    attention_mask=attention_mask))
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
          # get the earliest stop found
          idx = 1000000
          for candidate_idx in [idx1, idx2, idx3]:
              if candidate_idx >= 0 and candidate_idx < idx:
                  idx = candidate_idx
          print(f'set stop event in submit {idx1}, {idx2}, {idx3}, stop on {idx}')
          if idx < len(test_text) and idx > len(generated_text):  #see if there is anything to send
            conn.send(bytearray((test_text[len(generated_text):idx]).encode('utf8')))
        else:
          generated_text = test_text
          conn.send(bytearray((new_text).encode('utf8')))  # send data to the client
      return generated_text

    
def server(model=None, tokenizer=None, pipeline=None, stop_str=[]):
    global stopping_criteria
    # either model/tokenizer OR pipeline.
    # stop_str is a list of stop_strings
    # get the hostname
    host = socket.gethostname()
    host = ''
    port = 5004  # initiate port no above 1024
    stop_event = Event()
    print(f"starting server")
    

    while True:
      try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM)  as server_socket: # get instance
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))  # bind host address and port together
            server_socket.settimeout(3)

            while True:
                try:
                    conn = None
                    server_socket.listen(30)
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
                        gc.collect()
                        torch.cuda.empty_cache()
                        print('sending termination')
                        conn.sendall(b'x00xff')
                        time.sleep(0.5)
                    if conn is not None: conn.close()
                except TimeoutError:
                  #traceback.print_exc()
                    if conn is not None:
                        conn.sendall(b'x00xff')
                        time.sleep(0.5)
                        conn.close()
                except KeyboardInterrupt:
                  print("idle loop interrrupt")
                  traceback.print_exc()
                  sys.exit(0)
        print("resetting socket")
        server_socket.close()
      except BrokenPipeError:
        traceback.print_exc()
        pass
      except KeyboardInterrupt:
        sys.exit(0)
      except Exception:
        traceback.print_exc()

