import argparse
import json
import socket
import os
import traceback
import tkinter as tk
import tkinter.font
from tkinter import ttk
import ctypes
import asyncio
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as llm

temperature=0.1
top_p = 1
max_tokens=100
FORMAT='FORMAT'
prompt_text = ''
PROMPT = Prompt([
    SystemMessage(prompt_text),
    ConversationHistory('history', .5),
    UserMessage('{{$input}}')
])

#print(f' models: {llm.get_available_models()}')
parser = argparse.ArgumentParser()
#parser.add_argument('model', type=str, default='wizardLM', choices=['guanaco', 'wizardLM', 'zero_shot', 'vicuna_v1.1', 'dolly', 'oasst_pythia', 'stablelm', 'baize', 'rwkv', 'openbuddy', 'phoenix', 'claude', 'mpt', 'bard', 'billa', 'h2ogpt', 'snoozy', 'manticore', 'falcon_instruct', 'gpt_35', 'gpt_4'],help='select prompting based on modelto load')

model = ''
modelin = input('model name? ').strip()
if modelin is not None and len(modelin)>1:
    model = modelin.strip()
    models = llm.get_available_models()
    while model not in models:
        print(models)
        modelin = input('model name? ').strip()
        model=modelin

def set(text):
    global temperature, top_p, max_tokens
    try:
        if 'temp' in text:
            idx = text.find('=')
            if idx > 0:
                temperature = float(text[idx+1:].strip())
        if 'top_p' in text:
            idx = text.find('=')
            if idx > 0:
                top_p = float(text[idx+1:].strip())
        if 'max_tokens' in text:
            idx = text.find('=')
            if idx > 0:
                max_tokens = int(text[idx+1:].strip())
    except Exception as e:
        print ("parse failed", e)

def setFormat():
    global FORMAT
    if FORMAT:
        format_button.config(text='RAW')
        FORMAT=False
    else:
        format_button.config(text='FORMAT')
        FORMAT=True


def setPrompt(input_text):
    global PROMPT
    input_text = input_text.strip()
    if input_text.startswith("N"):
        input_text = ''
    elif input_text.startswith("H"):
        input_text = 'Respond as a knowledgable and friendly AI, speaking to an articulate, educated, conversant. Limit your response to 100 words where possible. Say "I don\'t know" when you don\'t know."'
        print(f'prompt set to: {input_text}')
    elif input_text.startswith("B"):
        input_text = 'Respond as a compassionate, empathetic, self-realized follower of Ramana Maharshi.Limit your response to 100 words where possible.'
        print(f'prompt set to: {input_text}')
    elif input_text.startswith("A"):
        input_text = 'Respond as a compassionate, empathetic, counselor familiar with Acceptance Commitment Therapy. Limit your response to 100 words where possible.'
        print(f'prompt set to: {input_text}')
    elif input_text.startswith("F"):
        input_text = 'Respond as a friendly, chatty young woman named Samantha.Limit your response to 100 words where possible.'
        print(f'prompt set to: {input_text}')
    elif input_text.startswith("R"):
        input_text =\
            """Define steps as:
orient: identify the task
thought: think step by step about the task. Identify ambiguities or logical consequences in the problem statement that impact your response to the task.
action: act to further progress on the task. Your available acts are:
i. answer: if the answer is available, respond with answer then respond STOP
ii. logic: use known fact or logical reasoning based on known fact to expand on thought, then respond with reasoning.
iii. ask: ask user a question to clarify the task. Display: question, Display: STOP
iv. search: use the llmsearch plugin to search the web for question based on thought
observation: revise or refine the task given thought and results of action

Define react as:
For step in steps: perform step
askUser 'should we continue?'
Display STOP

Assistant will concisely display all orient, thought, action, and observation
Assistant will follow the instructions in react above to respond to all questions and tasks.
"""
        print(f'prompt set to: {input_text}')
    else:
        print(f'valid prompts are N(one), H(elpful), B(hagavan), A(CT), F(riend), R(eact)')
    
    
    memory.set('prompt_text', input_text)
    PROMPT = Prompt([
        SystemMessage('{{$prompt_text}}'),
        ConversationHistory('history', .5),
        UserMessage('{{$input}}')
    ])

    
def clear():
    global memory, PREV_POS
    PREV_POS="1.0"
    input_area.delete("1.0", tk.END)
    memory.set('history', [])


functions = FunctionRegistry()
tokenizer = GPT3Tokenizer()
memory = VolatileMemory({'input':'', 'history':[]})
max_tokens = 2000
# Render the prompt for a Text Completion call
def render_text_completion():
    #print(f"\n***** chat memory pre render \n{memory.get('history')}\n")
    as_text = PROMPT.renderAsText(memory, functions, tokenizer, max_tokens)
    #print(as_text)
    text = ''
    if not as_text.tooLong:
        text = as_text.output
    return text

# Render the prompt for a Text Completion call
def render_messages_completion():
    #print(f"\n***** chat memory pre render input: \n{memory.get('input')}")
    #print(f"***** chat memory pre render \n{memory.get('history')}\n")
    as_msgs = PROMPT.renderAsMessages(memory, functions, tokenizer, max_tokens)
    msgs = []
    if not as_msgs.tooLong:
        msgs = as_msgs.output
    return msgs


host = '127.0.0.1'
port = 5004

def run_query(query):
    global model, temperature, top_p, max_tokens, memory
    try:
        memory.set('input', query)
        if FORMAT:
            msgs = render_messages_completion()
            for msg in msgs:
                print(str(msg))
            response = ut.ask_LLM(model, msgs, int(max_tokens), float(temperature), float(top_p), host, port)
            print(response)
            history = memory.get('history')
            history.append({'role':llm.USER_PREFIX, 'content': query.strip()})
            response = response.replace(llm.ASSISTANT_PREFIX+':', '')
            response = response.replace(llm.ASSISTANT_PREFIX, '')
            history.append({'role': llm.ASSISTANT_PREFIX, 'content': response.strip()})
            memory.set('history', history)
        else:
            # just send the raw input text to server
            llm.run_query(model, query, int(max_tokens), float(temperature), float(top_p), host, port)
    except Exception:
        traceback.print_exc()
        
while True:
    ip = input('?')
    if ip.strip().startswith('set '):
        if 'temp' in ip or 'top_p' in ip or 'max_tokens' in ip:
            set(text)
        elif 'format' in ip:
            setFormat()
        elif 'prompt' in ip:
            idx = ip.find('prompt')
            setPrompt(ip[idx+6:].strip())
        elif 'clear' in ip:
            clear()
    else:
        print(run_query(ip.strip()))
