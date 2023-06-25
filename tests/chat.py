import argparse
import json
import socket
import os
import traceback
import tkinter as tk
import tkinter.font
from tkinter import ttk
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

USER_PREFIX = 'User'
ASSISTANT_PREFIX = 'Assistant'
STOP_1 = '</s>'
STOP_2 = USER_PREFIX
FORMAT=True

prompt_text = 'You are helpful, creative, clever, and very friendly. '
PROMPT = Prompt([
    SystemMessage(prompt_text),
    ConversationHistory('history', .5),
    UserMessage('{{$input}}')
])

#print(f' models: {llm.get_available_models()}')
parser = argparse.ArgumentParser()
#parser.add_argument('model', type=str, default='wizardLM', choices=['guanaco', 'wizardLM', 'zero_shot', 'vicuna_v1.1', 'dolly', 'oasst_pythia', 'stablelm', 'baize', 'rwkv', 'openbuddy', 'phoenix', 'claude', 'mpt', 'bard', 'billa', 'h2ogpt', 'snoozy', 'manticore', 'falcon_instruct', 'gpt_35', 'gpt_4'],help='select prompting based on modelto load')

parser.add_argument('model', type=str, default='wizardLM', choices=llm.get_available_models(),help='select prompting based on modelto load')

parser.add_argument('--user', type=str, default='', help='user prefix, overrides model template')
parser.add_argument('--asst', type=str, default='', help='assistant prefix, overrides model template')
parser.add_argument('--stop1', type=str, default='',help='stop string')
parser.add_argument('--stop2', type=str, default='', help='alternative stop string')
args = parser.parse_args()
if args.user:
    USER_PREFIX = args.user
    STOP_2 = args.user
if args.asst:
    ASSISTANT_PREFIX = args.asst
if args.stop1:
    STOP_1 = args.stop1
if args.stop2:
    STOP_2 = args.stop2
if args.model:
    model = args.model
else:
    model = 'wizardLM'



def show_prompt():
    pass

def set_model():
    pass

def set(text):
    global temperature, top_p, max_tknns
    try:
        if 'temp' in text:
            idx = text.find('=')
            if idx > 0:
                temperature = float(text[idx+1:].strip())
        if 'top_p' in text:
            idx = text.find('=')
            if idx > 0:
                top_p = float(text[idx+1:].strip())
        if 'max_tkns' in text:
            idx = text.find('=')
            if idx > 0:
                max_tkns = int(text[idx+1:].strip())
    except Exception as e:
        print ("parse failed", e)

PREV_LEN = 0
def submit():
    global PREV_LEN
    input_text = input_area.get("1.0", tk.END)
    if FORMAT and len(input_text) > PREV_LEN:
        new_text = input_text[PREV_LEN:].strip()
    else:
        new_text = input_text.strip()
    input_area.insert(tk.END, '\n')
    response = run_query(new_text)
    input_area.insert(tk.END, '\n')
    PREV_LEN=len(input_area.get("1.0", tk.END))
    return(input, response)

def setFormat():
    global FORMAT
    if FORMAT:
        format_button.config(text='RAW')
        FORMAT=False
    else:
        format_button.config(text='FORMAT')
        FORMAT=True


def setPrompt():
    global PROMPT
    input_text = input_area.get("1.0", tk.END)
    input_text = input_text.strip()
    if input_text.startswith("NoPrompt"):
        input_text = ''
    if input_text.startswith("Helpfull"):
        input_text = 'Respond as a helpful, friendly, information assistant. Say "I don\'t know" if you have no information"'
    if input_text.startswith("Bhagaven"):
        input_text = 'Respond as a compassionate, empathetic, self-realized follower of Ramana Maharshi'
    if input_text.startswith("ACT"):
        input_text = 'Respond as a compassionate, empathetic, counselor familiar with Acceptance Commitment Theray'
    if input_text.startswith("Flirt"):
        input_text = 'Respond as a friendly, chatty, flirty young woman named Samantha. You are 21 years old.'
    if input_text.startswith("Agent"):
        input_text =\
            """INSTRUCTION"
Define steps as:
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
END INSTRUCTION
"""
    prompt_text = input_text
    PROMPT = Prompt([
        UserMessage(prompt_text),
        ConversationHistory('history', .5),
        UserMessage('{{$input}}')
    ])

    
def clear():
    global memory, PREV_POS
    PREV_POS="1.0"
    input_area.delete("1.0", tk.END)
    memory.set('history', [])

root = tk.Tk()

root.title(args.model)
root.geometry("1260x1024")
#root.config(cursor="watch")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

textFrame = tk.Frame(root, bg="#010101")
controlFrame = tk.Frame(root, bg="#101010")#, width=2)
textFrame.grid(row=0, column=0, sticky="nsew")
controlFrame.grid(row=0, column=1, sticky="nsew")

searchLevel = 0  # 0 == no search, 1 = normal search, 2 = deep search

#bold16 = tkinter.font.Font("Arial", size=16)
#normal14 =tkinter.font.Font("Arial", size=14)

def on_value_change(*args):
    print("The selected option is:", float(temperature.get()))
    
# Create a style
style = ttk.Style()

# Modify the label style
style.configure("TLabel", background="grey", foreground="black", font=("Arial", 12))

# Modify the combobox style
style.configure("TCombobox", fieldbackground="grey", foreground="black", selectbackground="grey", font=("Arial", 12))
# Set the font of the dropdown list
style.configure("TCombobox", selectbackground='gray', fieldbackground='black', background='light gray')
style.map('TCombobox', fieldbackground=[('readonly','grey')])
style.configure("TCombobox", font=('Arial', 12))  # sets the font in the entry part
style.configure("TCombobox.Listbox", font=('Arial', 12), background='grey')  # sets the font in the dropdown list part



input_area = tk.Text(textFrame, height=80,bg='black', fg='white', font=("Arial", 11))
input_area.grid(row=0, column=0, sticky='nsew')
input_area.pack(expand=True)
submit_button = tk.Button(controlFrame,  text="Submit", command=submit, font=("Arial", 12), bg='grey')
submit_button.pack(side='top', fill='x')

clear_button = tk.Button(controlFrame,  text="Clear", command=clear, font=("Arial",12), bg='grey')
clear_button.pack(side='top', fill='x')

temperature = tk.StringVar()
temperature.set('.1')
temperature.trace("w", on_value_change)
temp_label = ttk.Label(controlFrame, text="Temperature", style='TLabel')
temp_label.pack(side='top', fill='x')
temp_button = ttk.Combobox(controlFrame, textvariable=temperature, values = [.01,.1,.2,.4,.5,.7,.9,1.0], style='TCombobox')
temp_button.pack(side='top', fill='x')

top_p = tk.StringVar()
top_p.set('1.0')
top_p_label = ttk.Label(controlFrame, text="Top_p", style='TLabel')
top_p_label.pack(side='top', fill='x')
top_p_button = ttk.Combobox(controlFrame, textvariable=top_p, values = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0], style='TCombobox')
top_p_button.pack(side='top', fill='x')

max_tkns = tk.StringVar()
max_tkns.set('250')
max_tok_label = ttk.Label(controlFrame, text="MaxTokens", style='TLabel')
max_tok_label.pack(side='top', fill='x')
max_tok_button = ttk.Combobox(controlFrame, textvariable=max_tkns, values = [50,100,250,500,1000,1500,2000,4000,8000], style='TCombobox')
max_tok_button.pack(side='top', fill='x')

format_button = tk.Button(controlFrame, text="Format", command=setFormat, font=("Arial", 12), bg='grey')
format_button.pack(side='top', fill='x')

prompt_button = tk.Button(controlFrame, text="setPrompt", command=setPrompt, font=("Arial", 12), bg='grey')
prompt_button.pack(side='top', fill='x')

#model_button = tk.Button(controlFrame, text="gpt-3.5-turbo", command=set_model, font=("Arial", 12), bg='grey')
#model_button.pack(side='top', fill='x')

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

functions = FunctionRegistry()
tokenizer = GPT3Tokenizer()
memory = VolatileMemory({'input':'', 'history':[]})
max_tokens = 2000
# Render the prompt for a Text Completion call
async def render_text_completion():
    #print(f"\n***** chat memory pre render \n{memory.get('history')}\n")
    as_text = await PROMPT.renderAsText(memory, functions, tokenizer, max_tokens)
    #print(as_text)
    text = ''
    if not as_text.tooLong:
        text = as_text.output
    return text

# Render the prompt for a Text Completion call
async def render_messages_completion():
    #print(f"\n***** chat memory pre render input: \n{memory.get('input')}")
    #print(f"***** chat memory pre render \n{memory.get('history')}\n")
    as_msgs = await PROMPT.renderAsMessages(memory, functions, tokenizer, max_tokens)
    msgs = []
    if not as_msgs.tooLong:
        msgs = as_msgs.output
    return msgs


host = '127.0.0.1'
port = 5004

def run_query(query):
    global model, temperature, top_p, max_tkns, memory
    try:
        memory.set('input', query)
        if FORMAT:
            msgs = asyncio.run(render_messages_completion())
            #for msg in msgs:
            #    print(str(msg))
            response = ut.ask_LLM(model, msgs, int(max_tkns.get()), float(temperature.get()), float(top_p.get()), host, port, root, input_area)
            #print(response)
            history = memory.get('history')
            #print(f'***** chat post query user {llm.USER_PREFIX}, {llm.ASSISTANT_PREFIX}')
            #query = query.replace(llm.ASSISTANT_PREFIX+':', '')
            #query = query.replace(llm.ASSISTANT_PREFIX, '')
            #print(f'***** chat post query query {query}')
            history.append({'role':llm.USER_PREFIX, 'content': query.strip()})
            response = response.replace(llm.ASSISTANT_PREFIX+':', '')
            response = response.replace(llm.ASSISTANT_PREFIX, '')
            #print(f'response asst_prefix {llm.ASSISTANT_PREFIX}, response post-removal{response}')
            #print(f'***** chat post query response {response}')
            history.append({'role': llm.ASSISTANT_PREFIX, 'content': response.strip()})
            memory.set('history', history)
        else:
            # just send the raw input text to server
            llm.run_query(model, query, int(max_tkns.get()), float(temperature.get()), float(top_p.get()), host, port, root, input_area, format=False)
    except Exception:
        traceback.print_exc()
        
if args.user is not None:
    USER_PREFIX = args.user
    STOP_2=args.user
if args.asst is not None:
    ASSISTANT_PREFIX = args.asst
if args.stop1 is not None:
    STOP_1 = args.stop1
if args.stop2 is not None:
    STOP_2 = args.stop2
if args.model is not None:
    model = args.model
else:
    model = 'wizardLM-30B'




root.mainloop()

"""
Find an arithmetic expression over the integers 2, 3, and 6 that evaluates to 12. You must use each integer exactly once. Use the following approach. Create a trial expression. Evaluate the results. If it is false, build hypothesis for generating a better expression, then generate a new trial expression and evaluate again. Do this until the correct result is found. If the result is 12 then the process is over.
"""

"""
Human: you are an instruction-following AI. Follow these instructions: Find an arithmetic expression over the integers 2, 3, and 6 that evaluates to 12. You must use each integer exactly once. Use the following approach.
Loop:
  Generate a trial expression using previously generated hypotheses, if any
  Evaluate the trial expression. 
  Test if the evaluation equals 12?
  If not, build a hypothesis to obtain higher success rate in expression generation.
Untill Test is True.
"""
