import argparse
import json
import traceback
import socket
import requests
import re
from alphawave_pyexts.conversation import Conversation, SeparatorStyle
import alphawave_pyexts.conversation as cv
import tkinter as tk
import time

MODEL_NAME = None; WORKER_ADDR=None; CONTROLLER_ADDRESS = "http://localhost:21001"
USER_PREFIX = 'user'
ASSISTANT_PREFIX = 'assistant'
SYSTEM_PREFIX = 'system'

host='192.168.1.195'
port = 5004
cv.register_conv_template(Conversation(
        name="dolly",
        system="### Instruction",
        roles=("### Input", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="### Response",
    )
)
cv.register_conv_template(Conversation(
        name="falcon_instruct",
        system="",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\nAssistant:\n",
    )
)

cv.register_conv_template(Conversation(
        name="guanaco",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="Assistant:\n",
    )
)
cv.register_conv_template(Conversation(
        name="wizardLM",
        system="### Instruction",
        roles=("### Input", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n",
    )
)

def get_available_models():
    return list(cv.conv_templates.keys())

def run_query(model, messages, max_tokens, temp, top_p, host = host, port = port, tkroot = None, tkdisplay=None, format=True): 
    global USER_PREFIX, ASSISTANT_PREFIX, SYSTEM_PREFIX

    conv=cv.get_conv_template(model)
    user_prompt = conv.roles[0]
    asst_prompt = conv.roles[1]
    system_prompt = conv.roles[1]
    
    USER_PREFIX = conv.roles[0]
    ASSISTANT_PREFIX = conv.roles[1]
    SYSTEM_PREFIX = conv.system
    
    # set this so client can check for run-on, although this test should pbly be here!
    if format:
        conv.system='' # no default prompt
        for msg in messages:
            #print(f'***** llm input msf {msg}')
            role = msg['role']
            ### conv.system is a prompt msg, and will be inserted as the first entry by conv.get_prompt()
            if role.lower() == 'system':
                conv.system = msg['content']
                continue
            elif role.lower() == 'user' or role.lower() == 'system' or role == conv.roles[0]:
                role_index = 0
            else:
                role_index = 1
            conv.append_message(conv.roles[role_index], msg['content'])

            #priming prompt
            conv.append_message(conv.roles[1], '')
        prompt = conv.get_prompt()

    else:
        prompt = messages
    prompt = re.sub('\n{3,}', '\n\n', prompt)
    #print(f'***** llm output prompt string {prompt}')
    server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens, 'user_prompt':USER_PREFIX}
    smj = json.dumps(server_message)
    try:
        client_socket = socket.socket()  # instantiate
        client_socket.connect((host, port))  # connect to the server
        client_socket.settimeout(240)
        server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens, 'user':user_prompt, 'asst':asst_prompt}
        smj = json.dumps(server_message)
        client_socket.sendall(smj.encode('utf-8'))
        client_socket.sendall(b'x00xff')
        response = ''
        while True:
            s = client_socket.recv(1024) # break every 32 chars to stream output
            if s is None or not s:
                break
            if (len(s) > 5 and s[-3:] == b'xff' and s[-6:-3] == b'x00'):
                s = s[:-6].decode('utf-8')
                s = s.replace('</s>', '')
                s = s.replace('<s>', '')
                if tkdisplay is not None:
                    tkdisplay.insert(tk.END, s)
                    if tkroot is not None:
                        tkroot.update()
                response += s
                break
            else:
                s = s.decode('utf-8')
                s = s.replace('</s>', '')
                s = s.replace('<s>', '')
                if tkdisplay is not None:
                    tkdisplay.insert(tk.END, s)
                    if tkroot is not None:
                        tkroot.update()
                response += s
                sep2_idx = response.find(conv.sep2)
                if sep2_idx >= 0:
                    response = response[sep2_idx:]
                    break
        client_socket.close()  # close the connection
        #check for run on hallucination in response 
        runon_idx = response.find(conv.roles[0])
        if runon_idx > 0:
            response = response[:runon_idx]
        return response
    except: 
        traceback.print_exc()
    return ''
    


############################################ old fastchat code, needs update to new socket interface.

def send_w_stream(model, messages, temperature=0.1, max_tokens= 500):
    global MODEL_NAME, WORKER_ADDR, CONTROLLER_ADDRESS
    MODEL_NAME = model_nm
    CONTROLLER_ADDRESS = controller_addr
    ret = requests.post(controller_addr + "/refresh_all_workers")
    ret = requests.post(controller_addr + "/list_models")
    models = ret.json()["models"]
    models.sort()

    if model_name not in models:
        print(f'***** Fastchat {model_name} not available')
        print(f'***** Fastchat available models {models}')
        return

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_nm}
    )
    WORKER_ADDR = ret.json()["address"]
    print(f"worker_addr: {WORKER_ADDR}")
    conv=conv.get_conv_template(model_name)

    for msg in messages:
        role = msg['role']
        if role.lower() == 'user':
            role_index = 0
        else:
            role_index = 1
        conv.append_message(conv.roles[role_index], msg['content'])
    conv.append_message(conv.roles[1], '')
    prompt = conv.get_prompt()

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    response = requests.post(
        WORKER_ADDR + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
    )

    prev = 0
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            output = data["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)

    return output
