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
USER_PREFIX = ''
ASSISTANT_PREFIX = ''

host='192.168.1.195'
port = 5004
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
        name="falcon_instruct2",
        system="",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)
cv.register_conv_template(Conversation(
        name="falcon_instruct3",
        system="",
        roles=("HUMAN", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)
cv.register_conv_template(Conversation(
        name="wizardLM",
        system="",
        roles=("### HUMAN", "### ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

def run_query(model, messages, max_tokens, temp, top_p, host = host, port = port, tkroot = None, tkdisplay=None, format=True): 
    global USER_PREFIX, ASSISTANT_PREFIX
    conv=cv.get_conv_template(model)
    USER_PREFIX = conv.roles[0]
    ASSISTANT_PREFIX = conv.roles[1]
        # set this so client can check for run-on, although this test should pbly be here!
    if format:
        for msg in messages:
            print(f'***** llm input msf {msg}')
            role = msg['role']
            if role.lower() == 'user' or role.lower()=='system':
                role_index = 0
            else:
                role_index = 1
            conv.append_message(conv.roles[role_index], msg['content'])
        prompt = conv.get_prompt()
    else:
        prompt = messages
    prompt = re.sub('\n{3,}', '\n\n', prompt)
    print(f'***** llm output prompt string {prompt}')
    server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens, 'user_prompt':USER_PREFIX}
    smj = json.dumps(server_message)
    try:
        client_socket = socket.socket()  # instantiate
        client_socket.connect((host, port))  # connect to the server
        client_socket.settimeout(240)
        server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens}
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
    
