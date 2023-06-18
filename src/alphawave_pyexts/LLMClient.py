import argparse
import json
import traceback
import socket
import requests
from alphawave_pyexts.conversation import Conversation, SeparatorStyle
import alphawave_pyexts.conversation as cv
import tkinter as tk
import time

MODEL_NAME = None; WORKER_ADDR=None; CONTROLLER_ADDRESS = "http://localhost:21001"

host='192.168.1.195'
port = 5004
cv.register_conv_template(Conversation(
        name="wizard",
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

def run_query(model, messages, temp, top_p, max_tokens, host = host, port = port, tkroot = None, tkdisplay=None): 
    conv=cv.get_conv_template(model)
    # set this so client can check for run-on, although this test should pbly be here!
    for msg in messages:
        #print(f'  {msg}')
        role = msg['role']
        if role.lower() == 'user':
            role_index = 0
        else:
            role_index = 1
        conv.append_message(conv.roles[role_index], msg['content'])
    conv.append_message(conv.roles[1], '')
    prompt = conv.get_prompt()
    server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens}
    """
    prompt = ''
    for msg in messages:
        if not isinstance(msg, dict):
            msg = msg.__dict__
        role = msg['role']
        if role.lower() == 'user' or role.lower() == 'system':
            role_os = USER
        elif role.lower() == 'assistant':
            role_os = ASSISTANT
        else: print(f'***** unknown role {role}')
        prompt += role_os + str(msg['content'])
    prompt += ASSISTANT
    """
    server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens}
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
        client_socket.close()  # close the connection
        #check for run on hallucination in response 
        runon_idx = response.find(conv.roles[0])
        if runon_idx > 0:
            response = response[:runon_idx]
        return response
    except: 
        traceback.print_exc()
    return ''
    
