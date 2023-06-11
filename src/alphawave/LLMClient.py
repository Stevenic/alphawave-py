import argparse
import json
import traceback
import socket
import requests
from fastchat.conversation import Conversation, SeparatorStyle
import fastchat.conversation
import tkinter as tk
import time

MODEL_NAME = None; WORKER_ADDR=None; CONTROLLER_ADDRESS = "http://localhost:21001"
USER = '\n### HUMAN:\n'
ASSISTANT = '\n### ASSISTANT:\n'

host='192.168.1.195'
port = 5004
def run_query(messages, temp, top_p, max_tokens, tkroot = None, tkdisplay=None): 
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
        return response
    except: 
        traceback.print_exc()

    response = ''
    """try:
        client_socket = socket.socket()  # instantiate
        client_socket.connect((host, port))  # connect to the server
        client_socket.settimeout(240)
        
        client_socket.sendall(smj.encode('utf-8'))
        client_socket.sendall(b'x00xff')
        try:
            while True:
                s = client_socket.recv(64)
                if s is None or not s:
                    break
                if (len(s) > 5 and s[-3:] == b'xff' and s[-6:-3] == b'x00'):
                    s = s[:-6].decode('utf-8')
                    response +=  s
                    break
                else:
                    response +=  s.decode('utf-8')
            client_socket.close()  # close the connection
        except KeyboardInterrupt:
            pass
    except Exception:
        traceback.print_exc()
    return response
    """
    
def send(messages, temperature=0.0, max_tokens= 500):
    global MODEL_NAME, WORKER_ADDR, CONTROLLER_ADDRESS
    ###
    ### build prompt
    #
    conv=fastchat.conversation.get_conv_template('wizard')
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
        WORKER_ADDR + "/worker_generate_completion",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    return response

def initialize(model_nm, controller_addr=CONTROLLER_ADDRESS):
    global MODEL_NAME, WORKER_ADDR, CONTROLLER_ADDRESS
    MODEL_NAME = model_nm
    CONTROLLER_ADDRESS = controller_addr
    ret = requests.post(controller_addr + "/refresh_all_workers")
    ret = requests.post(controller_addr + "/list_models")
    models = ret.json()["models"]
    models.sort()


    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_nm}
    )
    WORKER_ADDR = ret.json()["address"]

    if WORKER_ADDR == "":
        return

def send_w_stream(messages, temperature=0.0, max_tokens= 500):
    global MODEL_NAME, WORKER_ADDR, CONTROLLER_ADDRESS
    ###
    ### build prompt
    #
    #conv = get_conversation_template(MODEL_NAME)
    #conv.append_message(conv.roles[0], args.message)
    #conv.append_message(conv.roles[1], None)
    #prompt = conv.get_prompt()

    ###
    ### build prompt
    #
    conv=fastchat.conversation.get_conv_template('wizard')
    """
    conv = Conversation(
        name="wizard",
        system="",
        roles=("USER", "ASSISTANT"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
    """
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
    print("")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--message", type=str, default="Tell me a story with more than 1000 words."
    )
    args = parser.parse_args()

    main()
