import argparse
import json
import traceback
import socket
import requests
import re
from alphawave_pyexts.conversation import Conversation, SeparatorStyle
import alphawave_pyexts.conversation as cv
import time

MODEL_NAME = None; WORKER_ADDR=None; CONTROLLER_ADDRESS = "http://localhost:21001"
USER_PREFIX = 'user'
ASSISTANT_PREFIX = 'assistant'
SYSTEM_PREFIX = 'system'

host='192.168.1.195'
port = 5004


def get_available_models():
    return list(cv.conv_templates.keys())

def run_query(model, messages, max_tokens, temp, top_p, host = host, port = port, choice_set=None, display=None, format=True): 
    global USER_PREFIX, ASSISTANT_PREFIX, SYSTEM_PREFIX

    conv=cv.get_conv_template(model)
    user_prompt = conv.roles[0]
    asst_prompt = conv.roles[1]
    system_prompt = conv.roles[1]
    
    USER_PREFIX = conv.roles[0]
    ASSISTANT_PREFIX = conv.roles[1]
    SYSTEM_PREFIX = conv.system
    
    if format:
        prime=''
        for msg_idx, msg in enumerate(messages):
            #print(msg_idx, msg)
            role = msg['role']
            ### conv.system is a prompt msg, and will be inserted as the first entry by conv.get_prompt()
            if role.lower() == 'system' and msg_idx==0:
                if len(conv.system)>0:
                    prime = conv.system+'\n'+msg['content']
                else:
                    prime = msg['content']
                if len(conv.roles)>2:
                    conv.append_message(conv.roles[2], prime)
                    prime=''
            elif role.lower() == 'user' or role.lower() == 'system' or role == conv.roles[0]:
                role_index = 0
                conv.append_message(conv.roles[role_index], msg['content'])
            else:
                role_index = 1
                conv.append_message(conv.roles[role_index], msg['content'])
                #print(conv.messages[-1])

        # priming prompt
        if conv.response_prime:
            conv.append_message(conv.roles[1], '')

        prompt = conv.get_prompt()
        if len(prime) > 0:
            prompt = prime+conv.sep+prompt
    else:
        prompt = messages
    prompt = re.sub('\n{3,}', '\n\n', prompt)
    #print(f'***** llm output prompt string {prompt}')
    server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens, 'user_prompt':USER_PREFIX}
    if choice_set:
        server_message['choice_set']=choice_set

    response = requests.post('http://127.0.0.1:5004/', json=server_message, stream=True)
    if response.status_code == 200:
        # Read and print the streaming content
        response_text = ''
        print()
        for chunk in response.iter_lines():
            if chunk:
                line = chunk.decode('utf-8') + '\n'
                if display is not None:
                    display(line)
                response_text += line
        print()
        return response_text
    else:
        print(f' server return code {response.status_code}')
        return ''
    

