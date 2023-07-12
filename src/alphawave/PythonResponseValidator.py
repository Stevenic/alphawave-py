from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from alphawave.alphawaveTypes import PromptResponse, Validation, PromptResponseValidator
from alphawave.Response import Response
import json
import ast
import traceback
import re
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO

class WritableObject(object):
    "dummy output stream for pylint"
    def __init__(self):
        self.content = []
    def write(self, st):
        "dummy write"
        self.content.append(st)
    def read(self):
        "dummy read"
        return self.content

class PythonResponseValidator(PromptResponseValidator):
    def __init__(self):
        pass

    def get_python(self, text):
        # change - there may be imports before class
    
        key = "```"
        key_len = len(key)
        start = text.find(key)
        if start < 0:
            print ('****** python extract cant find starting key')
            return text, text
        text = text[start+key_len:].lstrip(' \n,')
        print(f'****** python extract starting on {text[:10]}')
        if text.startswith('python'):
            print(f'****** python extract stripping starting "python"')
            text = text[6:]
            text = text.lstrip('\n')
            print(f'****** python extract stripped starting "python" {text[:24]}')
        
        # look for class name
        lines = text.split('\n')
        class_name = None
        for line in lines:
            print(f'   {line}')
            if line.lstrip(' \n').startswith('class'):
                class_name = (line[6:].strip()).replace(":", '')
                if '(' in class_name:
                    class_name = class_name[:class_name.find('(')]
                print(f'     found class name {class_name}')
                break
        end = text.find("```")
        if end < 0:
            print('****** python extract cant find end key')
            return class_name, text
        return class_name, text[:end].rstrip()

    """
    def get_python(self, text):
        # returns class name and code
        # change - there may be imports before class
        key = "```"
        key_len = len(key)
        start = text.find(key)
        if start < 0:
            print ('****** python extract cant find starting key')
            return '', text
        text = text[start+key_len:].lstrip()
        #print(f'****** python extract starting on {text}')
        if text.startswith('python'):
            #print('****** python extract stripping starting "python"')
            text = text[6:].lstrip()
        lines = text.split('\n')
        class_name = None
        for line in lines:
            #print(f'   {line}')
            if line.lstrip().startswith('class'):
                class_name = (line[6:].strip()).replace(":", '')
                if '(' in class_name:
                    class_name = class_name[:class_name.find('(')]
                print(f'     found class name {class_name}')
                break
        end = text.find("```")
        if end < 0:
            print('****** python extract cant find end key')
            return class_name, text
        #text = text.replace('\\*', '*')
        #text = text.replace('\\+', '+')
        #text = text.replace('\\-', '-')
        #text = text.replace('\\/', '/')
        return class_name, text[:end].rstrip()
    """
    
    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer , response: PromptResponse, remaining_attempts: int = 0) -> Validation:
        message = response['message']
        raw_text = message if isinstance(message, str) else message.get('content', '')
        class_name, code = self.get_python(raw_text)
        if len(code) == 0:
            print(f'***** PythonResponseValidator failure len(parsed) == 0')
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': 'No python code found'
            }

        # Validate the response against pylint
        try:
            print(f'***** PythonResponseValidator code \n{code}\n')
            with open(class_name+'.py', 'w') as cd:
                cd.write(code)
            pylint_output = WritableObject()
            Run(['--errors-only', class_name+'.py'], reporter=TextReporter(pylint_output), exit=False)
            error_msgs = ''
            for l in pylint_output.read():
                if len(l) > 4 and not l.startswith('***'):
                    error_msgs += l
            if len(error_msgs) > 0:
                print(f'***** PythonResponseValidator code failed \n{error_msgs}\n')
                return {
                    'type': 'Validation',
                    'valid': False,
                    'feedback': error_msgs
                }
            else:
                print(f'***** PythonResponseValidator code valid')
                return {
                    'type': 'Validation',
                    'valid': True,
                    'value': code
                }
        except Exception as e:
            print(f'***** PythonResponseValidator exception {str(e)}')
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': str(e)
            }
            
