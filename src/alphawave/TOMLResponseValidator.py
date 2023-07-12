import toml
from toml import TomlDecodeError
from cerberus import Validator
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from alphawave.alphawaveTypes import PromptResponse, Validation, PromptResponseValidator
from pyee import AsyncIOEventEmitter
import sys
import traceback
import re


class TOMLResponseValidator(PromptResponseValidator):
    def __init__(self, schema=None, missing_toml_feedback="Response was not formatted as expected. "):
        self.schema = schema
        self.missing_toml_feedback = missing_toml_feedback
        self.validator = Validator(schema)
        try:
            self.feedback_schema = self.extract_toml_template(schema)
        except Exception as e:
            traceback.print_exc()
        #print(f'***** TOMLResponseValidator feedback {self.feedback_schema}')
        
    def find_toml(self, s):
        if s is None:
            return s
        # Look for [RESPONSE] ... [STOP]

        s = re.sub('\n+', '\n', s)
        cleaned_s = ""
        for char in s:
            if ord(char) >= 10:
                cleaned_s += char
        s = cleaned_s
        s = s.replace('\\*', '*')
        s = s.replace('\\+', '+')
        s = s.replace('\\-', '-')
        s = s.replace('\\/', '/')
        s = s.replace('RESPONSE:', '[RESPONSE]')
        start = s.find('[RESPONSE]')
        if start < 0:
            #print(f'***** find_toml no start')
            return ''
        end = s[start:].find('[STOP]')
        if end < 0:
            end = (s[start+1:]).find('[')
        if end < 0:
            #print(f'***** find_toml {start} no end\n{s[start:]}')
            return s[start:]
        else:
            #print(f'***** find_toml {start}, {end}\n{s[start:start+end-1]}')
            return s[start:start+end-1]

    def extract_toml_template(self,schema, prefix=None):
        if prefix is None:  ### top level
            #print(f'\n***** template extract input\n{schema}')
            template = '[RESPONSE]\n'
        else:
            template = ''
        if schema is not None and type(schema) == dict:
            for item in schema:
                value = schema[item]
                if prefix is None:
                    key = item
                else:
                    key = prefix+"."+item
                if type(value) == dict:
                    if 'type' in value and value['type'] == 'dict':
                        #print(f'\n***** template dict {key},{value}')
                        # recurse on nested structure
                        if 'schema' in value:
                            subtemplate = self.extract_toml_template(value['schema'], key)
                            template = template+subtemplate
                        elif 'keysrules' in value:
                            template = template+key+"={}\n"
                    elif 'meta' in value:
                        template = template+key+'="'+value['meta']+'"\n'
                    elif 'type' in value:
                        template = template+key+'="'+value['type']+'"\n'
                else:
                    template = template+key+'='+value+'\n'
                    
        if prefix is None:
            template += '[STOP]\n'
        #print(f'\n***** template extract \n{template}')
        return template
                
    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts: int) -> Validation:
        message = response['message']
        
        text = message if isinstance(message, str) else message.get('content', '')
        #print(f'***** TOMLResponseValidator raw input type {type(text)}\n{text}\n')
        try:
            toml_extract = self.find_toml(text)
        except Exception as e:
            traceback.print_exc()
        #print(f'\n***** TOMLResponseValidator toml_extract \n{toml_extract}\n')
        if toml_extract == '':
            #print(f'***** TOMLResponseValidator failure no toml', file=sys.stderr)
            return {
                'type': 'Validation',
                'valid': 'True',
                'feedback': self.missing_toml_feedback+f' using this template:\n{self.feedback_schema}',
                'value': text
            }

        # Validate the response against the schema
        errors = None
        try:
            response_as_dict = toml.loads(toml_extract)
            #print(f'\n***** TOMLResponseValidator as_dict \n{response_as_dict}\n')
        except TomlDecodeError as e:
            #print(f'***** TOMLResponseValidator TomlDecodeError {str(e)}\n{toml_extract}')
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f'Response found but not a complete TOML form. e fixes:\n{str(e)}. respond using this format:\n{self.feedback_schema} '
            }

        if self.schema:
            try:
                #print(f'***** TOMLResponseValidator validate input against \n{self.schema}')
                v = Validator(self.schema)
                validation_result = v.validate(response_as_dict['RESPONSE'])
                if validation_result:
                    #print(f'***** TOMLResponseValidator validation passed!')
                    return {
                        'type': 'Validation',
                        'valid': True,
                        'value': response_as_dict['RESPONSE']
                    }
                else:
                    #print(f'***** TOMLResponseValidator schema validation failed {v._errors}\n{response_as_dict}\n')
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f'Response TOML found but didnt match schema. Apply these fixes:\n{v._errors} response template:\n{self.feedback_schema}\n'
                    }
            except Exception as e:
                #print(f'***** TOMLResponseValidator validator exception {str(e)}')
                return {
                    'type': 'Validation',
                    'valid': False,
                    'feedback': f'TOML form found but exception attempting to validate. Repair and respond using this template:\n{self.feedback_schema} '
                }      
    
        else:
            # Return the last object
            #print(f'***** TOMLResponseValidator exit last object {response_as_dict}')
            return {
                'type': 'Validation',
                'valid': True,
                'value': response_as_dict['RESPONSE']
            }

