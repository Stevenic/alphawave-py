from jsonschema import validate, ValidationError
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from alphawave.alphawaveTypes import PromptResponse, Validation, PromptResponseValidator
from alphawave.Response import Response
from pyee import AsyncIOEventEmitter
import json
import ast
import traceback
import re

class JSONResponseValidator(PromptResponseValidator):
    def __init__(self, schema=None, missing_json_feedback='Invalid JSON. Revise your previous response and return valid JSON as per the earlier schema.'):
        self.schema = schema
        self.missing_json_feedback = missing_json_feedback

    def parse_dict(self, s):
        if s is None:
            return s
        # Try to parse as JSON
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        
        # Try to parse as a Python literal
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            pass

        # Try to repair common errors and parse again
        s = s.strip()
        if not (s.startswith('{') and s.endswith('}')):
            s = '{' + s + '}'
        s = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', s)  # add quotes around keys
        #s = s.replace("'", '"')  # replace single quotes with double quotes

        # Try to parse the repaired string
        try:
            y = json.loads(s)
            return y
        except json.JSONDecodeError:
            return s

    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts: int) -> Validation:
        message = response['message']
        
        text = message if isinstance(message, str) else message.get('content', '')

        # Parse the response text
        parsed=[]
        try:
            parsed = Response.parse_all_objects(text)
        except Exception as e:
            raise e
        if len(parsed) == 0:
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': self.missing_json_feedback
            }

        # Validate the response against the schema
        if self.schema:
            errors = None
            for i in range(len(parsed) - 1, -1, -1):
                obj = parsed[i]
                try:
                    try:
                        obj = self.parse_dict(obj) if type(obj) == str else obj
                    except Exception as e:
                        pass
                    validate(obj, self.schema)
                    return {
                        'type': 'Validation',
                        'valid': True,
                        'value': obj
                    }
                except ValidationError as e:
                    path = str(list(e.relative_schema_path)[1:-1]).replace('[','').replace(']',"").replace(', ', ':')
                    if not errors:
                        errors = e
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(errors)}. Revise your previous response to strict JSON format standards and this schema {self.schema}'
                    }
                except Exception as e:
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(e)}. Revise your previous response to strict JSON format standards and this schema {self.schema}'
                    }      
    
        else:
            # Return the last object
            return {
                'type': 'Validation',
                'valid': True,
                'value': parsed[-1]
            }

    def get_error_fix(self, error: ValidationError) -> str:
        # Get argument as a string
        arg = error.validator
        path = str(list(error.relative_schema_path)[1:-1]).replace('[','').replace(']',"").replace(', ', ':')
        
        switcher = {
            'type': f'convert "{path}" value to a {error.validator_value}' if len(path)> 0 else '',
            'anyOf': f'convert "{path}" to one of the allowed types: {error.validator_value}',
            'additionalProperties': f'remove the "{arg}" field from "{path}"',
            'required': f"add the {error.validator_value} fields to {path if len(path)>0 else 'response'}",
            'format': f'change the "{path}" field to be a {error.validator_value}',
            'uniqueItems': f'remove all duplicate items from "{error.path}"',
            'enum': f'change the "{path}" value to be one of these values: {arg}',
            'const': f'change the "{path}" value to be {arg}',
        }

        return switcher[arg] if arg in switcher else error.message

