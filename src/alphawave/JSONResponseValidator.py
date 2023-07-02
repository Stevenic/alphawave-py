from jsonschema import validate, ValidationError
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from alphawave.alphawaveTypes import PromptResponse, Validation, PromptResponseValidator
from alphawave.Response import Response
from pyee import AsyncIOEventEmitter
import json
import ast
import traceback
import re

def extract_json_template(schema, p=True):
    template = {}
    if schema is not None and "properties" in schema:
        for key, value in schema["properties"].items():
            if "type" in value:
                if value["type"] == "object":
                    template[key] = extract_json_template(value, p=False)
                elif value["type"] == "array":
                    if 'description' in value:
                        template[key] = '['+value['description']+']'
                    elif "items" in value:
                        template[key] = [extract_json_template(value["items"], p=False)]
                    else:
                        template[key] = []
                else:   # plain old string value
                    if 'description' in value:
                        template[key] = value['description']
                    else: template[key] = '<'+key+'>'
            elif "$ref" in value:
                ref = value["$ref"]
                ref_schema = schema
                for part in ref.split("/")[1:]:
                    ref_schema = ref_schema[part]
                template[key] = extract_json_template(ref_schema, p=False)

    if p:
        pass #print(json.dumps(template))
    return json.dumps(template)

class JSONResponseValidator(PromptResponseValidator):
    def __init__(self, schema=None, missing_json_feedback='Invalid JSON. return valid JSON.'):
        self.schema = schema
        self.missing_json_feedback = missing_json_feedback

    def parse_dict(self, s):
        if s is None:
            return s
        # Try to parse as JSON
        try:
            s = json.loads(s)
            return s
        except json.JSONDecodeError as e:
           pass
        
        # Try to parse as a Python literal
        try:
            sast = ast.literal_eval(s)
            sast = re.sub(r"'([^\"']+)':", r'"\1":', str(sast)) # keys as doublequote
            s=sast
            sastj = json.loads(sast)
            return sast
        except (SyntaxError, ValueError) as e:
            pass

        # Try to repair common errors and parse again
        s = s.strip()
        if not (s.startswith('{') and s.endswith('}')):
            s = '{' + s + '}'
        s = re.sub(r"'([^\"']+)':", r'"\1":', s) # keys as doublequote

        # Try to parse the repaired string
        try:
            y = json.loads(s)
            return y
        except json.JSONDecodeError:
            return s

    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts: int) -> Validation:
        message = response['message']
        
        text = message if isinstance(message, str) else message.get('content', '')
        #print(f'***** JSONResponseValidator input {text}')
        # Parse the response text
        text = re.sub('\n+', '\n', text)
        cleaned_text = ""
        for char in text:
            if ord(char) >= 10:
                cleaned_text += char
        text = cleaned_text
        parsed=[]
        #print(f'***** JSONResponseValidator cleaned \n{text}\n')
        try:
            parsed = Response.parse_all_objects(text)
        except Exception as e:
            raise e
        if len(parsed) == 0:
            template = extract_json_template(self.schema)
            #print(f'***** JSONResponseValidator failure len(parsed) == 0')
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': self.missing_json_feedback+f' using this template: {template} '
            }

        # Validate the response against the schema
        if self.schema:
            #print(f'***** JSONResponseValidator schema {self.schema}')
            errors = None
            for i in range(len(parsed)):  # return first one that passes
                obj = parsed[i]
                try:
                    try:
                        obj = self.parse_dict(obj) if type(obj) == str else obj
                    except Exception as e:
                        pass
                    validate(obj, self.schema)
                    #print(f'***** JSONResponseValidator validation passed!')
                    return {
                        'type': 'Validation',
                        'valid': True,
                        'value': obj
                    }
                except ValidationError as e:
                    path = str(list(e.relative_schema_path)[1:-1]).replace('[','').replace(']',"").replace(', ', ':')
                    if not errors:
                        errors = e
                    template = extract_json_template(self.schema)
                    #print(f'***** JSONResponseValidator ValidationError exception {str(e)}\n{self.schema}\n')
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(errors)}. respond using this template: {template} '
                    }
                except Exception as e:
                    template = extract_json_template(self.schema)
                    #print(f'***** JSONResponseValidator validator generic exception {str(e)}\n{self.schema}')
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(e)}. respond using this template: {template} '
                    }      
    
        else:
            # Return the last object
            #print(f'***** JSONResponseValidator exit last object {parsed[-1]}')
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

