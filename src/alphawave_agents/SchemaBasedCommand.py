from pyee import AsyncIOEventEmitter
from jsonschema import validate, ValidationError
import toml
from toml import TomlDecodeError
from cerberus import Validator
from typing import Any, Dict, Optional, Union
import json
from promptrix.promptrixTypes import PromptMemory, PromptFunctions, Tokenizer
from dataclasses import dataclass, asdict
import traceback

@dataclass
class CommandSchema:
    def __init__(self, type: str, title: str='', description: str='', returns: Optional[str] = None):
        self.type = type
        self.title = title
        self.description = description
        self.returns = returns

class SchemaBasedCommand(AsyncIOEventEmitter):
    def __init__(self, schema: CommandSchema, title: Optional[str] = None, description: Optional[str] = None):
        super().__init__()
        self._schema = schema
        if self._schema is not None and type(self._schema) != dict:
            self._schema = self._schema.__dict__
        self._title = title
        self._description = description

    @property
    def description(self) -> Optional[str]:
        return self._description or self._schema['description']

    ### needs to be adapted for TOML
    @property
    def inputs(self) -> Optional[str]:
        if self._schema['properties']:
            properties = self._schema['properties'] or {}
            inputs = [f'"{key}":"<{property.get("description", property.get("type", "any"))}>"' for key, property in properties.items()]
            return ",".join(inputs)
        else:
            return None
    @property
    def output(self) -> Optional[str]:
        return self._schema['returns']

    @property
    def schema(self) -> CommandSchema:
        return self._schema

    @property
    def title(self) -> str:
        return self._title or self._schema['title']

    async def execute(self, inputs: Dict[str, Any], memory: 'PromptMemory', functions: 'PromptFunctions', tokenizer: 'Tokenizer') -> Any:
        raise NotImplementedError

    def validate(self, inputs: Dict[str, Any], memory: 'PromptMemory', functions: 'PromptFunctions', tokenizer: 'Tokenizer', syntax: str ='JSON') -> 'Validation':
        #print(f'***** SchemaBasedCommand validate inputs {inputs}\nschema\n{self._schema}')
        if self._schema is None:
            return {
                'type': 'Validation',
                'valid': True,
                'value': cleaned
            }
        if syntax == 'JSON':
            try:
                cleaned = self.clean_inputs(inputs)
                #print(f'***** SchemaBasedCommand validate assuming JSON schema:\n{self._schema}\ninputs:\n{inputs}\n')
                validate(cleaned, self._schema)
                #print(f'***** SchemaBasedCommand validate passed\n')
                return {
                    'type': 'Validation',
                    'valid': True,
                    'value': cleaned
                }
            except ValidationError as e:
                #print(f'***** SchemaBasedCommand validate fail {str(e)}\n')
                errors = f'"{e.path[0] if e.path else "inputs"}": {e.message}'
                message = errors
                if 'is a required property' in message:
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f"The ['command']['inputs'] field has errors:\n{message}\nRevise to use this format: {self._schema}"
                    }
                return {
                    'type': 'Validation',
                    'valid': False,
                    'feedback': f"The ['command']['inputs'] field has errors:\n{message}\n\nTry again."
                }
        else: ## not JSON, assume TOML
            #print(f'***** SchemaBasedCommand validate assuming TOML inputs type {type(inputs)}')
            try:
                if type(inputs) != dict:
                    #print(f'***** SchemaBasedCommand validate TOML inputs not dict')
                    inputs_as_dict = None
                    try:
                        inputs_as_dict = toml.loads(inputs)
                        #print(f'***** SchemaBasedCommand validate TOML inputs toml.loads success')
                    except:
                        try:
                            inputs_as_dict = json.loads(inputs)
                            #print(f'***** SchemaBasedCommand validate TOML json.loads success')
                        except:
                            pass
                    if inputs_as_dict is None:
                        #print(f'***** SchemaBasedCommand validate TOML inputs convert to dict fail')
                        return {
                            'type': 'Validation',
                            'valid': False,
                            'feedback': f"The ['command']['inputs'] field has errors, use this format: {self._schema}"
                        }

                else:
                    inputs_as_dict = inputs

                cleaned = self.clean_inputs(inputs_as_dict)
                #print(f'***** SchemaBasedCommand validate TOML inputs cleaned\n{cleaned}')
                validate(cleaned, self._schema)
                return {
                    'type': 'Validation',
                    'valid': True,
                    'value': cleaned
                }
            except Exception as e:
                return {
                    'type': 'Validation',
                    'valid': False,
                    'feedback': f"The ['command']['inputs'] field has errors: {str(e)}. use this format: {self._schema}"
                }
                
            
            
    def clean_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        #print(f'***** SchemaBasedCommand clean_inputs inputs\n{inputs}\n')
        cleaned = {}
        properties = self._schema['properties']
        if not properties:
            return cleaned

        for key, property in properties.items():
            type = property.get('type', 'any')
            value = inputs.get(key)

            if value is None or (isinstance(value, str) and value.startswith("<") and value.endswith(">")):
                continue

            if not isinstance(value, str):
                value = json.dumps(value) if isinstance(value, dict) else str(value)

            try:
                if type == 'string':
                    cleaned[key] = str(value)
                elif type == 'number':
                    cleaned[key] = float(value)
                elif type == 'boolean':
                    cleaned[key] = bool(value)
                elif type in ['array', 'object']:
                    cleaned[key] = json.loads(value)
            except Exception:
                pass

        return cleaned

    def one_shot(self, syntax):
        content=''
        if syntax == 'JSON':
            content += '\t\tformat: {'+f'"command":'+f'"{self.title}", "inputs":'+'{'
            args = ''
            for arg in self.inputs.split(','):
                arg_key_dscp = arg.split(':')
                key = arg_key_dscp[0].strip().strip('"\'') # strip key down to bare alpha key (':' elims type info)
                content += f'"{key}": "<{arg_key_dscp[1].strip()}>",'
                content = content[:-1]+'}}\n' # strip final comma 
            #print(f'***** AgentCommandSection one-shot prompt: {content}\n')
            return content
        else:
            content += f'\t\tformat:\n[RESPONSE]\ncommand="{self.title}"\n'
            args = ''
            for arg in self.inputs.split(','):
                arg_key_dscp = arg.split(':')
                key = 'inputs.'+arg_key_dscp[0].strip().strip('"\'') # strip key down to bare alpha key (':' elims type info)
                content += f'{key}="<{arg_key_dscp[1].strip()}>"\n'
                content +='[STOP]\n'
            #print(f'***** AgentCommandSection one-shot TOML: {content}\n')
            return content
