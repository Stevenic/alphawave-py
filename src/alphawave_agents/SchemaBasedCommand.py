from pyee import AsyncIOEventEmitter
from jsonschema import validate, ValidationError
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
        self._title = title
        self._description = description

    @property
    def description(self) -> str:
        return self._description or self._schema.description

    @property
    def inputs(self) -> Optional[str]:
        if self._schema.properties:
            properties = self._schema.properties or {}
            inputs = [f'"{key}":"<{property.get("description", property.get("type", "any"))}>"' for key, property in properties.items()]
            return ",".join(inputs)
        else:
            return None

    @property
    def output(self) -> Optional[str]:
        return self._schema.returns

    @property
    def schema(self) -> CommandSchema:
        return self._schema

    @property
    def title(self) -> str:
        return self._title or self._schema.title

    async def execute(self, input: Dict[str, Any], memory: 'PromptMemory', functions: 'PromptFunctions', tokenizer: 'Tokenizer') -> Any:
        raise NotImplementedError

    async def validate(self, input: Dict[str, Any], memory: 'PromptMemory', functions: 'PromptFunctions', tokenizer: 'Tokenizer') -> 'Validation':
        try:
            cleaned = self.clean_input(input)
            if self._schema is not None:
                validate(cleaned, asdict(self._schema))
            return {
                'type': 'Validation',
                'valid': True,
                'value': cleaned
            }
        except ValidationError as e:
            errors = f'"{e.path[0] if e.path else "input"}": {e.message}'
            message = errors
            if 'is a required property' in message:
                return {
                    'type': 'Validation',
                    'valid': False,
                    'feedback': f"The ['command']['input'] field has errors:\n{message}\n\nRevise to include the field with valid value."
                }
                
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f"The ['command']['input'] field has errors:\n{message}\n\nTry again."
            }

    def clean_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        properties = self._schema.properties
        if not properties:
            return cleaned

        for key, property in properties.items():
            type = property.get('type', 'any')
            value = input.get(key)

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
