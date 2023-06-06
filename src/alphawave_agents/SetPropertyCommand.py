from typing import Dict, Any
from abc import ABC, abstractmethod

class PromptMemory:
    def __init__(self):
        self.memory = {}

    def set(self, property: str, value: str):
        self.memory[property] = value

class PromptFunctions:
    pass

class Tokenizer:
    pass

class CommandSchema:
    def __init__(self, type: str, title: str, description: str, properties: Dict[str, Any], required: List[str], returns: str):
        self.type = type
        self.title = title
        self.description = description
        self.properties = properties
        self.required = required
        self.returns = returns

class SetPropertyCommandInput:
    def __init__(self, property: str, value: str):
        self.property = property
        self.value = value

class SchemaBasedCommand(ABC):
    @abstractmethod
    def __init__(self, schema: CommandSchema, title: str = None, description: str = None):
        self.schema = schema
        self.title = title
        self.description = description

    @abstractmethod
    def execute(self, input: Any, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer):
        pass

class SetPropertyCommand(SchemaBasedCommand):
    def __init__(self, title: str = None, description: str = None):
        schema = CommandSchema(
            type="object",
            title="setProperty",
            description="writes a string value to short term memory",
            properties={
                "property": {
                    "type": "string",
                    "description": "name of the property to set"
                },
                "value": {
                    "type": "string",
                    "description": "value to remember"
                }
            },
            required=["property", "value"],
            returns="confirmation of the assignment"
        )
        super().__init__(schema, title, description)

    def execute(self, input: SetPropertyCommandInput, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer):
        memory.set(input.property, input.value)
        return f'the "{input.property}" property was updated'
