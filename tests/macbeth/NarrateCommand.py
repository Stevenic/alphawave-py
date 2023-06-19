from typing import Any, Dict
import asyncio
from colorama import Fore, Style
from dataclasses import dataclass, asdict
from promptrix.promptrixTypes import PromptMemory
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
from alphawave_agents.SchemaBasedCommand import CommandSchema as sbcCommandSchema

class NarrateCommandInput:
    def __init__(self, text: str, performance: str = None):
        self.text = text
        self.performance = performance

@dataclass
class CommandSchema(sbcCommandSchema):
    type: str
    title: str
    description: str
    properties: Dict[str,Dict[str,str]]
    required: list[str]
    returns: str

schema = CommandSchema(
    type = "object",
    title = 'narrate',
    description = 'add narration to the story or set the scene',
    properties = {
        'text': {
            'type': 'string',
            'description': 'narration text'
        },
        'performance': {
            'type': 'string',
            'description': 'current act and scene'
        }
    },
    required = ['text'],
    returns = 'confirmation'
)

class NarrateCommand(SchemaBasedCommand):
    def __init__(self, title=schema.title, description=schema.description ):
        super().__init__(schema, title, description)

    async def execute(self, input: NarrateCommandInput, memory: PromptMemory, functions: Any, tokenizer: Any) -> Any:
        print(f"{Fore.GREEN}{input['text']}{Style.RESET_ALL}")
        if 'performance' in input:
            memory.set('performance', input['performance'])
        return 'next line of dialog'
