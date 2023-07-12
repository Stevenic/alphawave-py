from typing import Callable, Any, Dict, Union
from dataclasses import dataclass, asdict, field
import asyncio
from promptrix.promptrixTypes import PromptMemory, PromptFunctions, Tokenizer
from promptrix.Utilities import Utilities
from promptrix.Prompt import Prompt
from alphawave.AlphaWave import AlphaWave
from alphawave.AlphaWave import AlphaWaveOptions
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.MemoryFork import MemoryFork
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand, CommandSchema as sbcCommandSchema
from alphawave_agents.agentTypes import TaskResponse

def update_dataclass(instance, **kwargs):
    for key, value in kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)

@dataclass
class CommandSchema(sbcCommandSchema):
    schema_type: str
    title: str
    description: str
    properties: Dict[str,Dict[str,str]]
    required: list[str] = field(default_factory=list)
    returns: str = None

@dataclass
class PromptCommandOptions(AlphaWaveOptions):
    def __init__(self, prompt_options: PromptCompletionOptions, schema: CommandSchema, parseResponse: Callable[[str, Dict[str, Any], PromptMemory, PromptFunctions, Tokenizer], Any] = None):
        super().__init__(prompt_options)
        self.prompt_options=prompt_options
        self.schema=schema
        self.parseResponse=parseResponse

class PromptCommand(SchemaBasedCommand):
    def __init__(self, client, prompt: Prompt, options: PromptCommandOptions, title: str = None, description: str = None):
        super().__init__(options.schema, title, description)
        self.options = options
        self.client = client
        self.prompt = prompt
        self.prompt_options = options.prompt_options

    def execute(self, input: Dict[str, Any], memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer) -> Union[TaskResponse, str]:
        # Fork memory and copy the input into the fork
        fork = MemoryFork(memory)
        if type(input) != dict:
            print (f'***** PromptCommand execute input not dict\n{input}\n')
        for key, value in input.items():
            fork.set(key, value)

        # Create a wave and send it
        options = AlphaWaveOptions()
        update_dataclass(options, **self.options.__dict__)
        update_dataclass(options, memory=fork, functions= functions, tokenizer= tokenizer)
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.options.prompt_options, memory=fork, functions=functions, tokenizer=tokenizer)
        response = asyncio.run(wave.completePrompt())
        # Process the response
        message = response['message']['content'] if isinstance(response['message'], dict) else response['message']
        if response['status'] == "success":
            # Return the response
            parsed = self.options.parseResponse(message, input, fork, functions, tokenizer) if self.options.parseResponse else message
            return Utilities.to_string(tokenizer, parsed)
        else:
            # Return the error
            return {
                "type": "TaskResponse",
                "status": response['status'],
                "message": message
            }

