from typing import Callable, Any, Dict, Union
from promptrix import PromptMemory, PromptFunctions, Tokenizer, Utilities
from alphawave import AlphaWave, AlphaWaveOptions, MemoryFork
from SchemaBasedCommand import SchemaBasedCommand, CommandSchema
from types import TaskResponse

class PromptCommandOptions(AlphaWaveOptions):
    def __init__(self, schema: CommandSchema, parseResponse: Callable[[str, Dict[str, Any], PromptMemory, PromptFunctions, Tokenizer], Any] = None):
        super().__init__()
        self.schema = schema
        self.parseResponse = parseResponse

class PromptCommand(SchemaBasedCommand):
    def __init__(self, options: PromptCommandOptions, title: str = None, description: str = None):
        super().__init__(options.schema, title, description)
        self.options = options

    async def execute(self, input: Dict[str, Any], memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer) -> Union[TaskResponse, str]:
        # Fork memory and copy the input into the fork
        fork = MemoryFork(memory)
        for key, value in input.items():
            fork.set(key, value)

        # Create a wave and send it
        options = {**self.options.__dict__, "memory": fork, "functions": functions, "tokenizer": tokenizer}
        wave = AlphaWave(options)
        response = await wave.completePrompt()

        # Process the response
        message = response.message.content if isinstance(response.message, dict) else response.message
        if response.status == "success":
            # Return the response
            parsed = await self.options.parseResponse(message, input, memory, functions, tokenizer) if self.options.parseResponse else message
            return Utilities.toString(tokenizer, parsed)
        else:
            # Return the error
            return {
                "type": "TaskResponse",
                "status": response.status,
                "message": message
            }
