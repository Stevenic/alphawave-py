from pyee import AsyncIOEventEmitter
from typing import Dict, Any, List, Optional
from promptrix.promptrixTypes import PromptFunctions, PromptMemory, Tokenizer, Message, RenderedPromptSection
from promptrix.PromptSectionBase import PromptSectionBase
from alphawave_agents.agentTypes import Command


class AgentCommandSection(PromptSectionBase):
    def __init__(self, commands: Dict[str, Command], tokens: int = -1, required: bool = True):
        super().__init__(tokens, required)
        self._commands = commands
        self.tokens = tokens
        self.required = required

    async def renderAsMessages(self, memory: Any, functions: Any, tokenizer: Any, max_tokens: int) -> Dict[str, Any]:
        # Render commands to message content
        content = 'commands:\n'
        for command in self._commands.values():
            content += f'\t{command.title}:\n'
            content += f'\t\tuse: {command.description}\n'
            if command.inputs:
                content += f'\t\tinputs: {command.inputs}\n'
            if command.output:
                content += f'\t\toutput: {command.output}\n'

        # Return as system message
        length = len(tokenizer.encode(content))
        result = self.return_messages([{'role': 'system', 'content': content}], length, tokenizer, max_tokens)
        return result

