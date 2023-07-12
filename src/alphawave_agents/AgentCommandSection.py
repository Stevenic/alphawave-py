from pyee import AsyncIOEventEmitter
from typing import Dict, Any, List, Optional
from promptrix.promptrixTypes import PromptFunctions, PromptMemory, Tokenizer, Message, RenderedPromptSection
from promptrix.PromptSectionBase import PromptSectionBase
from alphawave_agents.agentTypes import Command
import traceback

class AgentCommandSection(PromptSectionBase):
    def __init__(self, commands: Dict[str, Command], tokens: int = -1, required: bool = True, one_shot=False, syntax = 'JSON'):
        super().__init__(tokens, required)
        self._commands = commands
        self.tokens = tokens
        self.required = required
        self.one_shot = one_shot
        self.syntax = syntax
        
    def renderAsMessages(self, memory: Any, functions: Any, tokenizer: Any, max_tokens: int) -> Dict[str, Any]:
        # Render commands to message content
        content = 'You have the following  commands available in addition to known facts and reasoning:\n'
        for command in self._commands.values():
            content += f'\t{command.title}:\n'
            content += f'\t\tuse: {command.description}\n'
            #if command.inputs:
            #    content += f'\t\tinputs: {command.inputs}\n'
            #if command.output:
            #    content += f'\t\toutput: {command.output}\n'
            #if self.one_shot:
            content += command.one_shot(self.syntax)
        # Return as system message
        result = None
        try:
            length = len(tokenizer.encode(content))
            result = self.return_messages([{'role': 'system', 'content': content}], length, tokenizer, max_tokens)
        except Exception as e:
            traceback.print_exc()
        return result

