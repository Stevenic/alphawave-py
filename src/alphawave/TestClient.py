from typing import Union
from promptrix.promptrixTypes import PromptFunctions, PromptMemory, PromptSection, Tokenizer
from  promptrix.promptrixTypes import Message
from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse, PromptResponseStatus
import json

class TestClient(PromptCompletionClient):
    def __init__(self, status: PromptResponseStatus = 'success', response: Union[str, Message] = {'role': 'assistant', 'content': "Hello World"}):
        self.status = status
        self.response = response

    async def completePrompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        return {'status': self.status, 'message': self.response}
