from typing import Union
from promptrix.promptrixTypes import PromptFunctions, PromptMemory, PromptSection, Tokenizer
from  promptrix.promptrixTypes import Message
from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse, PromptResponseStatus
from alphawave.Colorize import Colorize

class TestClient(PromptCompletionClient):
    def __init__(self, status: PromptResponseStatus = 'success', response: Union[str, Message] = {'role': 'assistant', 'content': "Hello World"}):
        self.status = status
        self.response = response

    async def completePrompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        result = await prompt.renderAsMessages(memory, functions, tokenizer, 500)
        if result.tooLong:
            return {'status': 'too_long', 'message': f"The generated chat completion prompt had a length of {result.length} tokens which exceeded the max_input_tokens of {max_input_tokens}."}
        print(Colorize.title('CHAT PROMPT:'))
        for msg in result.output:
            print(Colorize.output(json.dumps(msg, indent=2)))
        print(Colorize.title('CHAT RESPONSE:'))
        print(Colorize.output(self.response))
        return {'status': self.status, 'message': self.response}
