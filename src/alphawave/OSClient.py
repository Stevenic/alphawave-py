import requests
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union
import time
import json
import asyncio
from promptrix.promptrixTypes import PromptFunctions, PromptMemory, PromptSection, Tokenizer
from promptrix.SystemMessage import SystemMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.AssistantMessage import AssistantMessage

from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse
from alphawave.internalTypes import ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionResponse, CreateCompletionRequest, CreateCompletionResponse
import alphawave.Colorize as Colorize
import alphawave.utilityV2 as ut
import alphawave.LLMClient as client

@dataclass
class OSClientOptions:
    def __init__(self, apiKey, organization = None, endpoint = None, logRequests = None):
        self.apiKey = apiKey
        self.organization = organization
        self.endpoint = endpoint
        self.logRequests = logRequests

@dataclass
class Response:
    status_code: int
    text: str
    headers: Dict[str,str] = None
    reason: str = ''
    
class OSClient(PromptCompletionClient):
    DefaultEndpoint = 'https://api.openai.com'
    UserAgent = 'AlphaWave'

    def __init__(self, **kwargs):
        self.options = {'apiKey':None, 'organization':None, 'endpoint':None, 'logRequests':False}
        self.options.update(kwargs)
        if self.options['endpoint']:
            self.options['endpoint'] = self.options['endpoint'].strip()
            if self.options['endpoint'].endswith('/'):
                self.options['endpoint'] = self.options['endpoint'][:-1]

            if not self.options['endpoint'].lower().startswith('https://'):
                raise ValueError(f"Client created with an invalid endpoint of '{options['endpoint']}'. The endpoint must be a valid HTTPS url.")

        if not self.options['apiKey']:
            print("Client created without an apiKey.")

        self._session = requests.Session()

    async def complete_prompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        startTime = time.time()
        #print('enter complete prompt')
        max_input_tokens = options.max_input_tokens or 1024
        if options.completion_type == 'text':
            result = prompt.renderAsText(memory, functions, tokenizer, max_input_tokens)
            if result.tooLong:
                return {'status': 'too_long', 'message': f"The generated text completion prompt had a length of {result.length} tokens which exceeded the max_input_tokens of {max_input_tokens}."}
            if self.options.logRequests:
                print(Colorize.title('PROMPT:'))
                print(Colorize.output(result.output))

            request = self.copyOptionsToRequest(CreateCompletionRequest({
                'model': options.model,
                'prompt': result.output,
            }), options, ['max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user'])
            response = self.createCompletion(request)
            if self.options.logRequests:
                print(Colorize.title('RESPONSE:'))
                print(Colorize.value('statuse', response.status))
                print(Colorize.value('duration', time.time() - startTime, 'ms'))
                print(Colorize.output(response.message))

            if response.status == 'success':
                completion = response.message
                return {'status': 'success', 'message': completion}
            else:
                return {'status': 'error', 'message': f"The text completion API returned an error status of {response.status}: {response.message}"}
        else:
            result = await prompt.renderAsMessages(memory, functions, tokenizer, max_input_tokens)
            if result.tooLong:
                return {'status': 'too_long', 'message': f"The generated chat completion prompt had a length of {result.length} tokens which exceeded the max_input_tokens of {max_input_tokens}."}
            if self.options['logRequests']:
                print(Colorize.title('CHAT PROMPT:'))
                print(Colorize.output(result.output))
            #print(f'************* render as messages {result}')
            request = self.copyOptionsToRequest(CreateChatCompletionRequest(model = options.model, messages =  result.output), options, ['max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user'])
            response = await self.createChatCompletion(request)
            if self.options['logRequests']:
                print(Colorize.title('CHAT RESPONSE:'))
                print(Colorize.value('status', response.status))
                print(Colorize.value('duration', time.time() - startTime, 'ms'))
                print(Colorize.output(response.text))

            if response.status == 'success':
                completion = response.message
                return {'status': 'success', 'message': completion}
            else:
                return {'status': 'error', 'message': f"The chat completion API returned an error status of {response.status}: {response.message}"}

    def addRequestHeaders(self, headers: Dict[str, str], options: OSClientOptions):
        headers['Authorization'] = f"Bearer {options.apiKey}"
        if options.organization:
            headers['OS-Organization'] = options.organization

    def copyOptionsToRequest(self, target: Dict[str, Any], src: Any, fields: list) -> Dict[str, Any]:
        for field in fields:
            if hasattr(src, field) and getattr(src, field) is not None:
                setattr(target,field, getattr(src,field))
        return target

    def createCompletion(self, request: CreateCompletionRequest) -> requests.Response:
        url = f"{self.options.endpoint or self.DefaultEndpoint}/v1/completions"
        return self.post(url, request)

    def createChatCompletion(self, request: CreateChatCompletionRequest) -> requests.Response:
        url = f"{self.options['endpoint'] or self.DefaultEndpoint}/v1/chat/completions"
        return self.post(url, request)

    async def post(self, url: str, body: object) -> requests.Response:
        requestHeaders = {
            'Content-Type': 'application/json',
            'User-Agent': self.UserAgent
        }
        #print(f'***** OSClient sending {body.messages}')
        result = ''
        try:
            result = ut.ask_LLM(ut.MODEL, body.messages)
            runon_idx = result.find(client.USER)
            if runon_idx > 0:
                result = result[:runon_idx]
        except Exception as e:
            print(f'***** OSCLient model returned {result}')
            return PromptResponse(status='error',message=str(e))
        return PromptResponse(status='success', message = {'role':'assistant', 'content': result})
