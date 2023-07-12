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
from alphawave.Colorize import Colorize
import alphawave_pyexts.utilityV2 as ut
import alphawave_pyexts.LLMClient as client

@dataclass
class OSClientOptions(PromptCompletionOptions):
    def __init__(self, apiKey, model='wizardLM', organization = None, endpoint ='127.0.0.1', port=5004,logRequests =False):
        self.apiKey = apiKey
        self.organization = organization
        self.endpoint = endpoint
        self.port = port
        self.logRequests = logRequests
        self.model = model
        
def update_dataclass(instance, **kwargs):
    for key, value in kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance

def get_values(instance, keys):
    values = []
    for key in keys:
        if hasattr(instance, key):
            values.append(getattr(instance, key))
        else:
            values.append(None)
    return values

def display_options(options):
    options_dict = options.__dict__
    print()
    for item in options_dict.keys():
        print(item, options_dict[item])
    print()
    
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
        self.options = OSClientOptions(apiKey=None, organization=None, endpoint= '127.0.0.1', port = 5004, logRequests=False)
        update_dataclass(self.options, **kwargs)
        if self.options.endpoint:
            self.options.endpoint = self.options.endpoint.strip()
            if self.options.endpoint.endswith('/'):
                self.options.endpoint = self.options.endpoint[:-1]

        self._session = requests.Session()

    async def completePrompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        if isinstance(options, dict):
            argoptions = options
            options = PromptCompletionOptions(completion_type = argoptions['completion_type'],
                                              model = argoptions['model'],
                                              max_input_tokens = argoptions['max_input_tokens'],
                                              temperature = argoptions['temperature'],
                                              top_p = argoptions['top_p'],
                                              max_tokens = argoptions['max_tokens'],
                                              stop = argoptions['stop'],
                                              presence_penalty = argoptions['presence_penalty'],
                                              frequency_penalty = argoptions['frequency_penalty']
                                              )
        startTime = time.time()
        max_input_tokens = 1500
        if hasattr(options, 'max_input_tokens') and getattr(options, 'max_input_tokens') is not None:
            max_input_tokens = options.max_input_tokens
        
        result = prompt.renderAsMessages(memory, functions, tokenizer, max_input_tokens)
        
        if result.tooLong:
            return {'status': 'too_long', 'message': f"The generated chat completion prompt had a length of {result.length} tokens which exceeded the max_input_tokens of {max_input_tokens}."}
        if self.options.logRequests:
            print(Colorize.title('CHAT PROMPT:'))
            for msg in result.output:
                if not isinstance(msg, dict):
                    print(Colorize.output(msg))
                    msg = msg.__dict__
                print(Colorize.output(json.dumps(msg, indent=2)), end='')
            print()
        request = self.copyOptionsToRequest(CreateChatCompletionRequest(model = options.model, messages =  result.output), options, ['max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user'])
        response = await self.createChatCompletion(request)
        if self.options.logRequests:
            print(Colorize.title('CHAT RESPONSE:'))
            print(Colorize.value('status', response.status))
            print(Colorize.value('duration', time.time() - startTime, 'ms'))
            print(Colorize.output(response.message))
        
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

    async def createCompletion(self, request: CreateCompletionRequest) -> requests.Response:
        url = f"{self.options.endpoint or self.DefaultEndpoint}/v1/completions"
        return await self.post(url, request)

    async def createChatCompletion(self, request: CreateChatCompletionRequest) -> requests.Response:
        url = f"{self.options.endpoint or self.DefaultEndpoint}/v1/chat/completions"
        return await self.post(url, request)

    async def post(self, url: str, request: object) -> requests.Response:
        requestHeaders = {
            'Content-Type': 'application/json',
            'User-Agent': self.UserAgent
        }
        result = ''
        try:
            result = await ut.ask_LLM(request.model,
                                request.messages,
                                request.max_tokens,
                                request.temperature,
                                request.top_p,
                                self.options.endpoint,
                                self.options.port
                                )
        except Exception as e:
            return PromptResponse(status='error',message=str(e))
        return PromptResponse(status='success', message=result)
