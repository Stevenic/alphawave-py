import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json
from promptrix.promptrixTypes import PromptFunctions, PromptMemory, PromptSection, Tokenizer
from promptrix.SystemMessage import SystemMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.AssistantMessage import AssistantMessage

from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse
from alphawave.internalTypes import ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionResponse, CreateCompletionRequest, CreateCompletionResponse
from alphawave.Colorize import Colorize

@dataclass
class OpenAIClientOptions:
    def __init__(self, apiKey=None, organization = None, endpoint = None, logRequests = False):
        self.apiKey = apiKey
        self.organization = organization
        self.endpoint = endpoint
        self.logRequests = logRequests

def update_dataclass(instance, **kwargs):
    for key, value in kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)

class OpenAIClient(PromptCompletionClient):
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
            print("Client created without an 'apiKey'.")
            raise ValueError

        self._session = requests.Session()

    async def completePrompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        startTime = time.time()
        max_input_tokens = 1024
        if isinstance(options, dict):
            argoptions = options
            options = PromptCompletionOptions(completion_type = argoptions['completion_type'], model = argoptions['model'])
            update_dataclass(options, **argoptions)
        if hasattr(options, 'max_input_tokens') and getattr(options, 'max_input_tokens') is not None:
            max_input_tokens = options.max_input_tokens
        if hasattr(options, 'completion_type') and options.completion_type == 'text':
            result = prompt.renderAsText(memory, functions, tokenizer, max_input_tokens)
            if result.tooLong:
                return {'status': 'too_long', 'message': f"The generated text completion prompt had a length of {result.length} tokens which exceeded the max_input_tokens of {max_input_tokens}."}
            if self.options['logRequests']:
                print(Colorize.title('PROMPT:'))
                for msg in result.output:
                    if not isinstance(msg, dict):
                        msg = msg.__dict__
                    print(Colorize.output(json.dumps(msg, indent=2)), end='')
                print()

            request = self.copyOptionsToRequest(CreateCompletionRequest({
                'model': self.options['model'],
                'prompt': result['output'],
            }), options, ['max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user'])
            response = self.createCompletion(request)
            if self.options['logRequests']:
                print(Colorize.title('RESPONSE:'))
                print(Colorize.value('status', response.status))
                print(Colorize.value('duration', time.time() - startTime, 'ms'))
                print(Colorize.output(response.json()))

            if response.status_code < 300:
                completion = response.json().get('choices')[0]
                return {'status': 'success', 'message': {'role': 'assistant', 'content': completion.get('text', '')}}
            elif response.status_code == 429:
                if self.options['logRequests']:
                    print(Colorize.title('HEADERS:'))
                    print(Colorize.output(response.headers))
                return {'status': 'rate_limited', 'message': 'The text completion API returned a rate limit error.'}
            else:
                return {'status': 'error', 'message': f"The text completion API returned an error status of {response.status_code}: {response.reason}"}
        else:
            result = await prompt.renderAsMessages(memory, functions, tokenizer, max_input_tokens)
            if result.tooLong:
                return {'status': 'too_long', 'message': f"The generated chat completion prompt had a length of {result.length} tokens which exceeded the max_input_tokens of {max_input_tokens}."}
            if self.options['logRequests']:
                print(Colorize.title('CHAT PROMPT:'))
                for msg in result.output:
                    if not isinstance(msg, dict):
                        msg = msg.__dict__
                    print(Colorize.output(json.dumps(msg, indent=2)), end='')
                print()

            request = self.copyOptionsToRequest(CreateChatCompletionRequest(model=options.model, messages=result.output), options,
                                                    ['max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user'])
            response = self.createChatCompletion(request)
            if self.options['logRequests']:
                print(Colorize.title('CHAT RESPONSE:'))
                print(Colorize.value('status', response.status_code))
                print(Colorize.value('duration', time.time() - startTime, 'ms'))
                print(Colorize.output(response.json()))

            if response.status_code < 300:
                completion = response.json().get('choices')[0]
                return {'status': 'success', 'message': completion.get('message', {'role': 'assistant', 'content': ''})}
            elif response.status_code == 429:
                if self.options['logRequests']:
                    print(Colorize.title('HEADERS:'))
                    print(Colorize.output(response.headers))
                return {'status': 'rate_limited', 'message': 'The chat completion API returned a rate limit error.'}
            elif response.status_code == 503:
                if self.options['logRequests']:
                    print(Colorize.title('HEADERS:'))
                    print(Colorize.output(response.headers))
                return {'status': 'server unavailable', 'message': 'The chat completion API returned server unavailable or overloaded.'}
            else:
                return {'status': 'error', 'message': f"The chat completion API returned an error status of {response.status_code}: {response.reason}"}

    def addRequestHeaders(self, headers: Dict[str, str], options: OpenAIClientOptions):
        headers['Authorization'] = f"Bearer {options['apiKey']}"
        if options['organization']:
            headers['OpenAI-Organization'] = options['organization']

    def copyOptionsToRequest(self, target: Dict[str, Any], src: Any, fields: list) -> Dict[str, Any]:
        for field in fields:
            if hasattr(src, field) and getattr(src, field) is not None:
                setattr(target,field, getattr(src,field))
        return target

    def createCompletion(self, request: CreateCompletionRequest) -> requests.Response:
        url = f"{self.options['endpoint'] or self.DefaultEndpoint}/v1/completions"
        return self.post(url, request)

    def createChatCompletion(self, request: CreateChatCompletionRequest) -> requests.Response:
        url = f"{self.options['endpoint'] or self.DefaultEndpoint}/v1/chat/completions"
        return self.post(url, request)

    def post(self, url: str, body: object) -> requests.Response:
        requestHeaders = {
            'Content-Type': 'application/json',
            'User-Agent': self.UserAgent
        }
        self.addRequestHeaders(requestHeaders, self.options)
        jsonbody = asdict(body)
        keys = list(jsonbody.keys())
        for key in keys:
            if jsonbody[key] is None:
                del jsonbody[key]
        result = self._session.post(url, json=jsonbody, headers=requestHeaders)
        if result.status_code < 300:
            completion = result.json().get('choices')[0]
        return result
        
