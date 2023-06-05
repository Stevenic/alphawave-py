import requests
from typing import Optional, Dict, Any, Union
from promptrix import PromptFunctions, PromptMemory, PromptSection, Tokenizer
from types import PromptCompletionClient, PromptCompletionOptions, PromptResponse
from internals import ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionResponse, CreateCompletionRequest, CreateCompletionResponse
from internals import Colorize

class OpenAIClientOptions:
    def __init__(self, apiKey: str, organization: Optional[str] = None, endpoint: Optional[str] = None, logRequests: Optional[bool] = None):
        self.apiKey = apiKey
        self.organization = organization
        self.endpoint = endpoint
        self.logRequests = logRequests

class OpenAIClient(PromptCompletionClient):
    DefaultEndpoint = 'https://api.openai.com'
    UserAgent = 'AlphaWave'

    def __init__(self, options: OpenAIClientOptions):
        self.options = options

        if options.endpoint:
            options.endpoint = options.endpoint.strip()
            if options.endpoint.endswith('/'):
                options.endpoint = options.endpoint[:-1]

            if not options.endpoint.lower().startswith('https://'):
                raise ValueError(f"Client created with an invalid endpoint of '{options.endpoint}'. The endpoint must be a valid HTTPS url.")

        if not options.apiKey:
            raise ValueError("Client created without an 'apiKey'.")

        self._session = requests.Session()

    def complete_prompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        startTime = time.time()
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
                print(Colorize.output(response.json()))

            if response.status_code < 300:
                completion = response.json().get('choices')[0]
                return {'status': 'success', 'message': {'role': 'assistant', 'content': completion.get('text', '')}}
            elif response.status_code == 429:
                if self.options.logRequests:
                    print(Colorize.title('HEADERS:'))
                    print(Colorize.output(response.headers))
                return {'status': 'rate_limited', 'message': 'The text completion API returned a rate limit error.'}
            else:
                return {'status': 'error', 'message': f"The text completion API returned an error status of {response.status_code}: {response.reason}"}
        else:
            result = prompt.renderAsMessages(memory, functions, tokenizer, max_input_tokens)
            if result.tooLong:
                return {'status': 'too_long', 'message': f"The generated chat completion prompt had a length of {result.length} tokens which exceeded the max_input_tokens of {max_input_tokens}."}
            if self.options.logRequests:
                print(Colorize.title('CHAT PROMPT:'))
                print(Colorize.output(result.output))

            request = self.copyOptionsToRequest(CreateChatCompletionRequest({
                'model': options.model,
                'messages': result.output,
            }), options, ['max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user'])
            response = self.createChatCompletion(request)
            if self.options.logRequests:
                print(Colorize.title('CHAT RESPONSE:'))
                print(Colorize.value('statuse', response.status_code))
                print(Colorize.value('duration', time.time() - startTime, 'ms'))
                print(Colorize.output(response.json()))

            if response.status_code < 300:
                completion = response.json().get('choices')[0]
                return {'status': 'success', 'message': completion.get('message', {'role': 'assistant', 'content': ''})}
            elif response.status_code == 429:
                if self.options.logRequests:
                    print(Colorize.title('HEADERS:'))
                    print(Colorize.output(response.headers))
                return {'status': 'rate_limited', 'message': 'The chat completion API returned a rate limit error.'}
            else:
                return {'status': 'error', 'message': f"The chat completion API returned an error status of {response.status_code}: {response.reason}"}

    def addRequestHeaders(self, headers: Dict[str, str], options: OpenAIClientOptions):
        headers['Authorization'] = f"Bearer {options.apiKey}"
        if options.organization:
            headers['OpenAI-Organization'] = options.organization

    def copyOptionsToRequest(self, target: Dict[str, Any], src: Any, fields: list) -> Dict[str, Any]:
        for field in fields:
            if field in src:
                target[field] = src[field]
        return target

    def createCompletion(self, request: CreateCompletionRequest) -> requests.Response:
        url = f"{self.options.endpoint or self.DefaultEndpoint}/v1/completions"
        return self.post(url, request)

    def createChatCompletion(self, request: CreateChatCompletionRequest) -> requests.Response:
        url = f"{self.options.endpoint or self.DefaultEndpoint}/v1/chat/completions"
        return self.post(url, request)

    def post(self, url: str, body: object) -> requests.Response:
        requestHeaders = {
            'Content-Type': 'application/json',
            'User-Agent': self.UserAgent
        }
        self.addRequestHeaders(requestHeaders, self.options)
        return self._session.post(url, json=body, headers=requestHeaders)
