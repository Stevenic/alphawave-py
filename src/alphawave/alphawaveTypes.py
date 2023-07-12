from dataclasses import dataclass
from typing import Any, Optional, Union, List
import json

# Equivalent to TypeScript's import statement
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, PromptSection, Tokenizer

class PromptCompletionClient:
    def completePrompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: 'PromptCompletionOptions') -> 'Promise[PromptResponse]':
        pass

class PromptResponseValidator:
    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: 'PromptResponse', remaining_attempts: int) -> 'Promise[Validation]':
        pass

@dataclass
class PromptCompletionOptions:
    completion_type: str  = 'chat' # 'text' | 'chat'
    model: str = ''
    max_input_tokens: int =8000
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 500
    stop: str = None
    #presence_penalty: float = 1.0
    #frequency_penalty: float = 1.0
    #logit_bias = None
    #best_of = None

PromptResponseStatus = ['success', 'error', 'rate_limited', 'invalid_response', 'too_long']

@dataclass
class PromptResponse:
    status: str  # PromptResponseStatus
    message: Union[Message, str]

@dataclass
class Validation:
    type: str  # 'Validation'
    valid: bool
    feedback = None
    value = None
