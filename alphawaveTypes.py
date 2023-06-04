from dataclasses import dataclass
from typing import Any, Optional, Union, List

# Equivalent to TypeScript's import statement
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, PromptSection, Tokenizer

class PromptCompletionClient:
    def complete_prompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: 'PromptCompletionOptions') -> 'Promise[PromptResponse]':
        pass

class PromptResponseValidator:
    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: 'PromptResponse', remaining_attempts: int) -> 'Promise[Validation]':
        pass

@dataclass
class PromptCompletionOptions:
    completion_type: str  # 'text' | 'chat'
    model: str
    max_input_tokens = None
    temperature = None
    top_p = None
    max_tokens = None
    stop = None
    presence_penalty = None
    frequency_penalty = None
    logit_bias = None
    best_of = None

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
