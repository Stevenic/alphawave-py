from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict

@dataclass
class CreateCompletionRequest:
    model: str
    prompt: Optional[Union[List[Any], List[int], List[str], str]] = None
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    stop: Optional[Union[List[str], str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    logit_bias: Optional[Dict] = None
    user: Optional[str] = None

@dataclass
class CreateCompletionResponseChoicesInnerLogprobs:
    tokens: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[Dict]] = None
    text_offset: Optional[List[int]] = None

@dataclass
class CreateCompletionResponseChoicesInner:
    text: Optional[str] = None
    index: Optional[int] = None
    logprobs: Optional[CreateCompletionResponseChoicesInnerLogprobs] = None
    finish_reason: Optional[str] = None

@dataclass
class CreateCompletionResponseUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class CreateCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[CreateCompletionResponseChoicesInner]
    usage: Optional[CreateCompletionResponseUsage] = None

@dataclass
class ChatCompletionRequestMessage:
    role: str
    content: str
    name: Optional[str] = None

@dataclass
class CreateChatCompletionRequest:
    model: str
    messages: List[ChatCompletionRequestMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[List[str], str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict] = None
    user: Optional[str] = None

@dataclass
class CreateChatCompletionResponseChoicesInner:
    index: Optional[int] = None
    message: Optional[ChatCompletionRequestMessage] = None
    finish_reason: Optional[str] = None

@dataclass
class CreateChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[CreateChatCompletionResponseChoicesInner]
    usage: Optional[CreateCompletionResponseUsage] = None

@dataclass
class ChatCompletionResponseMessage:
    role: str
    content: str

@dataclass
class CreateModerationRequest:
    input: Union[List[str], str]
    model: Optional[str] = None

@dataclass
class CreateModerationResponseResultsInnerCategories:
    hate: bool
    hate_threatening: bool
    self_harm: bool
    sexual: bool
    sexual_minors: bool
    violence: bool
    violence_graphic: bool

@dataclass
class CreateModerationResponseResultsInnerCategoryScores:
    hate: float
    hate_threatening: float
    self_harm: float
    sexual: float
    sexual_minors: float
    violence: float
    violence_graphic: float

@dataclass
class CreateModerationResponseResultsInner:
    flagged: bool
    categories: CreateModerationResponseResultsInnerCategories
    category_scores: CreateModerationResponseResultsInnerCategoryScores

@dataclass
class CreateModerationResponse:
    id: str
    model: str
    results: List[CreateModerationResponseResultsInner]

@dataclass
class CreateEmbeddingRequest:
    model: str
    input: Union[List[Any], List[int], List[str], str]
    user: Optional[str] = None

@dataclass
class CreateEmbeddingResponseDataInner:
    index: int
    object: str
    embedding: List[float]

@dataclass
class CreateEmbeddingResponseUsage:
    prompt_tokens: int
    total_tokens: int

@dataclass
class CreateEmbeddingResponse:
    object: str
    model: str
    data: List[CreateEmbeddingResponseDataInner]
    usage: CreateEmbeddingResponseUsage

if __name__ == '__main__':
    chatCompletionObj = CreateChatCompletionRequest(model= 'gpt-3.5', messages= ['a', 'b'])
