from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from pyee import AsyncIOEventEmitter
from jsonschema import validate
from promptrix.promptrixTypes import PromptMemory, PromptFunctions, Tokenizer

class Command(ABC):
    title: str
    description: str
    inputs: Optional[str]
    output: Optional[str]

    @abstractmethod
    async def execute(self, input: Dict[str, Any], memory: 'PromptMemory', functions: 'PromptFunctions', tokenizer: 'Tokenizer') -> Any:
        pass

    @abstractmethod
    async def validate(self, input: Dict[str, Any], memory: 'PromptMemory', functions: 'PromptFunctions', tokenizer: 'Tokenizer') -> 'Validation':
        pass

TaskResponseStatus = Union['PromptResponseStatus', 'input_needed', 'too_many_steps']

@dataclass
class TaskResponse:
    type: str = 'TaskResponse'
    status: TaskResponseStatus = 'PromptResponseStatus'
    message: str = ''

class AgentThought:
    thoughts: Dict[str, str]
    command: Dict[str, Any]

AgentThoughtSchema: Dict[str,str] = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "reasoning": {"type": "string"},
                "plan": {"type": "string"}
            },
            "required": ["thought", "reasoning", "plan"]
        },
        "command": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "input": {"type": "object"}
            },
            "required": ["name"]
        }
    },
    "required": ["thoughts", "command"]
}

AgentThoughtSchema_py: Dict[str,str] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "plan": {"type": "string"},
        "command": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "input": {"type": "object"}
                },
            "required": ["name"]
        }
    },
    "required": ["reasoning", "plan", "command"]
}
