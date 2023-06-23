from typing import Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from promptrix.promptrixTypes import PromptMemory, Tokenizer, PromptFunctions
from alphawave_agents.agentTypes import TaskResponse
from alphawave_agents.SchemaBasedCommand import CommandSchema as sbcCommandSchema
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand


@dataclass
class CommandSchema(sbcCommandSchema):
    schema_type: str = "object"
    title: str = "finalAnswer"
    description: str = "show answer to the user"
    properties: dict = field(default_factory=lambda: {
        "answer": {
            "type": "string",
            "description": "final answer"
        }
    })
    required: list[str] = field(default_factory=lambda: ["answer"])
    returns: str = "a followup task or question"

@dataclass
class FinalAnswerCommandInput:
    answer:str

class FinalAnswerCommand(SchemaBasedCommand):
    def __init__(self, title: Optional[str] = None, description: Optional[str] = None):
        super().__init__(CommandSchema(), title, description)

    def execute(self, input: FinalAnswerCommandInput, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer) -> TaskResponse:
        print(f'\nAnswer: \n{input}\n')
        return {
            "type": "TaskResponse",
            "status": "input_needed",
            "message": "input['answer']"
        }
