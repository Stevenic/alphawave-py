from typing import Dict, Optional
from abc import ABC, abstractmethod

class CommandSchema:
    def __init__(self, type: str, title: str, description: str, properties: Dict, required: list):
        self.type = type
        self.title = title
        self.description = description
        self.properties = properties
        self.required = required

class SchemaBasedCommand(ABC):
    def __init__(self, schema: CommandSchema, title: Optional[str] = None, description: Optional[str] = None):
        self.schema = schema
        self.title = title
        self.description = description

    @abstractmethod
    def execute(self, input, memory, functions, tokenizer):
        pass

class CompleteTaskCommandInput:
    def __init__(self, status: str):
        self.status = status

class TaskResponse:
    def __init__(self, type: str, status: str, message: str):
        self.type = type
        self.status = status
        self.message = message

class CompleteTaskCommand(SchemaBasedCommand):
    def __init__(self, response: Optiona[str] = None, title: Optional[str] = None, description: Optional[str] = None):
        schema = CommandSchema(
            type="object",
            title="completeTask",
            description="signals that the task is completed",
            properties={
                "status": {
                    "type": "string",
                    "description": "brief completion status"
                }
            },
            required=["status"]
        )
        super().__init__(schema, title, description)

    def execute(self, input: CompleteTaskCommandInput, memory, functions, tokenizer):
        rsp = input.status
        if self.response is not None:
            rsp = self.response
        return asdict(TaskResponse(
            type="TaskResponse",
            status="success",
            message=rsp
        ))
