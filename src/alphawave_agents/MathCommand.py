from typing import Any, Dict, Optional
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
from alphawave_agents.SchemaBasedCommand import CommandSchema as sbcCommandSchema
from alphawave_agents.agentTypes import TaskResponse
from dataclasses import dataclass, asdict
import traceback

@dataclass
class CommandSchema(sbcCommandSchema):
    schema_type: str
    title: str
    description: str
    properties: Dict[str,Dict[str,str]]
    required: list[str]
    returns: str

@dataclass
class MathCommandInput:
    code:str
"""
class TaskResponse:
    def __init__(self, response_type: str, status: str, message: str):
        self.type = response_type
        self.status = status
        self.message = message
"""
schema = CommandSchema(
    schema_type="object",
    title="math",
    description="execute some python code to calculate a value",
    properties={
        "code": {
            "type": "string",
            "description": "python expression to evaluate"
        }
    },
    required=["code"],
    returns="the calculated value"
)

class MathCommand(SchemaBasedCommand):

    def __init__(self, title = None, description = None):
        super().__init__(schema, title, description)

    def execute(self, input: MathCommandInput, memory: Any, functions: Any, tokenizer: Any) -> Any:
        try:
            return eval(input['code'])
        except Exception as err:
            message = str(err)
            return asdict(TaskResponse('TaskResponse', 'error', message))

