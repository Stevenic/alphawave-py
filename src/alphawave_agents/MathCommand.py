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

schema = CommandSchema(
    schema_type="object",
    title="math",
    description="evaluate a mathematical expression to calculate its value",
    properties={
        "code": {
            "type": "string",
            "description": "expression to evaluate"
        }
    },
    required=["code"],
    returns="the calculated value"
)

class MathCommand(SchemaBasedCommand):

    def __init__(self, title = None, description = None):
        super().__init__(schema, title, description)

    async def execute(self, input: MathCommandInput, memory: Any, functions: Any, tokenizer: Any) -> Any:
        try:
            exp = input['code']
            exp = exp.replace('\\*', '*')
            exp = exp.replace('\\+', '+')
            exp = exp.replace('\\-', '-')
            exp = exp.replace('\\/', '/')
            return eval(exp)
        except Exception as err:
            message = str(err)
            return asdict(TaskResponse('TaskResponse', 'error', message))

