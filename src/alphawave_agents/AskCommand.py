from typing import Dict, Optional
from promptrix.promptrixTypes import PromptMemory, PromptFunctions, Tokenizer
from promptrix.Prompt import Prompt
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
from alphawave_agents.SchemaBasedCommand import CommandSchema as sbcCommandSchema
from alphawave_agents.agentTypes import TaskResponse
from dataclasses import dataclass, asdict

@dataclass
class CommandSchema(sbcCommandSchema):
    schema_type: str
    title: str
    description: str
    properties: Dict[str,Dict[str,str]]
    required: list[str]
    returns: str

schema = CommandSchema(
    schema_type="object",
    title= "ask",
    description= "ask the user a question and wait for their response",
    properties= {
        'question': {
            'type': "string",
            'description': "question to ask"
        }
    },
    required= ["question"],
    returns= "answer"
)


class AskCommandInput:
    def __init__(self, question: str):
        self.question = question

class AskCommand(SchemaBasedCommand):
    def __init__(self, title: Optional[str] = '', description: Optional[str] = ''):
        global schema
        super().__init__(schema, title, description)

    async def execute(self, input: AskCommandInput, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer) -> TaskResponse:
        return asdict(TaskResponse(status="input_needed",message=input['question']))
