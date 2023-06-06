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
    returns= "users answer"
)
print('AskCommand',asdict(schema))
