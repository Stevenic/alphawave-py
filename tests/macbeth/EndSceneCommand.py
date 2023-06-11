from typing import Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from promptrix.promptrixTypes import PromptMemory, PromptFunctions, Tokenizer
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand, CommandSchema as sbcCommandSchema

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
    title="endScene",
    description="marks the end of a scene and lets the narrator ask the user for next scene",
    properties={
        "question": {
            "type": "string",
            "description": "question for user"
        }
    },
    required=["question"],
    returns="users next scene request"
)

class EndSceneCommandInput:
    def __init__(self, question: str):
        self.question = question

class EndSceneCommand(SchemaBasedCommand):
    def __init__(self, title=schema.title, description=schema.description):
        super().__init__(schema, title, description)


    def execute(self, input: EndSceneCommandInput, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer):
        # Delete the dialog for the current scene
        memory.set('dialog', [])
        memory.set('performance', None)

        # Prompt user with questions
        return {'status':'success', 'message':input['question']}
