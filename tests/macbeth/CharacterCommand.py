from typing import Any, Dict, List
from colorama import Fore, Style
from dataclasses import dataclass, asdict
import copy
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
#from alphawave_agents.SchemaBasedCommand import CommandSchema as sbcCommandSchema
from alphawave_agents.PromptCommand import CommandSchema
from alphawave_agents.PromptCommand import PromptCommand, PromptCommandOptions
from alphawave.alphawaveTypes import PromptCompletionOptions
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage

"""
@dataclass
class CharacterCommandSchema():
    name: str
    scene: str
    schema_type: str
    title: str
    description: str
    properties: Dict[str,Dict[str,str]]
    required=["name", "scene"]
    returns="characters line of dialog"
"""

character_schema = CommandSchema(
    schema_type="object",
    title="extra",
    description="a character in the play Macbeth",
    properties={
        "name": {
            "type": "string",
            "description": "character name"
        },
        "scene": {
            "type": "string",
            "description": "scene description no more than 80 words"
        }
    }
)


class CharacterCommand(PromptCommand):
    def __init__(self, client, model, name, description=None):
        c_schema = copy.copy(character_schema)
        c_schema.title = name
        super().__init__(
            client=client,
            prompt=Prompt([SystemMessage("You are the character of {{$name}} from Macbeth.\n\nScene:\n{{$scene}}\n\nDialog:\n{{$dialog}}\n\nRespond with your next line of dialog formatted as '{{$name}}: <line of dialog>'.\n\n{{$name}}:")]),
            options = PromptCommandOptions(prompt_options=PromptCompletionOptions(                                        
                completion_type = 'chat' if model.startswith('gpt') else 'text',
                model = model,
                temperature = 0.4,
                max_input_tokens = 2500,
                max_tokens = 200),
                                           parseResponse = self.parse_response,
                                           schema = c_schema
                                           )
            )

    @staticmethod
    async def parse_response(response: str, input: Dict[str, Any], memory: Dict[str, Any], extraArg1, extraArg2) -> str:
        #print(f' CharacterCommand parse_response args response {response}')
        #print(f' CharacterCommand parse_response args input {input}')
        #print(f' CharacterCommand parse_response args memory {memory}')
        #print(f' CharacterCommand parse_response args ea1 {extraArg1}')
        #print(f' CharacterCommand parse_response args ea2 {extraArg2}')
        # Trim and combine dialog.
        response = response.replace('\n\n', '\n')
        response = ' '.join(line.strip() for line in response.split('\n'))

        # Say line of dialog
        print(f"{memory.get('name')}: {Fore.LIGHTBLACK_EX}{response}{Style.RESET_ALL}")

        # Add line to dialog
        response = f"{input['name']}: {response}"
        dialog = memory.get('dialog')
        dialog.append(response)
        memory['dialog'] = dialog
        return response
