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
            "description": "concise scene description"
        }
    }
)


class CharacterCommand(PromptCommand):
    def __init__(self, client, model, name, description=None):
        c_schema = copy.copy(character_schema)
        c_schema.title = name
        self.name = name
        super().__init__(
            client=client,
            prompt=Prompt([SystemMessage("You are "+name+" in Macbeth.\n\nScene:\n{{$scene}}\n\nDialog:\n{{$dialog}}\n\nRespond with your next line of dialog formatted as '"+name+": <line of dialog>'.\n\n"+name+":")]),
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
        # Trim and combine dialog.
        response = response.replace('\n\n', '\n')
        response = ' '.join(line.strip() for line in response.split('\n'))

        if type(input) is dict and  'name' in input:
            name = input['name']
        else:
            name = self.name
        # Say line of dialog
        response = f"{name}: {response}"
        print(f"{Fore.GREEN}{response}{Style.RESET_ALL}");
        # Add line to dialog
        dialog = memory.get('dialog')
        if dialog is None:
            dialog = []
        dialog.append(response)
        memory.set('dialog',dialog)
        return response
