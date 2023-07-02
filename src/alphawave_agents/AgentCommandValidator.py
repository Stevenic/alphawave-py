from pyee import AsyncIOEventEmitter
from typing import Dict, Any, Union
from alphawave_agents.agentTypes import Command, AgentThought, AgentThoughtSchemaJSON, AgentThoughtSchemaTOML
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.Response import Response
from alphawave.TOMLResponseValidator import TOMLResponseValidator
import traceback

class AgentCommandValidator:
    def __init__(self, commands: dict[str, Command], model: str, syntax:str='JSON'):
        self._syntax = syntax
        if syntax == 'JSON':
            self._input_validator = JSONResponseValidator(AgentThoughtSchemaJSON, 'No valid JSON objects were found in the response. Return a valid JSON object with reasoning, plan, and the next command to use.')
        else:
            #print(f"***** AgentCommandValidator creating TOML")
            self._input_validator = TOMLResponseValidator(AgentThoughtSchemaTOML, 'No valid TOML found. Return valid TOML with reasoning, plan, and next command to use.')

        self._commands = commands

    async def validate_response(self, memory, functions, tokenizer, response, remaining_attempts) -> Union[AgentThought, None]:
        # Validate that the response contains a thought
        try:
          #print(f"***** AgentCommandValidator validate syntax: {self._syntax}")
          validation_result = self._input_validator.validate_response(memory, functions, tokenizer, response, remaining_attempts)
          if not validation_result['valid']:
              #if self._syntax == 'JSON':
              #    #print(f"***** AgentCommandValidator initial validation fail schema:\n{AgentThoughtSchemaJSON}\n{validation_result}")
              #else:
              #    #print(f"***** AgentCommandValidator initial validation fail schema:\n{AgentThoughtSchemaTOML}\n{validation_result}")
              return validation_result

          # Validate that the command exists
          thought = validation_result['value']
          #print(f'*****AgentCommandValidator post validate thought  \n{thought}\n')
          if not('command' in thought) or not('input' in thought['command']) or not ('name' in thought['command']):
              pass  # this should throw an error
          if not('command' in thought) or not('name' in thought['command']):
              pass  # this should throw an error

          command_name = thought['command']['input']['name'] if thought['command']['name'] == 'character' else thought['command']['name']
          #command_name = thought['command']['name']
          if command_name not in self._commands:
              #print(f"***** AgentCommandValidator no such command {command_name}")
              return {
                  'type': 'Validation',
                  'valid': False,
                  'feedback': 'The command '+thought['command']['name']+f' does not exist. The only commands you have are {list(self._commands.keys())}'
              }
          
          # Validate that the command input is valid
          command = self._commands[command_name]
          command_validation_result = await command.validate(thought['command']['inputs'] or {}, memory, functions, tokenizer, syntax = self._syntax)
          if not command_validation_result['valid']:
              print(f"***** AgentCommandValidator command input validation fail {command_name}\n{command_validation_result}")
              return command_validation_result
          
          # Return the validated thought
          #print(f"***** AgentCommandValidator validation success\n{command_validation_result['value']}")
          validation_result['value']['command']['inputs'] = command_validation_result['value']
          return validation_result
        except Exception as e:
            traceback.print_exc()
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f'The command validation failed. try again {str(e)}'
                }
        print(f"***** AgentCommandValidator generic exit fail {thought['command']['name']}")
        return {
            'type': 'Validation',
            'valid': False,
            'feedback': 'The command '+thought['command']['name']+' validation failed, try again'
        }
