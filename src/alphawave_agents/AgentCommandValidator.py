from pyee import AsyncIOEventEmitter
from typing import Dict, Any, Union
from alphawave_agents.agentTypes import Command, AgentThought, AgentThoughtSchema, AgentThoughtSchema_py
from alphawave.JSONResponseValidator import JSONResponseValidator
import traceback

class AgentCommandValidator:
    def __init__(self, commands: dict[str, Command], model: str):
        if model.lower().startswith('gpt'):
            self._thought_validator = JSONResponseValidator(AgentThoughtSchema, 'No valid JSON objects were found in the response. Return a valid JSON object with your thoughts and the next command to perform.')
        else:
            self._thought_validator = JSONResponseValidator(AgentThoughtSchema_py, 'No valid JSON objects were found in the response. Return a valid JSON object with your thoughts and the next command to perform.')

        self._commands = commands

    async def validate_response(self, memory, functions, tokenizer, response, remaining_attempts) -> Union[AgentThought, None]:
        # Validate that the response contains a thought
        try:
          validation_result = self._thought_validator.validate_response(memory, functions, tokenizer, response, remaining_attempts)
          if not validation_result['valid']:
              return validation_result

          # Validate that the command exists
          thought = validation_result['value']
          if not('command' in thought) or not('input' in thought['command']) or not ('name' in thought['command']):
              pass
          if not('command' in thought) or not('name' in thought['command']):
              pass

          command_name = thought['command']['input']['name'] if thought['command']['name'] == 'character' else thought['command']['name']
          #command_name = thought['command']['name']
          if command_name not in self._commands:
              return {
                  'type': 'Validation',
                  'valid': False,
                  'feedback': 'The command '+thought['command']['name']+f' does not exist. The only commands you have are {list(self._commands.keys())}'
              }
          
          # Validate that the command input is valid
          command = self._commands[command_name]
          command_validation_result = await command.validate(thought['command']['input'] or {}, memory, functions, tokenizer)
          if not command_validation_result['valid']:
              return command_validation_result
          
          # Return the validated thought
          return validation_result
        except Exception as e:
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f'The command validation failed. try again {str(e)}'
                }
        return {
            'type': 'Validation',
            'valid': False,
            'feedback': 'The command '+thought['command']['name']+' validation failed, try again'
        }
