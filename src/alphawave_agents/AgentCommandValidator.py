from pyee import AsyncIOEventEmitter
from typing import Dict, Any, Union
from alphawave_agents.agentTypes import Command, AgentThought, AgentThoughtSchema
from alphawave.JSONResponseValidator import JSONResponseValidator
import traceback

class AgentCommandValidator:
    def __init__(self, commands: dict[str, Command]):
        self._thought_validator = JSONResponseValidator(AgentThoughtSchema, 'No valid JSON objects were found in the response. Return a valid JSON object with your thoughts and the next command to perform.')
        self._commands = commands

    async def validate_response(self, memory, functions, tokenizer, response, remaining_attempts) -> Union[AgentThought, None]:
        # Validate that the response contains a thought
        try:
          #print(f'AgentCommandValidator calling self._thought_validator')
          validation_result = self._thought_validator.validate_response(memory, functions, tokenizer, response, remaining_attempts)
          if not validation_result['valid']:
              print(f'AgentCommandValidator self._thought_validator failed {validation_result}')
              return validation_result

          # Validate that the command exists
          thought = validation_result['value']
          #print(f'AgentCommandValidator thought {thought}\n{self._commands.keys()}')
          #print(f"AgentCommandValidator command name {thought['command']['name']}")
          command_name = thought['command']['name']
          if thought['command']['name'] not in self._commands:
              print(f"AgentCommandValidator command name not found {self._commands.keys()}")
              return {
                  'type': 'Validation',
                  'valid': False,
                  'feedback': 'The command '+thought['command']['name']+f' does not exist. The only commands you have are {self._commands.keys()}'
              }
          
          # Validate that the command input is valid
          command_name = thought['command']['input']['name'] if thought['command']['name'] == 'character' else thought['command']['name']
          command = self._commands[command_name]
          #print(f"AgentCommandValidator command.validate input {thought['command']['input'] or {}}")
          command_validation_result = await command.validate(thought['command']['input'] or {}, memory, functions, tokenizer)
          if not command_validation_result['valid']:
              #print(f"AgentCommandValidator invalid input {thought['command']['input'] or {}}")
              return command_validation_result
          
          # Return the validated thought
          #print(f"AgentCommandValidator command.validate valid")
          return validation_result
        except Exception as e:
          traceback.print_exc()
        return {
            'type': 'Validation',
            'valid': False,
            'feedback': 'The command '+thought['command']['name']+' validation failed, try again'
        }
