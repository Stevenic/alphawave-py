from pyee import AsyncIOEventEmitter
from typing import Dict, Any, Union
from alphawave_agents.agentTypes import Command, AgentThought, AgentThoughtSchema

class AgentCommandValidator:
    def __init__(self, commands: dict[str, Command]):
        self._thought_validator = JSONResponseValidator(AgentThoughtSchema, 'No valid JSON objects were found in the response. Return a valid JSON object with your thoughts and the next command to perform.')
        self._commands = commands

    async def validate_response(self, memory, functions, tokenizer, response, remaining_attempts) -> Union[AgentThought, None]:
        # Validate that the response contains a thought
        validation_result = await self._thought_validator.validate_response(memory, functions, tokenizer, response, remaining_attempts)
        if not validation_result.valid:
            return validation_result

        # Validate that the command exists
        thought = validation_result.value
        if thought.command.name not in self._commands:
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f'The command "{thought.command.name}" does not exist. Try a different command.'
            }

        # Validate that the command input is valid
        command = self._commands[thought.command.name]
        command_validation_result = await command.validate(thought.command.input or {}, memory, functions, tokenizer)
        if not command_validation_result.valid:
            return command_validation_result

        # Return the validated thought
        return validation_result
