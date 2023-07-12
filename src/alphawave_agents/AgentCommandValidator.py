from pyee import AsyncIOEventEmitter
import asyncio
from typing import Dict, Any, Union
from promptrix.promptrixTypes import Message, PromptFunctions, PromptSection, PromptMemory, Tokenizer
from promptrix.ConversationHistory import ConversationHistory
from promptrix.AssistantMessage import AssistantMessage
from promptrix.UserMessage import UserMessage
from promptrix.FunctionRegistry import  FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.VolatileMemory import VolatileMemory
from promptrix.Utilities import Utilities
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave.Response import Response
from alphawave.AlphaWave import AlphaWave, AlphaWaveOptions
from alphawave.MemoryFork import MemoryFork
from alphawave_agents.agentTypes import Command, AgentThought, AgentThoughtSchemaJSON, AgentThoughtSchemaTOML
from alphawave_agents.MathCommand import MathCommand
import traceback


class AgentCommandValidator:
    def __init__(self, commands: dict[str, Command], client=None, model: str='', syntax:str='JSON',
                 memory=VolatileMemory(), history_variable='history'):
        self._model = model
        self._client = client
        self._syntax = syntax
        self._memory=memory
        self._history_variable = history_variable
        
        if syntax == 'JSON':
            self._input_validator = JSONResponseValidator(AgentThoughtSchemaJSON, 'No valid command was found in the response. Respond using a completed instance of the specified template for the next command.')
        else:
            #print(f"***** AgentCommandValidator creating TOML")
            self._input_validator = TOMLResponseValidator(AgentThoughtSchemaTOML, 'No valid command was found. Respond using an instantiation of the specified template for the next command.')

        self._commands = commands

    def validate_response(self, memory, functions, tokenizer, response, remaining_attempts) -> Union[AgentThought, None]:
        try:
          #print(f"***** AgentCommandValidator validate syntax: \n{self._syntax}\nresponse\n{response}")
          message = response['message']
          raw_text = message if isinstance(message, str) else message.get('content', '')
          if (self._syntax == 'JSON' and '{' not in raw_text) or (self._syntax == 'TOML' and 'response' not in raw_text.lower()):
              # means no form found, assume llm doesn't need to use a command
              return {
                  'type': 'Validation',
                  'valid': True,
                  'value': raw_text,
                  'feedback': 'no command'
              }
          
          # Validate that the response contains a thought
          validation_result = self._input_validator.validate_response(memory, functions, tokenizer, response, remaining_attempts)
          if not validation_result['valid']:
            return validation_result

          # Validate that the command exists
          thought = validation_result['value']
          #print(f'*****AgentCommandValidator post validate thought  \n{thought}\n')
          if not('command' in thought) or not('inputs' in thought):
              print(f"***** AgentCommandValidator command or inputs not found")
              return {
                  'type': 'Validation',
                  'valid': False,
                  'feedback': f'command not found or invalid, or inputs missing. The commands you have are {list(self._commands.keys())}'
              }

          command_name = thought['command']
          if command_name not in self._commands:
              print(f"***** AgentCommandValidator no such command {command_name}")
              return {
                  'type': 'Validation',
                  'valid': False,
                  'feedback': f'The command {command_name} does not exist. The only commands you have are {list(self._commands.keys())}'
              }
          
          # Validate that the command input is valid
          command = self._commands[command_name]
          command_validation_result = command.validate(thought['inputs'] or {}, memory, functions, tokenizer, syntax = self._syntax)
          if command_validation_result['valid']:
              #print(f"***** AgentCommandValidator command validation success\n{command_validation_result}\n")
              #validation_result['value']['inputs'] = command_validation_result['value']
              return validation_result
          else:
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': 'The command '+thought['command'] +' inputs were missing or malformed. Provide them in this format:\n'
                +command.one_shot(self._syntax)+'\n, substituting actual values for command inputs\n'
                }

        except Exception as e:
            traceback.print_exc()
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f'The command validation failed. try again {str(e)}'
                }
        #print(f"***** AgentCommandValidator generic exit fail {thought['command']['name']}")
        return {
            'type': 'Validation',
            'valid': False,
            'feedback': f'The command validation failed. try again {str(e)}'
                }

    ### interesting experiment, set aside for now
    def repair_args(self, command, fail_thought):
        #print(f"***** AgentCommandValidator recovery attempt keys {list(self._memory._memory.keys())}")
        args_validator = JSONResponseValidator(command.schema, "invalid command inputs syntax, use: {command.one_shot()\n}")
        fork = MemoryFork(self._memory)
        prompt_options=PromptCompletionOptions(completion_type='chat', model=self._model, temperature=0.1)
        args_prompt=Prompt([ConversationHistory(self._history_variable),
                            UserMessage(f'invalid command args: {fail_args}, repair using this format: {command.schema["properties"]}')])
        #print(f"***** AgentCommandValidator recovery attempt wave built\ninvalid command args: {fail_args}, repair using this format: {command.schema['properties']}")
        wave = AlphaWave(client=self._client, prompt=args_prompt, prompt_options=prompt_options, memory=fork)
        #print(f"***** AgentCommandValidator recovery attempt wave built")
        args = None
        try:
            args = wave.completePrompt()
            #print(f"***** AgentCommandValidator recovery wave result {args}")
        except Exception as e:
            traceback.print_exc()
        #print(f'***** recovery result {args}')
        if args:
            return args
        #print(f"***** AgentCommandValidator recovery returning fail_args {fail_args}")
        return fail_args


from alphawave.OSClient import OSClient

