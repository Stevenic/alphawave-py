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
            self._input_validator = JSONResponseValidator(AgentThoughtSchemaJSON, 'No valid JSON objects were found in the response. Return a valid JSON object with reasoning, plan, and the next command to use.')
        else:
            #print(f"***** AgentCommandValidator creating TOML")
            self._input_validator = TOMLResponseValidator(AgentThoughtSchemaTOML, 'No valid TOML found. Return valid TOML with reasoning, plan, and next command to use.')

        self._commands = commands

    async def validate_response(self, memory, functions, tokenizer, response, remaining_attempts) -> Union[AgentThought, None]:
        # Validate that the response contains a thought
        try:
          #print(f"***** AgentCommandValidator validate syntax: \n{self._syntax}\nresponse\n{response}")
          validation_result = self._input_validator.validate_response(memory, functions, tokenizer, response, remaining_attempts)
          if not validation_result['valid']:
              #if self._syntax == 'JSON':
              #    print(f"***** AgentCommandValidator initial validation fail schema:\n{AgentThoughtSchemaJSON}\n{validation_result}")
              #else:
              #    print(f"***** AgentCommandValidator initial validation fail schema:\n{AgentThoughtSchemaTOML}\n{validation_result}")
              return validation_result

          # Validate that the command exists
          thought = validation_result['value']
          print(f'*****AgentCommandValidator post validate thought  \n{thought}\n')
          if not('command' in thought) or not('inputs' in thought['command']) or not ('name' in thought['command']):
              print(f"***** AgentCommandValidator no command found")
              return {
                  'type': 'Validation',
                  'valid': False,
                  'feedback': 'No command found. The commands you have are {list(self._commands.keys())}'
              }

          command_name = thought['command']['name']
          if command_name not in self._commands:
              print(f"***** AgentCommandValidator no such command {command_name}")
              return {
                  'type': 'Validation',
                  'valid': False,
                  'feedback': f'The command {command_name} does not exist. The only commands you have are {list(self._commands.keys())}'
              }
          
          # Validate that the command input is valid
          command = self._commands[command_name]
          command_validation_result = await command.validate(thought['command']['inputs'] or {}, memory, functions, tokenizer, syntax = self._syntax)
          if not command_validation_result['valid']:
              print(f"***** AgentCommandValidator command.validate fail, trying repair {command_validation_result}")
              try:
                  command_validation_retry =  await self.repair_args(command, thought['command']['inputs'] or {})
              except Exception as e:
                  traceback.print_exc()
                  raise e
              print(f"***** AgentCommandValidator command.validate repair {command_validation_retry}")
              if not command_validation_retry['valid']:
                  return command_validation_retry
              else:
                  command_validation_result = command_validation_retry
          print(f"***** AgentCommandValidator command validation success\n{command_validation_result['value']}")
          validation_result['value']['command']['inputs'] = command_validation_result['value']
          return validation_result
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
            'feedback': 'The command '+thought['command']['name']+' inputs were missing or malformed. Provide them in this format: '+command.schema
        }

    async def repair_args(self, command, fail_args):
        print(f"***** AgentCommandValidator recovery attempt ")
        args_validator = JSONResponseValidator(command.schema, "invalid command args syntax, use: {command.schema}")

        prompt_options=PromptCompletionOptions(completion_type='chat', model=self._model, temperature=0.1)
        args_prompt=Prompt([ConversationHistory('history'),
                            UserMessage(f'invalid command args: {fail_args}, repair using this format: {command.schema["properties"]}')])
    
        wave = AlphaWave(client=self._client, prompt=args_prompt, prompt_options=prompt_options, memory=self._memory)
        print(f"***** AgentCommandValidator recovery attempt ")
        try:
            args = await wave.completePrompt()
        except Exception as e:
            traceback.print_exc()
        print(f'***** recovery result {args}')
        if args:
            return args
        return command_validation_result


from alphawave.OSClient import OSClient

if __name__ == '__main__':
    memory=VolatileMemory()
    memory.set('history', {'role': 'assistant', 'content': {"reasoning":"use math", "command":{"name":"math", "inputs":{"abc":"xyz"}}}})
    acv = AgentCommandValidator(commands={"math":MathCommand()}, client=OSClient(logRequests=True), model='vicuna_v1.1', syntax='JSON', memory=memory)
    print('acv created')
    result = asyncio.run(acv.repair_args(MathCommand(), {'abc':'xyz'}))
    print(result)
