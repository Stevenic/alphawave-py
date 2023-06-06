from typing import Callable, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pyee import AsyncIOEventEmitter
import readline as re

from promptrix.promptrixTypes import Message, PromptFunctions, PromptSection, PromptMemory, Tokenizer
from promptrix.ConversationHistory import ConversationHistory
from promptrix.FunctionRegistry import  FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.VolatileMemory import VolatileMemory
from promptrix.Utilities import Utilities

from alphawave.alphawaveTypes import  PromptCompletionClient, PromptCompletionOptions, PromptResponse,Validation, PromptResponseValidator 
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.MemoryFork import  MemoryFork
from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse, Validation, PromptResponseValidator
from alphawave.Colorize import Colorize
import traceback

@dataclass
class AlphaWaveOptions:
    def __init__(self, client: PromptCompletionClient, prompt: PromptSection, prompt_options: PromptCompletionOptions, functions: Optional[PromptFunctions] = None, history_variable: Optional[str] = None, input_variable: Optional[str] = None, max_history_messages: Optional[int] = None, max_repair_attempts: Optional[int] = None, memory: Optional[PromptMemory] = None, tokenizer: Optional[Tokenizer] = None, validator: Optional[PromptResponseValidator] = None, logRepairs: Optional[bool] = None):
        self.client = client
        self.prompt = prompt
        self.prompt_options = prompt_options
        self.functions = functions
        self.history_variable = history_variable
        self.input_variable = input_variable
        self.max_history_messages = max_history_messages
        self.max_repair_attempts = max_repair_attempts
        self.tokenizer = tokenizer
        self.validator = validator
        self.logRepairs = logRepairs


class AlphaWave(AsyncIOEventEmitter):
    def __init__(self, **kwargs):
        super().__init__()
        self.options = {
            'client': PromptCompletionClient,
            'prompt': PromptSection,
            'prompt_options': None,
            'memory':VolatileMemory(),
            'functions': FunctionRegistry(),
            'history_variable': 'history',
            'input_variable': 'input',
            'max_history_messages': 10,
            'max_repair_attempts': 3,
            'tokenizer': GPT3Tokenizer(),
            'validator': DefaultResponseValidator(),
            'logRepairs': False
        }
        self.options.update(kwargs)

    async def completePrompt(self, input=None):
        client, prompt, prompt_options, memory, functions, history_variable, input_variable, max_history_messages, max_repair_attempts, tokenizer, validator, log_repairs = self.options.values()
        
        if self.options['input_variable']:
            if input:
                memory.set(input_variable, input)
            else:
                input = memory.get(input_variable) if memory.has(input_variable) else ''
        elif not input:
            input = ''

        try:
            self.emit('beforePrompt', memory, functions, tokenizer, prompt, prompt_options)
            response = await client.complete_prompt(memory, functions, tokenizer, prompt, prompt_options)
            self.emit('afterPrompt', memory, functions, tokenizer, prompt, prompt_options, response)
            if response['status'] != 'success':
                return response

            if not isinstance(response['message'], dict):
                response['message'] = {'role': 'assistant', 'content': response['message'] or ''}

            self.emit('beforeValidation', memory, functions, tokenizer, response, max_repair_attempts)
            validation = validator.validate_response(memory, functions, tokenizer, response, max_repair_attempts)
            self.emit('afterValidation', memory, functions, tokenizer, response, max_repair_attempts, validation)
            if validation['valid']:
                if 'value' in validation:
                    response['message']['content'] = validation['value']

                self.addInputToHistory(memory, history_variable, input)
                self.addResponseToHistory(memory, history_variable, response['message'])
                return response

            fork = MemoryFork(memory)
            self.addInputToHistory(fork, history_variable, input)
            self.addResponseToHistory(fork, history_variable, response['message'])

            if self.options['logRepairs']:
                print(Colorize.title('REPAIRING RESPONSE:'))
                print(Colorize.output(response['message']['content']))

            self.emit('beforeRepair', fork, functions, tokenizer, response, max_repair_attempts, validation)
            repair = await self.repairResponse(fork, functions, tokenizer, response, validation, max_repair_attempts)
            self.emit('afterRepair', fork, functions, tokenizer, response, max_repair_attempts, validation)

            if self.options['logRepairs']:
                if repair['status'] == 'success':
                    print(Colorize.success('Response Repaired'))
                else:
                    print(Colorize.error('Response Repair Failed'))

            if repair['status'] == 'success':
                self.addInputToHistory(memory, history_variable, input)
                self.addResponseToHistory(memory,history_variable, repair['message'])
            return repair
        except Exception as err:
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(err)
            }

    def addInputToHistory(self, memory, variable, input):
        if variable and len(input) > 0:
            history = memory.get(variable) or []
            history.append({'role': 'user', 'content': input})
            if len(history) > self.options['max_history_messages']:
                history = history[-self.options['max_history_messages']:]
            memory.set(variable, history)

    def addResponseToHistory(self, memory, variable, message):
        if variable:
            history = memory.get(variable) or []
            history.append(message)
            if len(history) > self.options['max_history_messages']:
                history = history[-self.options['max_history_messages']:]
            memory.set(variable, history)

    async def repairResponse(self, fork, functions, tokenizer, response, validation, remaining_attempts):
        client, prompt, prompt_options, memory, functions, history_variable, input_variable, max_history_messages, max_repair_attempts, tokenizer, validator, log_repairs = self.options.values()

        print(f'repairResponse {remaining_attempts}, {validation}\n {response}')
        # Are we out of attempts?
        feedback = validation.get('feedback', 'The response was invalid. Try another strategy.')
        if remaining_attempts <= 0:
            return {
                'status': 'invalid_response',
                'message': feedback
            }

        # Add response and feedback to repair history
        self.addResponseToHistory(fork, f"{self.options['history_variable']}-repair", response['message'])
        self.addInputToHistory(fork, f"{self.options['history_variable']}-repair", feedback)

        # Append repair history to prompt
        repair_prompt = Prompt([
            prompt,
            ConversationHistory(f"{self.options['history_variable']}-repair")
        ])

        # Log the repair
        if self.options['logRepairs']:
            print(Colorize.value('feedback', feedback))

        # Ask client to complete prompt
        repair_response = await client.complete_prompt(fork, functions, tokenizer, repair_prompt, prompt_options)
        if repair_response['status'] != 'success':
            return repair_response

        # Ensure response is a message
        if not isinstance(repair_response['message'], dict):
            repair_response['message'] = { 'role': 'assistant', 'content': repair_response.get('message', '') }

        # Validate response
        validation = validator.validate_response(fork, functions, tokenizer, repair_response, remaining_attempts)
        if validation['valid']:
            # Update content
            if 'value' in validation:
                repair_response['message']['content'] = validation['value']

            return repair_response

        # Try next attempt
        remaining_attempts -= 1
        return await self.repairResponse(fork, functions, tokenizer, repair_response, validation, remaining_attempts)
    """
    {feedback = validation.get('feedback', 'The response was invalid. Try another strategy.')
        if remaining_attempts <= 0:
            return {
                'status': 'invalid_response',
                'message': feedback
            }

        fork.set(input_variable, feedback)

        if self.options['logRepairs']:
            print(Colorize.value('feedback', feedback))

        self.emit('beforePrompt', fork, functions, tokenizer, prompt, prompt_options)
        response = await client.complete_prompt(fork, functions, tokenizer, prompt, prompt_options)
        self.emit('afterPrompt', fork, functions, tokenizer, prompt, prompt_options, response)
        if response['status'] != 'success':
            return response

        if not isinstance(response['message'], dict):
            response['message'] = {'role': 'assistant', 'content': response['message'] or ''}

        self.emit('beforeValidation', fork, functions, tokenizer, response, remaining_attempts)
        validation = await validator.validate_response(fork, functions, tokenizer, response, remaining_attempts)
        self.emit('afterValidation', fork, functions, tokenizer, response, remaining_attempts, validation)
        if validation['valid']:
            if 'value' in validation:
                response['message']['content'] = validation['value']
            return response

        remaining_attempts -= 1
        self.emit('nextRepair', fork, functions, tokenizer, response, remaining_attempts, validation)
        return await self.repairResponse(fork, functions, tokenizer, validation, remaining_attempts)
    """
