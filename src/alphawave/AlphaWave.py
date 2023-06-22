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
    client: PromptCompletionClient = None
    prompt: Prompt = None
    prompt_options: PromptCompletionOptions = None
    memory: PromptMemory = VolatileMemory()
    functions: PromptFunctions = FunctionRegistry()
    history_variable: str = 'history'
    input_variable: str = 'input'
    max_history_messages: int = 10
    max_repair_attempts: int = 3
    tokenizer: Tokenizer = GPT3Tokenizer()
    validator: DefaultResponseValidator = DefaultResponseValidator()
    logRepairs: bool = False

def update_dataclass(instance, **kwargs):
    for key, value in kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)

def display_dataclass(obj):
    print ('\n ', obj)
    for attr_name in dir(obj):
        if not attr_name.startswith('__'):  # Exclude dunder (double underscore) attributes
            attr_value = getattr(obj, attr_name)
            print(f"    {attr_name}: {attr_value}")

def get_values(instance, keys):
    values = []
    for key in keys:
        if hasattr(instance, key):
            values.append(getattr(instance, key))
        else:
            values.append(None)
    return values

class AlphaWave(AsyncIOEventEmitter):
    def __init__(self, **kwargs):
        super().__init__()
        self.options = AlphaWaveOptions(
        )
        update_dataclass(self.options, **kwargs)
        #display_dataclass(self.options)
    async def completePrompt(self, input=None):
        client, prompt, prompt_options, memory, functions, history_variable, input_variable, max_history_messages, max_repair_attempts, tokenizer, validator, logRepairs = get_values(self.options, ('client', 'prompt', 'prompt_options', 'memory', 'functions', 'history_variable', 'input_variable', 'max_history_messages', 'max_repair_attempts', 'tokenizer', 'validator', 'logRepairs'))

        if self.options.input_variable:
            if input:
                memory.set(input_variable, input)
            else:
                input = memory.get(input_variable) if memory.has(input_variable) else ''
        elif not input:
            input = ''

        try:
            self.emit('beforePrompt', memory, functions, tokenizer, prompt, prompt_options)
            response = await client.completePrompt(memory, functions, tokenizer, prompt, prompt_options)
            self.emit('afterPrompt', memory, functions, tokenizer, prompt, prompt_options, response)
            if response['status'] != 'success':
                return response

            if not isinstance(response['message'], dict):
                response['message'] = {'role': 'assistant', 'content': response['message'] or ''}

            self.emit('beforeValidation', memory, functions, tokenizer, response, max_repair_attempts)
            validation = validator.validate_response(memory, functions, tokenizer, response, max_repair_attempts)
            self.emit('afterValidation', memory, functions, tokenizer, response, max_repair_attempts, validation)
            if 'coroutine' in str(type(validation)).lower():
                validation = await validation
            if validation['valid']:
                if 'value' in validation:
                    response['message']['content'] = validation['value']

                self.addInputToHistory(memory, history_variable, input)
                self.addResponseToHistory(memory, history_variable, response['message'])
                return response

            if self.options.logRepairs:
                print(Colorize.title('REPAIRING RESPONSE:'))
                print(Colorize.output(memory))
            fork = MemoryFork(memory)
            #self.addInputToHistory(fork, history_variable, input)
            #self.addResponseToHistory(fork, history_variable, response['message'])

            if self.options.logRepairs:
                print(Colorize.output(response['message']['content']))

            self.emit('beforeRepair', fork, functions, tokenizer, response, max_repair_attempts, validation)
            repair = self.repairResponse(fork, functions, tokenizer, response, validation, max_repair_attempts)
            if 'coroutine' in str(type(repair)).lower():
                repair = await repair
            self.emit('afterRepair', fork, functions, tokenizer, response, max_repair_attempts, validation)

            if self.options.logRepairs:
                if repair['status'] == 'success':
                    self.emit('repairSuccess', fork, functions, tokenizer, response, max_repair_attempts, validation)
                    print(Colorize.success('Response Repaired'))
                else:
                    print(Colorize.error('Response Repair Failed'))

            if repair['status'] == 'success':
                self.addInputToHistory(memory, history_variable, input)
                self.addResponseToHistory(memory,history_variable, repair['message'])
            return repair
        except Exception as err:
            return {
                'status': 'error',
                'message': str(err)
            }

    def addInputToHistory(self, memory, variable, input):
        if variable and input is not None and len(input) > 0:
            history = memory.get(variable) or []
            history.append({'role': 'user', 'content': input})
            if len(history) > self.options.max_history_messages:
                history = history[int(self.options.max_history_messages/2):]
            memory.set(variable, history)

    def addResponseToHistory(self, memory, variable, message):
        if variable:
            history = memory.get(variable) or []
            history.append(message)
            if len(history) > self.options.max_history_messages:
                history = history[int(self.options.max_history_messages/2):]
            memory.set(variable, history)

    async def repairResponse(self, fork, functions, tokenizer, response, validation, remaining_attempts):
        client, prompt, prompt_options, memory, functions, history_variable, input_variable, max_history_messages, max_repair_attempts, tokenizer, validator, log_repairs = get_values(self.options, ('client', 'prompt', 'prompt_options', 'memory', 'functions', 'history_variable', 'input_variable', 'max_history_messages', 'max_repair_attempts', 'tokenizer', 'validator', 'log_repairs'))

        # Are we out of attempts?
        feedback = validation.get('feedback', 'The response was invalid. Try another strategy.')
            
        if remaining_attempts <= 0:
            return {
                'status': 'invalid_response',
                'message': feedback
            }

        # Add response and feedback to repair history
        self.addResponseToHistory(fork, f"{self.options.history_variable}-repair", response['message'])
        self.addInputToHistory(fork, f"{self.options.history_variable}-repair", feedback)

        # Append repair history to prompt
        repair_prompt = Prompt([
            prompt,
            ConversationHistory(f"{self.options.history_variable}-repair")
        ])

        # Log the repair
        if self.options.logRepairs:
            print(Colorize.value('feedback', feedback))

        # Ask client to complete prompt
        repair_response = await client.completePrompt(fork, functions, tokenizer, repair_prompt, prompt_options)
        if repair_response['status'] != 'success':
            return repair_response

        # Ensure response is a message
        if not isinstance(repair_response['message'], dict):
            repair_response['message'] = { 'role': 'assistant', 'content': repair_response.get('message', '') }

        # Validate response
        validation = validator.validate_response(fork, functions, tokenizer, repair_response, remaining_attempts)
        if 'coroutine' in str(type(validation)).lower():
            validation = await validation
        if validation['valid']:
            # Update content
            if 'value' in validation:
                repair_response['message']['content'] = validation['value']

            return repair_response

        # Try next attempt
        remaining_attempts -= 1
        return await self.repairResponse(fork, functions, tokenizer, repair_response, validation, remaining_attempts)
