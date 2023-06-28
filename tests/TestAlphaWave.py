import asyncio
import aiounittest, unittest
from assertpy import assert_that
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from promptrix.FunctionRegistry import FunctionRegistry 
from promptrix.Prompt import Prompt
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from alphawave.alphawaveTypes import PromptCompletionOptions, PromptResponse, PromptResponseValidator, Validation
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.RepairTestClient import TestClient as TestClient
from alphawave.AlphaWave import AlphaWave
import json

class TestValidator(PromptResponseValidator):
    def __init__(self, client):
        self.feedback = 'Something is wrong'
        self.repairAttempts = 0
        self.exception = None
        self.clientErrorDuringRepair = False
        self.returnContent = False
        self.client = client

    def validate_response(self, memory, functions, tokenizer, response, remaining_attempts):
        print(f'***** validate_response {remaining_attempts}, {self.clientErrorDuringRepair}, {response}')
        if self.exception:
            exception = self.exception
            self.exception = None
            raise exception

        if self.clientErrorDuringRepair and self.repairAttempts == 1:
            print(f'***** validate_response CLDR==True, rA ==1')
            self.clientErrorDuringRepair = False
            self.client.status = 'error'
            self.client.response = 'Some Error'
            return {'type': 'Validation', 'valid': False, 'feedback': self.feedback }
        elif self.repairAttempts > 0:
            print(f'***** validate_response self.rA > 0')
            self.repairAttempts -= 1
            return { 'type': 'Validation', 'valid': False, 'feedback': self.feedback }
        elif self.returnContent:
            print(f"***** validate_response self.Content True { 'type': 'Validation', 'valid': True, 'value': response['message']['content'] }")
            self.returnContent = False
            return { 'type': 'Validation', 'valid': True, 'value': response['message']['content'] }
        else:
            print(f'***** validate_response final else')
            return { 'type': 'Validation', 'valid': True }

class TestAlphaWave(aiounittest.AsyncTestCase):
    def setUp(self):
        self.client = TestClient('success', { 'role': 'assistant', 'content': 'Hello' })
        self.prompt = Prompt([])
        self.prompt_options = PromptCompletionOptions(completion_type='chat', model='test')
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.validator = TestValidator(self.client)

    def test_constructor(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options)
        assert_that(wave).is_not_none()
        assert_that(wave.options).is_not_none()
        assert_that(wave.options.client).is_equal_to(self.client)
        assert_that(wave.options.prompt).is_equal_to(self.prompt)
        assert_that(wave.options.prompt_options).is_equal_to(self.prompt_options)
        assert_that(isinstance(wave.options.memory, VolatileMemory)).is_true()
        assert_that(isinstance(wave.options.functions, FunctionRegistry)).is_true()
        assert_that(isinstance(wave.options.tokenizer, GPT3Tokenizer)).is_true()
        assert_that(isinstance(wave.options.validator, DefaultResponseValidator)).is_true()
        assert_that(wave.options.history_variable).is_equal_to('history')
        assert_that(wave.options.input_variable).is_equal_to('input')
        assert_that(wave.options.max_repair_attempts).is_equal_to(3)
        assert_that(wave.options.max_history_messages).is_equal_to(10)

        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator, history_variable='test_history', input_variable='test_input', max_repair_attempts=5, max_history_messages=20)
        assert_that(wave).is_not_none()
        assert_that(wave.options).is_not_none()
        assert_that(wave.options.client).is_equal_to(self.client)
        assert_that(wave.options.prompt).is_equal_to(self.prompt)
        assert_that(wave.options.prompt_options).is_equal_to(self.prompt_options)
        assert_that(wave.options.memory).is_equal_to(self.memory)
        assert_that(wave.options.functions).is_equal_to(self.functions)
        assert_that(wave.options.tokenizer).is_equal_to(self.tokenizer)
        assert_that(wave.options.validator).is_equal_to(self.validator)
        assert_that(wave.options.history_variable).is_equal_to('test_history')
        assert_that(wave.options.input_variable).is_equal_to('test_input')
        assert_that(wave.options.max_repair_attempts).is_equal_to(5)
        assert_that(wave.options.max_history_messages).is_equal_to(20)

    async def test_basic_prompt_completion(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        response = await wave.completePrompt()
        print(f'***** {response}')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'assistant', 'content': 'Hello' }])
        input = self.memory.get('input')
        assert_that(input).is_none()
        self.memory.clear()

        self.client.response = 'Hello'
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': 'Hello' }])
        input = self.memory.get('input')
        assert_that(input).is_equal_to('Hi')
        self.memory.clear()

    async def test_prompt_completion_with_validation(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.response = 'Hello'
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('Hi')
        print(response)
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': 'Hello' }])
        self.memory.clear()

        self.client.response = 'Hello'
        self.validator.repairAttempts = 2
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': 'Hello' }])
        self.memory.clear()

    async def test_prompt_completion_with_repair(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.response = 'Hello'
        self.validator.repairAttempts = 4
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('invalid_response')
        assert_that(response['message']).is_equal_to(self.validator.feedback)
        history = self.memory.get('history')
        assert_that(history).is_none()
        self.memory.clear()

        self.client.response = 'Hello'
        self.validator.repairAttempts = 2
        self.validator.clientErrorDuringRepair = True
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('error')
        assert_that(response['message']).is_equal_to('Some Error')
        self.memory.clear()

    async def test_prompt_completion_with_default_feedback(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = 'Hello'
        self.validator.repairAttempts = 1
        self.validator.feedback = None
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': 'Hello' }])
        self.memory.clear()

    async def test_prompt_completion_with_undefined_response(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = None
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': None })
        self.memory.clear()

    async def test_prompt_completion_with_message_object_response(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = { 'role': 'assistant', 'content': { 'foo': 'bar'} }
        self.validator.repairAttempts = 1
        self.validator.returnContent = True
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': { 'foo': 'bar'} })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': { 'foo': 'bar'} }])
        self.memory.clear()

    async def test_prompt_completion_with_repaired_response(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = { 'role': 'assistant', 'content': 'Hello World' }
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello World' })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': 'Hello World' }])
        self.memory.clear()

    async def test_prompt_completion_with_parsed_content_object(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = { 'role': 'assistant', 'content': { 'foo': 'bar'} }
        self.validator.repairAttempts = 1
        self.validator.returnContent = True
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': { 'foo': 'bar'} })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': { 'foo': 'bar'} }])
        self.memory.clear()

    async def test_prompt_completion_with_repaired_response_undefined(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = None
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': '' })
        self.memory.clear()

    async def test_prompt_completion_with_repaired_response_message_object(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = { 'role': 'assistant', 'content': 'Hello World' }
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello World' })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': 'Hello World' }])
        self.memory.clear()

    async def test_prompt_completion_with_repaired_response_parsed_content_object(self):
        wave = AlphaWave(client=self.client, prompt=self.prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        
        self.client.status = 'success'
        self.client.response = { 'role': 'assistant', 'content': { 'foo': 'bar'} }
        self.validator.repairAttempts = 1
        self.validator.returnContent = True
        response = await wave.completePrompt('Hi')
        assert_that(response['status']).is_equal_to('success')
        assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': { 'foo': 'bar'} })
        history = self.memory.get('history')
        assert_that(history).is_equal_to([{ 'role': 'user', 'content': 'Hi' },{ 'role': 'assistant', 'content': { 'foo': 'bar'} }])
        self.memory.clear()

if __name__ == '__main__':
    print('starting')
    unittest.main()
