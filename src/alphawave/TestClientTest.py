import unittest
import yaml
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import  GPT3Tokenizer
from promptrix.Prompt import Prompt

from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.TestClient import TestClient
import asyncio
import json

class TestClientTest(unittest.TestCase):
    def setUp(self):
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.prompt = Prompt([])
        self.options = PromptCompletionOptions(
            completion_type='text',
            model='davinci',
        )

    def test_constructor_default_params(self):
        client = TestClient()
        self.assertEqual(client.status, 'success')
        self.assertEqual(client.response, {'role': 'assistant', 'content': 'Hello World'})

    def test_constructor_custom_params(self):
        client = TestClient('error', 'Hello Error')
        self.assertEqual(client.status, 'error')
        self.assertEqual(client.response, 'Hello Error')

    def test_completePrompt(self):
        client = TestClient()
        response = asyncio.run(client.completePrompt(self.memory, self.functions, self.tokenizer, self.prompt, self.options))
        self.assertEqual(response['status'], 'success')
        self.assertEqual(response['message'], {'role': 'assistant', 'content': 'Hello World'})

if __name__ == '__main__':
    unittest.main()
