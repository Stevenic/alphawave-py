import asyncio
import aiounittest, unittest
import os
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from alphawave.TestClient import TestClient
from alphawave.alphawaveTypes import PromptCompletionOptions
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.TemplateSection import TemplateSection
from promptrix.UserMessage import UserMessage
from promptrix.VolatileMemory import VolatileMemory
from promptrix.promptrixTypes import Message
from alphawave_agents.PromptCommand import PromptCommand, CommandSchema, PromptCommandOptions
from alphawave.AlphaWave import AlphaWave
#from alphawave_agents.SchemaBasedCommand import CommandSchema

class TestPromptCommand(aiounittest.AsyncTestCase):
    def setUp(self):
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.prompt = Prompt([UserMessage("return this string: '{{$fact}}'")])
        self.prompt_response = Message(role='assistant', content="fact remembered")
        #self.client = TestClient('success', self.prompt_response)
        #self.client = OSClient(apiKey=os.getenv("OPENAI_API_KEY"))
        self.client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
        self.schema = CommandSchema(
            schema_type='object',
            title='test',
            description='test description',
            properties={
                'fact': {
                    'type': 'string',
                    'description': 'a fact'
                }
            },
            required=['fact'],
            returns="line of dialog"
        )
        self.prompt_options = PromptCommandOptions(prompt_options=PromptCompletionOptions(completion_type='chat', model='gpt-3.5-turbo'), schema = self.schema)

    def test_constructor(self):
        command = PromptCommand(self.client, prompt=self.prompt, options=self.prompt_options)
        self.assertEqual(command.title, 'test')
        self.assertEqual(command.description, 'test description')
        self.assertEqual(command.inputs, '"fact":"<a fact>"')
        self.assertEqual(command.output, 'line of dialog')

    async def test_validate_valid_input(self):
        command = PromptCommand(prompt=self.prompt, options=self.prompt_options, client=self.client)
        input = {
            'fact': 'test fact'
        }
        result = await command.validate(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['valid'], True)
        self.assertEqual(result['value'], input)


    async def test_validate_invalid_input(self):
        command = PromptCommand(prompt=self.prompt, options=self.prompt_options, client=self.client)
        input = {
            'test': 'test fact'
        }
        result = await command.validate(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['valid'], False)
        print(result['feedback'])
        #self.assertEqual(result['feedback'], 'The command.input has errors:\n"input": \'fact\' is a required property\n\nTry again.')

    async def test_execute(self):
        command = PromptCommand(prompt=self.prompt, options=self.prompt_options, client=self.client)
        input = {
            'fact': 'test fact'
        }
        result = await command.execute(input, self.memory, self.functions, self.tokenizer)
        print (result)
        self.assertEqual(result, 'test fact')
        
if __name__ == '__main__':
    unittest.main()
