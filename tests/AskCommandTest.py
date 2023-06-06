import asyncio
import aiounittest, unittest

from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from alphawave_agents.AskCommand import AskCommand

class TestAskCommand(aiounittest.AsyncTestCase):

    def setUp(self):
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()

    def test_constructor_default_params(self):
        command = AskCommand()
        self.assertEqual(command.title, 'ask')
        self.assertEqual(command.description, 'ask the user a question and wait for their response')
        self.assertEqual(command.inputs, '"question":"<question to ask value>"')
        self.assertEqual(command.output, 'users answer')

    def test_constructor_custom_params(self):
        command = AskCommand('custom title', 'custom description')
        self.assertEqual(command.title, 'custom title')
        self.assertEqual(command.description, 'custom description')
        self.assertEqual(command.inputs, '"question":"<question to ask value>"')
        self.assertEqual(command.output, 'users answer')

    async def test_validate_valid_input(self):
        print('test_validate_valid_input')
        command = AskCommand()
        input = {
            'question': 'how are you?'
        }
        result = await command.validate(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['valid'], True)
        self.assertEqual(result['value'], input)

    async def test_validate_invalid_input(self):
        print('\n\ntest_validate_invalid_input')
        command = AskCommand()
        input = {
            'ask': 'how are you?'
        }
        result = await command.validate(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['valid'], False)
        #self.assertEqual(result['feedback']. 'The command.input has errors:\n"input": requires property "question"\n\nTry again.')

    async def test_execute(self):
        print('\n\ntest_execute')
        command = AskCommand()
        input = {
            'question': 'how are you?'
        }
        result = command.execute(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result, {
            'type': "TaskResponse",
            'status': "input_needed",
            'message': "how are you?"
        })

if __name__ == '__main__':
    unittest.main()
