import asyncio
import aiounittest, unittest
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from alphawave_agents.MathCommand import MathCommand

class TestMathCommand(aiounittest.AsyncTestCase):
    def setUp(self):
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()

    def test_constructor(self):
        command = MathCommand()
        self.assertEqual(command.title, 'math')
        self.assertEqual(command.description, 'execute some python code to calculate a value')
        self.assertEqual(command.inputs, '"code":"<python expression to evaluate>"')
        self.assertEqual(command.output, 'the calculated value')

        command = MathCommand('custom title', 'custom description')
        self.assertEqual(command.title, 'custom title')
        self.assertEqual(command.description, 'custom description')
        self.assertEqual(command.inputs, '"code":"<python expression to evaluate>"')
        self.assertEqual(command.output, 'the calculated value')

    async def test_validate(self):
        command = MathCommand()
        input = {
            'code': '7 + 3'
        }
        result = await command.validate(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['valid'], True)
        self.assertEqual(result['value'], input)

    def test_execute(self):
        command = MathCommand()
        input = {
            'code': '7 + 3'
        }
        result = command.execute(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result, 10)
        input = {
            'code': '7 +'
        }
        result = command.execute(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['message'], 'invalid syntax (<string>, line 1)')

if __name__ == '__main__':
    unittest.main()
