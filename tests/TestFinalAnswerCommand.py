import asyncio
import aiounittest, unittest
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from alphawave_agents.FinalAnswerCommand import FinalAnswerCommand


class TestFinalAnswerCommand(aiounittest.AsyncTestCase):
    def setUp(self):
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()

    def test_constructor(self):
        command = FinalAnswerCommand()
        self.assertEqual(command.title, 'finalAnswer')
        self.assertEqual(command.description, 'show answer to the user')
        self.assertEqual(command.inputs, '"answer":"<answer to show user>"')
        self.assertEqual(command.output, 'a followup task or question')

        command = FinalAnswerCommand('custom title', 'custom description')
        self.assertEqual(command.title, 'custom title')
        self.assertEqual(command.description, 'custom description')
        self.assertEqual(command.inputs, '"answer":"<answer to show user>"')
        self.assertEqual(command.output, 'a followup task or question')

    async def test_validate(self):
        command = FinalAnswerCommand()
        input = {
            'answer': 'final answer'
        }
        result = command.validate(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['valid'], True)
        self.assertEqual(result['value'], input)

        command = FinalAnswerCommand()
        input = {
            'foo': 'final answer'
        }
        result = command.validate(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result['valid'], False)
        print(f"***** TestFinalAnswerCommand feedback {result['feedback']}")
        #self.assertEqual(result['feedback'], f'The command.input has errors:\n{input}: requires property "answer"\n\nTry again.')

    def test_execute(self):
        command = FinalAnswerCommand()
        input = {
            'answer': 'final answer'
        }
        result = command.execute(input, self.memory, self.functions, self.tokenizer)
        self.assertEqual(result, {
            'type': "TaskResponse",
            'status': "input_needed",
            'message': "final answer"
        })


if __name__ == '__main__':
    unittest.main()
