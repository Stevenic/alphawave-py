import argparse
import json
import asyncio
import aiounittest, unittest
import os, sys
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.AssistantMessage import AssistantMessage
from promptrix.FunctionRegistry import FunctionRegistry 
from promptrix.Prompt import Prompt
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from alphawave.alphawaveTypes import PromptCompletionOptions, PromptResponse, PromptResponseValidator, Validation
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.RepairTestClient import TestClient as TestClient
from alphawave.AlphaWave import AlphaWave
from alphawave.OpenAIClient import OpenAIClient
import alphawave_pyexts.LLMClient as llm
from alphawave.OSClient import OSClient
from alphawave_agents.Agent import Agent, AgentOptions
from alphawave_agents.MathCommand import MathCommand
from alphawave_agents.FinalAnswerCommand import FinalAnswerCommand

# Create an OpenAI client
#client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
#model = 'gpt-3.5-turbo'

#create OS client
client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)

#global model
#model = 'chatglm2'
model = ''

chat_prime = 'You are a friendly, helpful, courteous AI named John.'

json_prime = """
Reason step by step how to answer the users query below.
Return a JSON object with your reasoning and the answer, using the following format.
Response Format:

{"reasoning": '<reflections on how to construct an answerto the user query>',"answer": '<answer to user query'}}
"""

agent_prime = """
Reason step by step about the input.
Next, chose the command to use next.
Finally, build the input to pass to the command.
You have the following commands available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer input must respond to the original user input
Return a JSON object with your reasoning, command, and input, using the following format.
Response Format:

{"reasoning": 'concise plan for developing answer',"command": 'name of command to perform next', "input": 'input for command'}}
"""

chat_prompt =Prompt([SystemMessage(chat_prime),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"),
                     AssistantMessage('')])

zeroShot_prompt=Prompt([SystemMessage(json_prime),
                        ConversationHistory('history'),
                        UserMessage("{{$input}}"),
                        AssistantMessage('')])

oneShot_prompt=Prompt([SystemMessage(json_prime),
                       UserMessage("What is 2 + 2"),
                       AssistantMessage('{"reasoning":\'That is easy, I know the answer\', "answer":\'4\'}'),
                       ConversationHistory('history'),
                       UserMessage("{{$input}}"), AssistantMessage('')])

agent_prompt=Prompt([SystemMessage(agent_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('{"reasoning":\"I\'m not great at math, better use the math command\", "command":"math", "input":"2 + 2"}'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"), AssistantMessage('')])


agent_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "step-by-step reasoning about task"
        },
        "command": {
            "type":"string",
            "description": "name of next command to use"
        },
        "input": {
            type: "string",
            "description": "input for next command"
        },
    },
    "required": ["reasoning", "command", "input"]
}

json_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string"
        },
        "answer": {
            "type":"string"
        }
    },
    "required": ["reasoning", "answer"]
}


def make_agent(model):
    initial_prompt = \
    """
    Hi! I'm here to help.
    """

    agent_options = AgentOptions(
        client = client,
        prompt=[
            "You are a helpful information bot, willing to dig deep to answer questions"
        ],
        prompt_options=PromptCompletionOptions(
            completion_type = 'chat',
            model = model,
            temperature = 0.2,
            max_input_tokens = 1200,
            max_tokens = 200,
        ),
        initial_thought={
            "reasoning": "I don't know the users needs",
            "plan": "- use the ask command to ask the user what problem I can help with",
            "command": {
                "name": "ask",
                "input": {"question": initial_prompt}
            }
        },
        step_delay=1000,
        max_steps=50,
        max_repair_attempts=2,
        logRepairs=True
    )

    # Create an agent
    agent = Agent(options = agent_options)
    return agent
    


class TestAlphaWave():
    def __init__(self, name):
        #model = input('template_name? ').strip()
        self.name = name
        self.model=model
        self.client = client
        self.prompt = oneShot_prompt
        self.prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature=0.1)
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.validator = JSONResponseValidator(json_validation_schema)
        self.ok = True
        
    def assertTrue(self, x):
        if not x:
            print(f'### {self.name} failed')
            self.ok = False
            
    def end(self):
        if self.ok:
            print(f'### {self.name} passed')
            
    async def test_chat(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] =='success')
        self.end()

    async def test_json_oneShot(self):
        print('\n###############################################################################################')
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=oneShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('Hello. How are you today?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

    async def test_agent_prompt(self):
        print('\n###############################################################################################')
        # Add core commands to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=agent_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=JSONResponseValidator(agent_validation_schema))
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('Hello. How are you today?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        response = await wave.completePrompt('What is your name?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        response = await wave.completePrompt('What is the square root of 2?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

if __name__ == '__main__':
    model = input('template_name? ').strip()
    test = TestAlphaWave('chat')
    asyncio.run(test.test_chat())

    test = TestAlphaWave('json_oneShot ')
    asyncio.run(test.test_json_oneShot())
    test = TestAlphaWave('agent')
    asyncio.run(test.test_agent_prompt())
