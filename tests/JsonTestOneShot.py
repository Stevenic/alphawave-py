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
client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)

#global model
#model = 'chatglm2'
model = ''

chat_prime = 'You are a friendly, helpful, courteous AI named John.'

prime = """
Reason step by step how to answer the users query below.
Return a JSON object with your thoughts and the answer, using the following format.
Response Format:

{"reasoning": '<reflections on how to construct an answerto the user query>',"answer": '<answer to user query'}}
"""

agent_prime = """
Reason step by step how to answer the users query below.
Return a JSON object with your thoughts and the answer, using the following format.
Response Format:

{"reasoning": '<reflections on how to construct an answerto the user query>',"action": {"name":'name of action', "input": 'input for action'}}}
"""


pre_agent_prime = """
Reason step by step about the input.
You have the following actions available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer input must respond to the original user input
chose the action to use next.
Finally, build the input to pass to the action.
Return a JSON object with your reasoning, action, and input, using the following format.
Response Format:

{"reasoning": 'concise plan for developing answer',"action": 'name of action to use next', "input": 'input for action'}}
"""


planning_prime="""
Reason step by step about the user task.
You have the following actions available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer input must respond to the original user input
Develop a concise plan to perform the task and present the result to the user.
Return a JSON object with your reasoning and plan. Use the following format.
Response Format:

{"reasoning": 'thoughts about the task', "plan":'concise plan for developing answer'}
"""

yplanning_prime="""
Reason step by step about the user task.
You have the following actions available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer input must respond to the original user input
Develop a concise plan to perform the task and present the result to the user.
Return a TOML object with your reasoning and plan. Use the following format.
Response Format:

[RESPONSE]
reasoning = 'thoughts about the task'
plan ='concise plan for developing answer'


"""


chat_prompt =Prompt([SystemMessage(chat_prime),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"),
                     AssistantMessage('')])

zeroShot_prompt=Prompt([SystemMessage(prime),
                        ConversationHistory('history'),
                        UserMessage("{{$input}}"),
                        AssistantMessage('')])

oneShot_prompt=Prompt([SystemMessage(prime),
                       UserMessage("What is 2 + 2"),
                       AssistantMessage('{"reasoning":\'That is easy, I know the answer\', "answer":\'4\'}'),
                       ConversationHistory('history'),
                       UserMessage("{{$input}}"), AssistantMessage('')])

planning_prompt=Prompt([SystemMessage(yplanning_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('{"reasoning":\"I\'m not great at math, and I need to answer the user.\", "plan":"math, answer"}'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"), AssistantMessage('')])

yplanning_prompt=Prompt([SystemMessage(planning_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('[RESPONSE]\nreasoning = "I\'m not great at math, and I need to answer the user.\"\nplan = "math, answer"\n'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"), AssistantMessage('')])

pre_agent_prompt=Prompt([SystemMessage(pre_agent_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('{"reasoning":\"I\'m not great at math, better use the math action\", "action":"math", "input":"2 + 2"}'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"), AssistantMessage('')])

agent_prompt=Prompt([SystemMessage(agent_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('{"reasoning":\'That is easy, I know the answer\', "answer":\'4\'}'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}"), AssistantMessage('')])


validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "answer": {
            "type":"string",
            "description": "<response to query>"
        }
    },
    "required": ["reasoning", "answer"]
}

planning_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "plan": {
            "type":"string",
            "description": "actions in plan"
        },
        "required": ["reasoning", "plan"]
    }
}
pre_agent_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "action": {
            "type":"string",
            "description": "<name of chosen action>"
        },
        "input": {
            type: "string",
            "description": "<input for chosen action>"
        },
    },
    "required": ["reasoning", "action", "input"]
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
            "plan": "- use the ask action to ask the user what problem I can help with",
            "action": {
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
        self.validator = JSONResponseValidator(validation_schema)
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

    async def test_chat_recall(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.assertTrue(len(self.memory.get('history')) == 2)
        self.memory.set('input', 'What is your name?')
        #print(f"history: {self.memory.get('history')}")
        response = await wave.completePrompt()
        self.assertTrue(len(self.memory.get('history')) == 4)
        self.assertTrue(response['status'] == 'success')
        print(response['message']['content'])
        self.assertTrue('John' in response['message']['content'])
        self.end()

    async def test_reasoning(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', """Situation: In the room there are John, Mark, a cat, a box, and a basket. John takes the cat and puts it in the basket. He leaves the room and goes to school. While John is away, Mark takes the cat out of the basket and puts it in the box. Mark leaves the room and goes to work. John comes back from school and enters the room. He doesnt know what happened in the room when he was away.

Question: Where does John think the cat is when he re-enters the room??""")
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.assertTrue('basket' in response['message']['content'])
        print(response['message']['content'])
        self.end()

    async def test_json_zeroShot(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=zeroShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

    async def test_json_zeroShot_2turn(self):
        print('\n###############################################################################################')
        wave = AlphaWave(client=self.client, prompt=zeroShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.memory.set('input', 'Hello. What is your name?')
        response = await wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        print(response)
        #assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        self.end()

    async def test_json_oneShot(self):
        print('\n###############################################################################################')
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=oneShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.validator.repairAttempts = 0
        response = await wave.completePrompt('Hello. How are you today?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()
        
    async def test_json_oneShot_repair(self):
        print('\n###############################################################################################')
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=oneShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)

        self.validator.repairAttempts = 2
        response = await wave.completePrompt('Hello. How are you today?')
        print(response)
        self.assertTrue(response is not None)
        if response is not None:
            self.assertTrue('status' in response)
            if 'status' in response:
                self.assertTrue(response['status'] == 'success')
        self.end()
        
    async def test_planning(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=planning_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=JSONResponseValidator(planning_validation_schema))
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('What is the square root of 2?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

    async def test_yplanning(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=yplanning_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.validator.repairAttempts = 1
        response = await wave.completePrompt('What is the square root of 2?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

    async def test_pre_agent(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=pre_agent_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=JSONResponseValidator(pre_agent_validation_schema))
        self.validator.repairAttempts = 1
        #response = await wave.completePrompt('Hello. How are you today?')
        #self.assertTrue(response['status'] == 'success')
        #print(response)
        #self.memory.set('history', [])
        #response = await wave.completePrompt('What is your name?')
        #self.memory.set('history', [])
        #self.assertTrue(response['status'] == 'success')
        #print(response)
        response = await wave.completePrompt('What is the square root of 2?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

    async def test_agent_w_Math(self):
        print('\n###############################################################################################')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        agent = make_agent(self.model)
        agent.addCommand(MathCommand())
        agent.addCommand(FinalAnswerCommand())
        self.memory.clear()
        result = await agent.completeTask('what is 35 times 64?')
        if type(result) == dict and 'type' in result and result['type'] == 'TaskResponse':
            self.assertTrue(result['status'] == 'success' or result['status'] == 'input_needed')
            print(result['message'])
            self.assertTrue('2240' in result['message'])
        else:
            print(f' Unknown Agent result type: {result}')
        self.end()

if __name__ == '__main__':
    model = input('template_name? ').strip()

    test = TestAlphaWave('chat')
    asyncio.run(test.test_chat())
    test = TestAlphaWave('chat_recall')
    asyncio.run(test.test_chat_recall())
    test = TestAlphaWave('reasoning')
    asyncio.run(test.test_reasoning())

    #test = TestAlphaWave('json_zeroShot')
    #asyncio.run(test.test_json_zeroShot())
    #test = TestAlphaWave('json_zeroShot_2turn')
    #asyncio.run(test.test_json_zeroShot_2turn())
    #test = TestAlphaWave('json_oneShot no repair')
    #asyncio.run(test.test_json_oneShot())
    #test = TestAlphaWave('json_oneShot_repair')
    #asyncio.run(test.test_json_oneShot_repair())
    #test = TestAlphaWave('test_planning')
    #asyncio.run(test.test_planning())
    test = TestAlphaWave('test_yplanning')
    asyncio.run(test.test_yplanning())
    test = TestAlphaWave('test_pre_agent')
    asyncio.run(test.test_pre_agent())
    test = TestAlphaWave('test_agent')
    asyncio.run(test.test_agent_w_Math())
