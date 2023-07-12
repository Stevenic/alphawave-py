from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union
from uuid import uuid4
from pyee import AsyncIOEventEmitter
import asyncio, copy
import json
import sys
import time
import traceback
from promptrix.promptrixTypes import Message, PromptFunctions, PromptSection, PromptMemory, Tokenizer
from promptrix.FunctionRegistry import  FunctionRegistry
from promptrix.GroupSection import GroupSection
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from promptrix.ConversationHistory import ConversationHistory
from promptrix.TextSection import TextSection
from promptrix.AssistantMessage import AssistantMessage
from promptrix.UserMessage import UserMessage
from promptrix.Utilities import Utilities
from promptrix.TemplateSection import TemplateSection
from promptrix.Prompt import Prompt

from alphawave.AlphaWave import AlphaWave, AlphaWaveOptions
from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse, Validation, PromptResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator

from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand, CommandSchema
from alphawave_agents.AgentCommandValidator import AgentCommandValidator
import promptrix.Utilities
from alphawave_agents.agentTypes import Command, AgentThought
from alphawave_agents.AgentCommandSection import AgentCommandSection

from typing import Dict, Any
from pydantic import BaseModel, Field
import traceback

class AgentCommandSchema(BaseModel):
    type: str = Field("object", description="an agent that can perform a task")
    title: str = "Agent"
    properties: Dict[str, Any] = {
        "input": {
            "type": "string",
            "description": "input for command",
        }
    }
    required: list[str] = ["input"]



@dataclass
class AgentOptions:
    client: PromptCompletionClient
    context_variable: Optional[str] = None
    prompt: Union[str, list[str], PromptSection] = None
    prompt_options: PromptCompletionOptions = None
    agent_variable: Optional[str] = None
    functions: Optional[PromptFunctions] = None
    history_variable: Optional[str] = None
    initial_thought: Optional[AgentThought] = None
    input_variable: Optional[str] = None
    logRepairs: Optional[bool] = None
    max_history_messages: Optional[int] = None
    max_repair_attempts: Optional[int] = None
    max_steps: Optional[int] = None
    memory: Optional[PromptMemory] = None
    retry_invalid_responses: Optional[bool] = None
    step_delay: Optional[int] = None
    tokenizer: Optional[Tokenizer] = None
    syntax: Optional[str] = 'JSON'

@dataclass
class ConfiguredAgentOptions:
    agent_variable: str = None
    client: PromptCompletionClient = None
    context_variable: str = None
    functions: PromptFunctions = None
    history_variable: str = None
    initial_thought: Optional[AgentThought] = None
    input_variable: str = None
    logRepairs: bool = None
    max_history_messages: int = 10
    max_repair_attempts: int = 3
    max_steps: int = 5
    memory: PromptMemory = None
    prompt: Union[str, list[str], PromptSection] = None
    prompt_options: PromptCompletionOptions = None
    retry_invalid_responses: bool = False
    step_delay: int = 5
    tokenizer: Tokenizer = None
    syntax: Optional[str] = 'JSON'

def update_dataclass(instance, source):
    for key, value in source.items():
        if key in instance and value is not None:
            instance[key] =  value

##
PromptInstructionSectionJSON = TemplateSection(\
"""
Reason step-by-step about the user task.
If the task can be completed from known fact, reasoning, or has been completed earlier, respond immediately. 
Otherwise, develop a concise plan for completing the task. Base your plan on known fact, reasoning, prior results, and the available commands.
2. Select the first step of that plan as the next command to perform. 
3. Respond with:
  - your reasoning 
  - the JSONformat shown above for that next command, instantiating actual inputs where indicated
""", 'system')
PromptOneShotJSON = [
    UserMessage("What is 35 * 64?"),
    AssistantMessage("""I will use the math command.
{"command": "math", "inputs":{"code":"35*64"}}""")]


PromptInstructionSectionTOML = TemplateSection(\
"""
Reason step-by-step about the user task:
1. Develop a concise plan for finding an answer and showing it to the user. Base your plan on known fact, reasoning, and the available commands.
2. Select the first step of that plan as the next command to perform. 
3. Respond with:
  - your reasoning 
  - the JSONformat shown above for that next command, substituting your inputs where indicated
""", 'system')

PromptOneShotTOML = [
    UserMessage("What is 35 * 64?"),
    AssistantMessage("""I will use the math command.
[RESPONSE]
command = "math"
inputs.code = "35*64"
[STOP]
""")]


@dataclass
class AgentCommandInput:
    agentId: str
    input: str


@dataclass
class AgentState:
    totalSteps: int
    context: Optional[str] = None
    child: Optional[Dict[str, str]] = None


class Agent(SchemaBasedCommand):
    def __init__(self, options: AgentOptions, title: Optional[str] = None, description: Optional[str] = None):
        super().__init__(CommandSchema, title, description)
        self._commands: Dict[str, Command] = {}
        self._options: ConfiguredAgentOptions = {
            'agent_variable': 'agent',
            'client': None,
            'context_variable': 'context',
            'functions': FunctionRegistry(),
            'history_variable': 'history',
            'initial_thought': None,
            'input_variable': 'input',
            'logRepairs': False,
            'max_history_messages': 10,
            'max_repair_attempts': 3,
            'max_steps': 5,
            'memory': VolatileMemory(),
            'prompt': None,
            'prompt_options': None,
            'retry_invalid_responses': False,
            'step_delay': 5,
            'tokenizer': GPT3Tokenizer(),
            'syntax': 'JSON'
        }
        update_dataclass(self._options, options.__dict__)
        #self._options = self._options.__dict__
        self._events: EventEmitter = AsyncIOEventEmitter()
        self.top_level_task=None

    @property
    def client(self):
        return self._options.client

    @property
    def events(self):
        return self._events

    @property
    def functions(self):
        return self._options['functions']

    @property
    def memory(self):
        return self._options['memory']

    @property
    def options(self):
        return self._options

    @property
    def tokenizer(self):
        return self._options['tokenizer']

    # Command management

    def addCommand(self, command: Command):
        if command.title in self._commands:
            raise ValueError(f"A command with the title {command.schema.title} already exists.")
        self._commands[command.title] = command
        return self

    def getCommand(self, title: str):
        return self._commands.get(title)

    def hasCommand(self, title: str):
        return title in self._commands

    async def completeTask(self, input: Optional[str] = None, agentId: Optional[str] = None, executeInitialThought: bool = False):
      try:
        # Initialize the input to the next step
        stepInput = input if input is not None else self.memory.get(self.options['input_variable'])
        if self.top_level_task is None:
            self.top_level_task = stepInput

        # Dispatch to child agent if needed
        step = 0
        state = self.get_agent_state(agentId)
        # Start main task loop
        while step < self.options['max_steps']:
            # Wait for step delay to prevent gpt overrun
            if step > 0 and self._options['step_delay'] > 0:
                sys.stdout.flush()
                time.sleep(self._options['step_delay']/100) # assumes step delay is in seconds
            # Execute next step
            #print(f'***** Agent completeTask calling execute_next_step {stepInput}')
            result, ran_command = await self.execute_next_step(stepInput, agentId)
            #print(f'***** Agent completeTask return from execute_next_step {ran_command}, {result}')
            if ran_command:
                # check for command in response from command
                # Create command validator
                #validator = AgentCommandValidator(self._commands, self._options['client'], self._options['prompt_options'].model,
                #                                  self._options['syntax'], memory=self.memory, history_variable=history_variable)
                # just a stub, need to finish if good idea
                return {'type':'TaskResponse', 'status':'success', 'message':result}
            else:
                return {'type':'TaskResponse', 'status':'success', 'message':result}

            step += 1
            executeInitialThought = False
        # Return too many steps
        return {
            "type": "TaskResponse",
            "status": "too_many_steps",
            "message": "The current task has taken too many steps."
        }
      except Exception as e:
        traceback.print_exc()
        pass

    def get_agent_state(self, agentId: Optional[str] = None):
        key = f'{self.options.agent_variable}-{agentId}' if agentId else self._options['agent_variable']
        state = self.memory.get(key) or {}
        if 'totalSteps' not in state:
            state['totalSteps'] = 0
        return state

    def set_agent_state(self, state: AgentState, agentId: Optional[str] = None):
        key = f"{self.options['agent_variable']}-{agentId}" if agentId else self.options['agent_variable']
        self.memory.set(key, state)

    def get_agent_history_variable(self, agentId: Optional[str] = None):
        return f"{self._options['history_variable']}-{agentId}" if agentId else self._options['history_variable']


    async def execute_next_step(self, input: Optional[str] = None, agent_id: Optional[str] = None, execute_initial_thought: bool = False):
        try:
            #print(f'***** Agent execute_next_step input {input}')
            state = self.get_agent_state(agent_id)
            # Create agents prompt section
            if isinstance(self._options['prompt'], list):
                agent_prompt = TemplateSection('\n'.join(self._options['prompt']), 'system')
            elif isinstance(self._options['prompt'], dict):
                agent_prompt = self._options['prompt']
            else:
                agent_prompt = TemplateSection(self._options['prompt'], 'system')
            
            # Ensure the context variable is set
            if 'context' in state:
                self.memory.set(self.options['context_variable'], state['context'])

            # Create prompt
            history_variable = self.get_agent_history_variable(agent_id)
            try:
                sections = [agent_prompt]
                sections.append(AgentCommandSection(self._commands, one_shot=True, syntax=self._options['syntax']))
                
                if self._options['syntax'] == 'JSON':
                    pis = PromptInstructionSectionJSON
                    pos = PromptOneShotJSON
                else:
                    pis = PromptInstructionSectionTOML
                    pos = PromptOneShotTOML
                    
                sections.append(pis)
                sections.extend(pos)
                prompt = Prompt([
                    GroupSection(sections, 'system'),
                    ConversationHistory(history_variable, 1.0, True)
                ])
                if input:
                    #print(f'***** Agent append Text Section {input}')
                    prompt.sections.append(TextSection(input, 'user', -1, True, '\n', 'user'))
                    # Ensure input variable is set otherwise the history will be wrong.
                    self.memory.set(self.options['input_variable'], input)
            except Exception as e:
                tracebak.print_exc()

            if execute_initial_thought and self._options['initial_thought']:
                # Just use initial thought as response
                # - This is used when agents are being called as commands.
                response = {
                    'status': 'success',
                    'message': {'role': 'assistant', 'content': self._options['initial_thought']}
                }
            else:
                # Add initial thought to history
                if state['totalSteps'] == 0 and self._options['initial_thought']:
                    history = self.memory.get(history_variable) or []
                    message = {'role': 'assistant', 'content': json.dumps(self._options['initial_thought'])}
                    history.append(message)
                    self.memory.set(history_variable, history)

                # Create command validator
                validator = AgentCommandValidator(self._commands, self._options['client'], self._options['prompt_options'].model, self._options['syntax'], memory=self.memory, history_variable=history_variable)

                # Create a wave for the prompt
                ### this is the ONLY wave call in Agent. 
                wave = AlphaWave(
                    client = self._options['client'],
                    prompt = prompt,
                    prompt_options = self._options['prompt_options'],
                    functions = self._options['functions'],
                    history_variable = history_variable,
                    input_variable = self._options['input_variable'],
                    max_history_messages = self._options['max_history_messages'],
                    max_repair_attempts = self._options['max_repair_attempts'],
                    memory = self._options['memory'],
                    tokenizer = self._options['tokenizer'],
                    logRepairs = self._options['logRepairs'],
                    validator = validator
                )

                # Complete the prompt
                max_attempts = 2 if self._options['retry_invalid_responses'] else 1
                for attempt in range(max_attempts):
                    response = await wave.completePrompt()
                    
                    if response['status'] != 'invalid_response':
                        break

                # Ensure response succeeded
                # Note at this point AgentCommandValidator has approved response
                if response['status'] != 'success':
                    return {
                        'type': "TaskResponse",
                        'status': response['status'],
                        'message': response['message']
                    }, False

            # Get agents thought and execute command
            message = response['message']
            thought = message['content']

            #test if LLM wants to execute command. If no, just return!
            if type(thought) != dict or 'command' not in thought:
                #print(f'***** Agent execute_next_step returning no next command')
                return thought, False  # False means don't keep going, no command to run
            self._events.emit('newThought', thought)
            #print(f'***** Agent execute_next_step calling execute_command state {state}, thought {thought}')

            command_name = thought['command']
            command_input = str(thought['inputs'] or '')
            result = await self.execute_command(state, thought)
            #print(f'command_result \n{result}\n')
            # Check for task result and error
            task_response = result if isinstance(result, dict) and result.get('type') == 'TaskResponse' else None
            if task_response:
                if task_response['status'] in ['error', 'invalid_response', 'rate_limited', 'too_many_steps', 'too_long']:
                    #print(f'***** Agent execute_next_step fail {task_response}')
                    return task_response, False

            # Update history
            return_msg = task_response['message'] if task_response else result
            history = self.memory.get(history_variable) or []
            history.append({'role': 'assistant', 'content': command_name+' '+str(input)+' result:\n'+str(return_msg)})
            self.memory.set(history_variable, history)

            # Save the agents state
            state['totalSteps'] += 1
            self.set_agent_state(state, agent_id)

            # Return result
            #return_msg = task_response if task_response else Utilities.to_string(self.tokenizer, result)
            #print(f'***** Agent execute_next_step returning {return_msg}')
            return return_msg, True
        except Exception as err:
            traceback.print_exc()
            return {
                'type': "TaskResponse",
                'status': "error",
                'message': str(err)
            }, False

    async def execute_command(self, state: AgentState, thought: AgentThought):
        # a command to execute
        command_name = thought['command']
        command = self._commands.get(command_name, None) or {}
        input = thought['inputs'] or {}
        # Execute command and return result
        #print(f'***** Agent execute_command  {command_name}, {input}')
        response = await command.execute(input, self.memory, self.functions, self.tokenizer)
        if 'coroutine' in str(type(response)).lower():
            return await response
        #print(f'***** Agent execute_command {command_name}\n{response}\n')
        return response

