from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union
from uuid import uuid4
from pyee import AsyncIOEventEmitter
import asyncio, copy
import json
import sys
import traceback
from promptrix.promptrixTypes import Message, PromptFunctions, PromptSection, PromptMemory, Tokenizer
from promptrix.FunctionRegistry import  FunctionRegistry
from promptrix.GroupSection import GroupSection
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from promptrix.ConversationHistory import ConversationHistory
from promptrix.TextSection import TextSection
from promptrix.Utilities import Utilities
from promptrix.TemplateSection import TemplateSection
from promptrix.Prompt import Prompt

from alphawave.AlphaWave import AlphaWave, AlphaWaveOptions
from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse, Validation, PromptResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator

from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand, CommandSchema
from alphawave_agents.AgentCommandValidator import AgentCommandValidator
import promptrix.Utilities
from alphawave_agents.agentTypes import Command, AgentThought, AgentThoughtSchema
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

PromptInstructionSection = TemplateSection("\n".join([
    "Return a JSON object with your thoughts and the next command to perform",
    "Only respond with the JSON format below and based your plan on the commands above",
    "Response Format:",
    '{"thoughts":{"thought":"<your current thought>","reasoning":"<self reflect on why you made this decision>","plan":"- short bulleted\n- list that conveys\n- long-term plan"},"command":{"name":"<command name>","input":{"<name>":"<value>"}}}'
]), 'system')

PromptInstructionSection_py = TemplateSection(\
"""
Reason step by step how to answer the users query below.
Return a JSON object with your thoughts and the next command to perform, using the following format and available commands.
Response Format:

{\"reasoning\":\"<reflections on how to construct an answerto the user query>\",\"plan\":\"concise plan for constructing answer\",\"command\":{\"name\":\"<command name of next action to perform>\",\"input\":{\"<key>\":\"<value>\"}}</s>"
"""
                                              , 'system')

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

def update_dataclass(instance, **kwargs):
    for key, value in kwargs.items():
        if key in instance and value is not None:
            instance[key] =  value


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
        }
        update_dataclass(self._options, **asdict(options))
        self._events: EventEmitter = AsyncIOEventEmitter()

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
        # Dispatch to child agent if needed
        step = 0
        state = self.get_agent_state(agentId)
        if 'child' in state and state['child'] is not None:
            childAgent = self.getCommand(state['child'].title)
            
            response = await childAgent.completeTask(input, state['child'].agentId)
            if response.status != 'success':
                return response

            # Delete child and save state
            del state.child
            self.setAgentState(state, agentId)

            # Use agents response as input to the next step
            # We don't know how many steps the child agent took, so we'll just assume it took one
            stepInput = response.message
            step = 1
            executeInitialThought = False

        # Start main task loop
        while step < self.options['max_steps']:
            # Wait for step delay
            
            if step > 0 and self.options['step_delay'] > 0:
                sys.stdout.flush()
                await asyncio.sleep(self.options['step_delay']/1000)

            # Execute next step
            result = await self.execute_next_step(stepInput, agentId, executeInitialThought)
            if isinstance(result, str):
                stepInput = result
            else:
                return result

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

    # Agent as Commands
    async def execute(self, input: AgentCommandInput, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer):
        # Initialize the agents state
        agentId = input.agentId
        state = self.get_agent_state(agentId)
        state['context'] = input.input
        self.set_agent_state(state, agentId)

        # Start the task
        return await self.completeTask(None, agentId)

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
            sections = [agent_prompt]
            sections.append(AgentCommandSection(self._commands))
            pis = PromptInstructionSection
            if not (self._options['prompt_options']['model']).lower().startswith('gpt'):
                pis = PromptInstructionSection_py
            sections.append(pis)
            prompt = Prompt([
                GroupSection(sections, 'system'),
                ConversationHistory(history_variable, 1.0, True)
            ])
            if input:
                prompt.sections.append(TextSection(input, 'user', -1, True, '\n', 'user'))
                # Ensure input variable is set otherwise the history will be wrong.
                self.memory.set(self.options['input_variable'], input)


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
                validator = AgentCommandValidator(self._commands, self._options['prompt_options']['model'])

                # Create a wave for the prompt
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
                if response['status'] != 'success':
                    return {
                        'type': "TaskResponse",
                        'status': response['status'],
                        'message': response['message']
                    }

            # Get agents thought and execute command
            message = response['message']
            thought = message['content']
            self._events.emit('newThought', thought)
            result = await self.execute_command(state, thought)

            # Check for task result and error
            task_response = result if isinstance(result, dict) and result.get('type') == 'TaskResponse' else None
            if task_response:
                if task_response['status'] in ['error', 'invalid_response', 'rate_limited', 'too_many_steps', 'too_long']:
                    return task_response

            # Update history
            history = self.memory.get(history_variable) or []
            if input:
                #history.append({'role': 'user', 'content': input})
                pass
            #history.append({'role': 'assistant', 'content': json.dumps(thought)})
            self.memory.set(history_variable, history)

            # Save the agents state
            state['totalSteps'] += 1
            self.set_agent_state(state, agent_id)

            # Return result
            return task_response if task_response else Utilities.to_string(self.tokenizer, result)
        except Exception as err:
            return {
                'type': "TaskResponse",
                'status': "error",
                'message': str(err)
            }

    async def execute_command(self, state: AgentState, thought: AgentThought):
        # Get command
        command_name = thought['command']['input']['name'] if thought['command']['name'] == 'character' else thought['command']['name']
        command = self._commands.get(command_name, None) or {}
        input = thought['command']['input'] or {}
        if isinstance(command, Agent):
            # Pass control to child agent
            agentId = str(uuid4())
            childAgent = command
            response = await childAgent.execute(input['input'], self.memory, self.functions, self.tokenizer)
            if response.status == 'success':
                # Just return the response message since agent completed without additional input
                return response.message
            elif response.status == 'input_needed':
                # Remember that we're talking to the agent
                state.child = {
                    'title': command_name,
                    'agentId': agentId
                }
                return response
            else:
                # Return the response since the agent failed
                return response
        else:
            # Execute command and return result
            response = command.execute(input, self.memory, self.functions, self.tokenizer)
            if 'coroutine' in str(type(response)).lower():
                return await response
            return response

