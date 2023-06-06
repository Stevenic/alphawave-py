from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from uuid import uuid4
from pyee import AsyncIOEventEmitter
import asyncio

from promptrix.promptrixTypes import Message, PromptFunctions, PromptSection, PromptMemory, Tokenizer
from promptrix.FunctionRegistry import  FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from promptrix.Utilities import Utilities
from promptrix.TemplateSection import TemplateSection
from promptrix.Prompt import Prompt

from alphawave.AlphaWave import AlphaWave
from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse, Validation, PromptResponseValidator

from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
from alphawave_agents.AgentCommandValidator import AgentCommandValidator
import promptrix.Utilities
from alphawave_agents.agentTypes import Command, AgentThought, AgentThoughtSchema
from alphawave_agents.AgentCommandSection import AgentCommandSection

@dataclass
class AgentOptions:
    client: PromptCompletionClient
    context_variable: Optional[str] = None
    prompt: Union[str, List[str], PromptSection]
    prompt_options: PromptCompletionOptions
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
    agent_variable: str
    client: PromptCompletionClient
    context_variable: str
    functions: PromptFunctions
    history_variable: str
    initial_thought: Optional[AgentThought]
    input_variable: str
    logRepairs: bool
    max_history_messages: int
    max_repair_attempts: int
    max_steps: int
    memory: PromptMemory
    prompt: Union[str, List[str], PromptSection]
    prompt_options: PromptCompletionOptions
    retry_invalid_responses: bool
    step_delay: int
    tokenizer: Tokenizer


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
        super().__init__(AgentCommandSchema, title, description)
        self._commands: Dict[str, Command] = {}
        self._options: ConfiguredAgentOptions = {
            'agent_variable': 'agent',
            'context_variable': 'context',
            'functions': FunctionRegistry(),
            'history_variable': 'history',
            'input_variable': 'input',
            'logRepairs': False,
            'max_history_messages': 10,
            'max_repair_attempts': 3,
            'max_steps': 5,
            'memory': VolatileMemory(),
            'retry_invalid_responses': False,
            'step_delay': 0,
            'tokenizer': GPT3Tokenizer(),
            **options
        }
        self._events: EventEmitter = EventEmitter()

    @property
    def client(self):
        return self._options.client

    @property
    def events(self):
        return self._events

    @property
    def functions(self):
        return self._options.functions

    @property
    def memory(self):
        return self._options.memory

    @property
    def options(self):
        return self._options

    @property
    def tokenizer(self):
        return self._options.tokenizer

    # Command management

    def addCommand(self, command: Command):
        if command.title in self._commands:
            raise ValueError(f'A command with the title "{command.title}" already exists.')
        self._commands[command.title] = command
        return self

    def getCommand(self, title: str):
        return self._commands.get(title)

    def hasCommand(self, title: str):
        return title in self._commands

    # Task execution

    async def completeTask(self, input: Optional[str] = None, agentId: Optional[str] = None, executeInitialThought: bool = False):
        # Initialize the input to the next step
        stepInput = input if input is not None else self.memory.get(self.options.input_variable)

        # Dispatch to child agent if needed
        step = 0
        state = self.getAgentState(agentId)
        if state.child:
            childAgent = self.getCommand(state.child.title)
            response = await childAgent.completeTask(input, state.child.agentId)
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
        while step < self.options.max_steps:
            # Wait for step delay
            if step > 0 and self.options.step_delay > 0:
                await asyncio.sleep(self.options.step_delay)

            # Execute next step
            result = await self.executeNextStep(stepInput, agentId, executeInitialThought)
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

    # Agent as Commands
    async def execute(self, input: AgentCommandInput, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer):
        # Initialize the agents state
        agentId = input.agentId
        state = self.get_agent_state(agentId)
        state.context = input.input
        self.set_agent_state(state, agentId)

        # Start the task
        return await self.complete_task(None, agentId)

    def get_agent_state(self, agentId: Optional[str] = None):
        key = f'{self.options.agent_variable}-{agentId}' if agentId else self.options.agent_variable
        state = self.memory.get(key) or {}
        if 'totalSteps' not in state:
            state['totalSteps'] = 0
        return state

    def set_agent_state(self, state: AgentState, agentId: Optional[str] = None):
        key = f'{self.options.agent_variable}-{agentId}' if agentId else self.options.agent_variable
        self.memory.set(key, state)

    def get_agent_history_variable(self, agentId: Optional[str] = None):
        return f'{self.options.history_variable}-{agentId}' if agentId else self.options.history_variable

    async def execute_next_step(self, input: Optional[str] = None, agentId: Optional[str] = None):
        # The rest of the method implementation goes here...
        pass

    async def execute_command(self, state: AgentState, thought: AgentThought):
        # Get command
        command = self._commands.get(thought.command.name)
        input = thought.command.input or {}
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
                    'title': thought.command.name,
                    'agentId': agentId
                }
                return response
            else:
                # Return the response since the agent failed
                return response
        else:
            # Execute command and return result
            return await command.execute(input, self.memory, self.functions, self.tokenizer)

    async def completeTask(self, input: Optional[str] = None, agentId: Optional[str] = None, executeInitialThought: bool = False):
        # Initialize the input to the next step
        stepInput = input if input is not None else self.memory.get(self.options.input_variable)

        # Dispatch to child agent if needed
        step = 0
        state = self.getAgentState(agentId)
        if state.child:
            childAgent = self.getCommand(state.child.title)
            response = await childAgent.completeTask(input, state.child.agentId)
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
        while step < self.options.max_steps:
            # Wait for step delay
            if step > 0 and self.options.step_delay > 0:
                await asyncio.sleep(self.options.step_delay)

            # Execute next step
            result = await self.executeNextStep(stepInput, agentId, executeInitialThought)
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
