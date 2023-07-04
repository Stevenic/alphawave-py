import os
import time
import asyncio
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.MemoryFork import MemoryFork
from alphawave.OpenAIClient import OpenAIClient
from alphawave.OSClient import OSClient
from alphawave_agents.Agent import Agent, AgentOptions
from alphawave_pyexts.SearchCommand import SearchCommand
from alphawave_agents.AskCommand import AskCommand
from alphawave_agents.FinalAnswerCommand import FinalAnswerCommand

# Create an OpenAI client
search_client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
search_model = 'gpt-3.5-turbo'

#create OS client
client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
#model = 'wizardLM'
#model = 'vicuna_v1.1'
#model ='guanaco'
#model='mpt_instruct'
model=input('PromptTemplate? ')
initial_prompt = \
"""
Hi! I'm here to help.
"""

agent_options = AgentOptions(
    client = client,
    prompt=[
        "You are a helpful information bot "
    ],
    prompt_options=PromptCompletionOptions(
        completion_type = 'chat',
        model = model,
        temperature = 0.1,
        max_input_tokens = 1200,
        max_tokens = 200,
    ),
    initial_thought=None,
    step_delay=1000,
    max_steps=50,
    logRepairs=True
)

# Create an agent
agent = Agent(options = agent_options)

# Add core commands to the agent
agent.addCommand(SearchCommand(search_client, search_model))
agent.addCommand(FinalAnswerCommand())


# Listen for new thoughts
#agent.events.on('newThought', lambda thought: print(f"[{thought['thoughts']['thought']}]"))

# Define main chat loop
async def chat(bot_message=None):
    # Show the bot's message
    if bot_message:
        print(f"{bot_message}")

    # Prompt the user for input
    user_input = input('User: ')
    
    # Check if the user wants to exit the chat
    if user_input.lower() == 'exit':
        exit()
    else:
        # Route user's message to the agent
        result = await agent.completeTask(user_input)
        if result['status'] in ['success', 'input_needed']:
            if result['message']:
                print(f"****** SearchCommandAgentTest chat result {result['status']}: {result['message']}")
            else:
                print(f"***** SearchCommandAgentTest chat result A result status of '{result['status']}' was returned.")
            await chat(result['message'])
        else:
            if result['message']:
                print(f"****** SearchCommandAgentTest chat result {result['status']}: {result['message']}")
            else:
                print(f"***** SearchCommandAgentTest chat result A result status of '{result['status']}' was returned.")
            exit()

# Start chat session
asyncio.run(chat(initial_prompt))
