import os
import time
import asyncio
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.OpenAIClient import OpenAIClient
from alphawave.OSClient import OSClient
from alphawave_agents.Agent import Agent, AgentOptions
from alphawave_pyexts.SearchCommand import SearchCommand
from alphawave_agents.AskCommand import AskCommand
from alphawave_agents.FinalAnswerCommand import FinalAnswerCommand

# Create an OpenAI client
client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
model = 'gpt-3.5-turbo',

#create OS client
#client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
#model = 'wizardLM'

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
        temperature = 0.5,
        max_input_tokens = 1200,
        max_tokens = 800,
    ),
    initial_thought={
        "thoughts": {
            "thought": "I want the user to know I want to help.",
            "reasoning": "",
            "plan": "- use the ask command to ask the user what he would like me to figure out"
        },
        "command": {
            "name": "ask",
            "input": {"question": initial_prompt}
        }
    },
    step_delay=5000,
    max_steps=50
)

# Create an agent
agent = Agent(options = agent_options)

# Add core commands to the agent
agent.addCommand(AskCommand())
agent.addCommand(SearchCommand(client, model))
agent.addCommand(FinalAnswerCommand())


# Listen for new thoughts
agent.events.on('newThought', lambda thought: print(f"[{thought['thoughts']['thought']}]"))

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
            await chat(result['message'])
        else:
            if result['message']:
                print(f"{result['status']}: {result['message']}")
            else:
                print(f"A result status of '{result['status']}' was returned.")
            exit()

# Start chat session
asyncio.run(chat(initial_prompt))
