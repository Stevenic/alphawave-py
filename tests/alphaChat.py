import os
#from dotenv import load_dotenv
from pathlib import Path
import readline
from alphawave.AlphaWave import AlphaWave
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.OpenAIClient import OpenAIClient
from alphawave.OSClient import OSClient
import promptrix
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.UserMessage import UserMessage
import asyncio

# Read in .env file.
env_path = Path('..') / '.env'
#load_dotenv(dotenv_path=env_path)

# Create an OpenAI or AzureOpenAI client
client = OpenAIClient(apiKey=os.getenv("OPENAI_API_KEY"), logRequests= True)
#client = OSClient(apiKey=os.getenv("OPENAI_API_KEY"), logRequests= True)

# Create a wave
wave = AlphaWave(
    client=client,
    prompt=Prompt([
        SystemMessage('You are an AI assistant that is friendly, kind, and helpful', 50),
        ConversationHistory('history', 1.0),
        UserMessage('{{$input}}', 450)
    ]),
    prompt_options=PromptCompletionOptions(
        completion_type = 'chat',
        model = 'gpt-3.5-turbo',
        temperature = 0.7,
        max_input_tokens= 2000,
        max_tokens= 1000
    )
)

# Define main chat loop
async def chat(bot_message=None):
    # Show the bots message
    if bot_message:
        print(f"\033[32m{bot_message}\033[0m")

    # Prompt the user for input
    user_input = input('User: ')
    # Check if the user wants to exit the chat
    if user_input.lower() == 'exit':
        # Exit the process
        exit()
    else:
        # Route users message to wave
        result = await wave.completePrompt(user_input)
        if result['status'] == 'success':
            await chat(result['message']['content'])
        else:
            if result['message']:
                print(f"{result['status']}: {result['message']}")
            else:
                print(f"A result status of '{result['status']}' was returned.")
            # Exit the process
            exit()

# Start chat session
asyncio.run(chat("Hello, how can I help you?"))
