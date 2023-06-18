# AlphaWave
minor bug fixes, OS Client now correctly handles host, port, temperature, top_p, max_tokens

New: SearchCommand will search the web. You will need a google api key.
See tests/SearchCommandAgentTest.py

AlphaWave is a very opinionated client for interfacing with Large Language Models (LLM). It uses [Promptrix](https://github.com/Stevenic/promptrix) for prompt management and has the following features:

- Supports calling OpenAI and Azure OpenAI hosted models out of the box but a simple plugin model lets you extend AlphaWave to support any LLM.
- Supports OS LLMs through an OSClient. Currently assumes a server on port 5004, see details below.
- Promptrix integration means that all prompts are universal and work with either Chat Completion or Text Completion API's.
- Automatic history management. AlphaWave manages a prompts conversation history and all you have todo is tell it where to store it. It uses an in-memory store by default but a simple plugin interface (provided by Promptrix) lets you store short term memory, like conversation history, anywhere.
- State-of-the-art response repair logic. AlphaWave lets you provide an optional "response validator" plugin which it will use to validate every response returned from an LLM. Should a response fail validation, AlphaWave will automatically try to get the model to correct its mistake. More below...

## Automatic Response Repair
A key goal of AlphaWave is to be the most reliable mechanisms for talking to an LLM on the planet. If you lookup the wikipedia definition for Alpha Waves you see that it's believed that they may be used to help predict mistakes in the human brain. One of the key roles of the AlphaWave library is to help automatically correct for mistakes made by an LLM, leading to more reliable output. It can correct for everything from hallucinations to just malformed output. It does this by using a series of techniques.

First it uses validation to programmatically verify the LLM's output. This would be the equivalent of a "guard" in other libraries like LangChain. When a validation fails, AlphaWave immediately forks the conversation to isolate the mistake. This is critical because the last thing you want to do is promote a mistake/hallucination to the conversation history as the LLM will just double down on the mistake. They are primarily pattern matchers.

Once AlphaWave has isolated the mistake, it will attempt to get the model to repair the mistake itself. It uses a process called "feedback" which simply tells the model the mistake it made and asks it to correct it. For GPT-4 this works more often then not in 1 turn. For the other models it sometimes works but it depends on the type of mistake. AlphaWave will even ask the model to slow down and think step-by-step on the last try, to give it every shot at fixing itself.

If the LLM can correct its mistake, AlphaWave will delete the conversation fork, write the corrected response to the conversation history, and move forward as if nothing ever happened. For GPT-4, you should be able to make several hundred sequential model calls before running into a sequence that can't be repaired.

In the event that the model isn't able to repair itself, a result with a status of `invalid_response` will be returned and the app can either abort the task or give it one more go. For well defined prompts and tasks I'd recommend given it one more go. The reason for that is that, if you've made it hundreds of model calls without it making a mistake, the odds of it making a mistake if you simply try again are low. You just hit the stochastic nature of talking to LLMs.

So why even use "feedback" at all if retrying can work? It doesn't always work. Some mistakes, especially hallucinations, the LLM will make over and over again. They need to be confronted with their mistake and then they will happily correct it. You need both appproaches, feedback & retry, to build a system that's as reliable as possible.

## Installation
To get started, you'll want to install the latest versions of both AlphaWave and Promptrix. Pip should pull both if you just install alphawave

```bash
pip install alphawave

```

## Basic Usage
You'll need to import a couple of components from "alphawave", along with the various prompt parts you want to use from "promptrix". Here's a super simple wave that creates a basic ChatGPT like bot:

```python
import os
from pathlib import Path
import readline
from alphawave.AlphaWave import AlphaWave
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.OpenAIClient import OpenAIClient
import promptrix
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.UserMessage import UserMessage
import asyncio

# Create an OpenAI or AzureOpenAI client
client = OpenAIClient(apiKey=os.getenv("OPENAI_API_KEY"))

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
        temperature = 0.9,
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
            print(result)
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
```

One of the key features of Promptrix is its ability to proportionally layout prompts, so this prompt has an overall budget of 2000 input tokens. It will give the `SystemMessage` up to 50 tokens, the `UserMessage` up to 450 tokens, and then the `ConversationHistory` gets 100% of the remaining tokens.

Once the prompt is formed, we just need to call `completePrompt()` on the wave to process the users input

The  parameter to wave.completePrompt is optional and the wave can also take input directly from memory, but you don't have to pass prompts input. You can see in the example that if the prompt doesn't reference the input via a `{{$input}}` template variable it won't use it anyway.

# Logging
if you want to see the traffic with the server, the Client constructors (OSClient and OpenAIClient) take a logRequests parameter - False by default, set it to True to see prompts and responses on the console.

# OSClient
the 'default' way to use Alphawave-py with OpenAI is to use the OpenAI client as in line 49 of the example above. 
If you want to use your own LLM, you can use instead:

```python
client = OSClient(apiKey=None)
```

The current OSClient assumes a server exists on localhost port 5004, using my own unique protocol.
Not very useful, I know.
Short term plans include: 
1. allow specification of the host and port in the client constructor
2. allow FastChat-like specification of the conversation template (user/assistant/etc). Support for this is already in the OSClient, just need to bring it out to the Constructor
3. Implementation of a FastChat compatible api. Again, this was running in a dev version of the code, just need to re-insert it now that basic port is stable.

## OSClient protocol
### OSClient sends json to the server:
```python
        server_message = {'prompt':prompt, 'temp': temp, 'top_p':top_p, 'max_tokens':max_tokens}
        smj = json.dumps(server_message)
        client_socket.sendall(smj.encode('utf-8'))
        client_socket.sendall(b'x00xff')
```
where prompt is a string containing the messages:
```python
{
  "role": "system",
  "content": "You are an AI assistant that is friendly, kind, and helpful"
}{
  "role": "user",
  "content": "Hi. How are you today?"
}
```

and the x00xff is the end of send message because I know nothing about sockets

### OSClient expects to receive from the server:
1. streaming or all at once, the text, followed by 'x00xff'
2. that's it, no return code, no json wrapper, no {role: assistant, content: str), just the response.





4. **the x00ff signals end of messages
   
