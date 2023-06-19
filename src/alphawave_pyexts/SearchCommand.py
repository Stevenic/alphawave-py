from typing import Any, Dict, List
from colorama import Fore, Style
from dataclasses import dataclass, asdict
import copy
from promptrix.SystemMessage import SystemMessage
from promptrix.VolatileMemory import VolatileMemory
from alphawave_agents.agentTypes import TaskResponse
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
from alphawave_agents.SchemaBasedCommand import CommandSchema as sbcCommandSchema
from alphawave_pyexts.llmsearch import search_service
import json

@dataclass
class CommandSchema(sbcCommandSchema):
    schema_type: str
    title: str
    description: str
    properties: Dict[str,Dict[str,str]]
    required: list[str]
    returns: str


search_schema = CommandSchema(
    schema_type="object",
    title="search",
    description="web search for provided query",
    properties={
        "query": {
            "type": "string",
            "description": "web query string"
            }
    },
    required=['query'],
    returns='search result'
)


class SearchCommand(SchemaBasedCommand):
    def __init__(self, client, model, title=None, name=None, description=None, return_urls=False, logResponse=False, max_chars=1500):
        super().__init__(search_schema, title, description)
        self.client = client
        self.model = model
        self.return_urls = return_urls
        self.max_chars = max_chars
        self.logResponse = logResponse
        
    async def execute(self, input: input, memory: Any, functions: Any, tokenizer: Any) -> Any:
        try:
            if type(input) == dict and 'query' in input:
                query = input['query']
                memory = VolatileMemory()  # don't let anything bleed back to surrounding task
                response =await search_service.run_chat(self.client, query, self.model, memory, functions, tokenizer, self.max_chars)
                sc_text = ''
                sc_urls = []
                if type(response) is list:
                    for item in response:
                        if type(item) == dict and 'url' in item:
                            sc_urls.append(item['url'])
                        if type(item) == dict and 'text' in item:
                            sc_text += '\n'+(item['text'])
                    sc_text = sc_text[:max(len(sc_text)-1, self.max_chars)]
                if self.return_urls:
                    return {'status':'success', 'message':{'text': sc_text, 'urls':sc_urls}}
                else:
                    return {'status':'success', 'message':sc_text}
                    
        except Exception as err:
            message = str(err)
            return asdict(TaskResponse('TaskResponse', 'error', message))
