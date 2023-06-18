from typing import Any, Dict, List
from colorama import Fore, Style
from dataclasses import dataclass, asdict
import copy
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
from alphawave_agents.SchemaBasedCommand import CommandSchema as sbcCommandSchema
from promptrix.SystemMessage import SystemMessage
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
    def __init__(self, client, model, title=None, name=None, description=None):
        super().__init__(search_schema, title, description)
        self.client = client
        self.model = model

    async def execute(self, input: input, memory: Any, functions: Any, tokenizer: Any) -> Any:
        try:
            print(f'***** SearchCommand input {input}')
            if type(input) == dict and 'query' in input:
                query = input['query']
                response =await search_service.run_chat(self.client, query, self.model, memory, functions, tokenizer)
                if type(response) is list:
                    print('Search Command result\n')
                    for item in response:
                        print(json.dumps(item, indent=2))
                return {'status':'success', 'message':response}
        except Exception as err:
            message = str(err)
            print(f'***** SearchCommand error {str(err)}')
            return asdict(TaskResponse('TaskResponse', 'error', message))
