from typing import Any, Dict, List
from colorama import Fore, Style
from dataclasses import dataclass, asdict
import copy
from alphawave_agents.SchemaBasedCommand import SchemaBasedCommand
from promptrix.SystemMessage import SystemMessage
from llmsearch import search_service

search_schema = CommandSchema(
    schema_type="object",
    title="search",
    description="web search for provided query",
    properties={
        "query": {
            "type": "string",
            "description": "web query string"
            }
        }
    required: {'query']
    }
)


class SearchCommand(SchemaBasedCommand):
    def __init__(self, client, model, name, description=None):
        super().__init__(search_schema, search_schema.title, search_schema.description)


    def execute(self, input: input, memory: Any, functions: Any, tokenizer: Any) -> Any:
        try:
            print(f'***** SearchCommand input {input}')
            if type(input) is dict and 'query' in dict:
                query = input['query']
                response = search_service.run_chat(query)
                print(f'***** SearchCommand response {response}')
                return {'status':'success', 'message':response}
        except Exception as err:
            message = str(err)
            print(f'***** SearchCommand error {str{e}')
            return asdict(TaskResponse('TaskResponse', 'error', message))
