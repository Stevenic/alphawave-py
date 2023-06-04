from jsonschema import validate, ValidationError
from promptrix import Message, PromptFunctions, PromptMemory, Tokenizer
from types import PromptResponse, Validation
from Response import Response

class JSONResponseValidator:
    def __init__(self, schema=None):
        self.schema = schema

    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts: int):
        message = response.message
        text = message if isinstance(message, str) else message.content if message.content else ''

        # Parse the response text
        parsed = Response.parse_all_objects(text)
        if len(parsed) == 0:
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': 'No valid JSON objects were found in the response. Return a valid JSON object.'
            }

        # Validate the response against the schema
        if self.schema:
            errors = []
            for i in range(len(parsed) - 1, -1, -1):
                obj = parsed[i]
                try:
                    validate(obj, self.schema)
                    return {
                        'type': 'Validation',
                        'valid': True,
                        'value': obj
                    }
                except ValidationError as e:
                    if not errors:
                        errors = e.message

            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f'The JSON returned had the following errors:\n{errors}\n\nReturn a JSON object that fixes these errors.'
            }
        else:
            # Return the last object
            return {
                'type': 'Validation',
                'valid': True,
                'value': parsed[-1]
            }
