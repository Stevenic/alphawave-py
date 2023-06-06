from jsonschema import validate, ValidationError
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from alphawave.alphawaveTypes import PromptResponse, Validation, PromptResponseValidator
from alphawave.Response import Response
from pyee import AsyncIOEventEmitter

class JSONResponseValidator(PromptResponseValidator):
    def __init__(self, schema=None, missing_json_feedback='No valid JSON objects were found in the response. Return a valid JSON object.'):
        self.schema = schema
        self.missing_json_feedback = missing_json_feedback

    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts: int) -> Validation:
        message = response['message']
        text = message if isinstance(message, str) else message.get('content', '')

        # Parse the response text
        parsed = Response.parse_all_objects(text)
        if len(parsed) == 0:
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': self.missing_json_feedback
            }

        # Validate the response against the schema
        if self.schema:
            errors = None
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
                        errors = e

            return {
                'type': 'Validation',
                'valid': False,
                'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(errors)}'
            }
        else:
            # Return the last object
            return {
                'type': 'Validation',
                'valid': True,
                'value': parsed[-1]
            }

    def get_error_fix(self, error: ValidationError) -> str:
        # Get argument as a string
        arg = ''
        if isinstance(error.argument, list):
            arg = ','.join(error.argument)
        elif isinstance(error.argument, dict):
            arg = str(error.argument)
        else:
            arg = str(error.argument)

        switcher = {
            'type': f'convert "{error.property}" to a {arg}',
            'anyOf': f'convert "{error.property}" to one of the allowed types: {arg}',
            'additionalProperties': f'remove the "{arg}" property from "{error.property}"',
            'required': f'add the "{arg}" property to "{error.property}"',
            'format': f'change the "{error.property}" property to be a {arg}',
            'uniqueItems': f'remove all duplicate items from "{error.property}"',
            'enum': f'change the "{error.property}" property to be one of these values: {arg}',
            'const': f'change the "{error.property}" property to be {arg}',
        }

        return switcher.get(error.name, f'"{error.property}" {error.message}. Fix that')

