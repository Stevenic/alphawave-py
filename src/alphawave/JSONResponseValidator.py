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
        #print(f'***** JSON ResponseValidator response\n{response},\nschema\n{self.schema}\n\n')
        #print(f'***** JSON ResponseValidator \n{response}, \n {message}\n\n')
        #print(f'***** JSON ResponseValidator \n{response}, \n {message}\n\n')
        text = message if isinstance(message, str) else message.get('content', '')

        # Parse the response text
        parsed = Response.parse_all_objects(text)
        #print(f'***** JSON ResponseValidator Response.parse_all \n{parsed}\n')
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
                    #print(f'***** JSON ResponseValidator VALID\n{obj}\n')
                    return {
                        'type': 'Validation',
                        'valid': True,
                        'value': obj
                    }
                except ValidationError as e:
                    path = str(list(e.relative_schema_path)[1:-1]).replace('[','').replace(']',"").replace(', ', ':')
                    #print(f'***** JSONResponseValidator error msg {e.message}')
                    #print(f'***** JSONResponseValidator error validator {e.validator}')
                    #print(f'***** JSONResponseValidator error validation_value {e.validator_value}')
                    #print(f'***** JSONResponseValidator error relative_schema_path {path}')
                    #print(f'***** JSONResponseValidator error relative_path {e.relative_path}')
                    #print(f'***** JSONResponseValidator error absolute_path {e.absolute_path}')
                    #print(f'***** JSONResponseValidator error context {e.context}')
                    #print(f'***** JSONResponseValidator error cause {e.cause}')
                    if not errors:
                        errors = e
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(errors)}'
                        #'feedback': f'The JSON returned had errors: {errors}'
            }
        else:
            #print(f'***** JSON ResponseValidator SKIPPING VALIDATION\n')
            # Return the last object
            return {
                'type': 'Validation',
                'valid': True,
                'value': parsed[-1]
            }

    def get_error_fix(self, error: ValidationError) -> str:
        # Get argument as a string
        arg = error.validator
        path = str(list(error.relative_schema_path)[1:-1]).replace('[','').replace(']',"").replace(', ', ':')
        #print(f'\n\n***** JSONResponseValidator GFIN {error.validator}, {error.validator_value}, {path}')
        
        switcher = {
            'type': f'convert "{path}" value to a {error.validator_value}',
            'anyOf': f'convert "{path}" to one of the allowed types: {error.validator_value}',
            'additionalProperties': f'remove the "{arg}" field from "{path}"',
            'required': f"add the {error.validator_value} fields to {path if len(path)>0 else 'response'}",
            'format': f'change the "{path}" field to be a {error.validator_value}',
            'uniqueItems': f'remove all duplicate items from "{error.path}"',
            'enum': f'change the "{path}" value to be one of these values: {arg}',
            'const': f'change the "{path}" value to be {arg}',
        }

        return switcher[arg] if arg in switcher else error.message

