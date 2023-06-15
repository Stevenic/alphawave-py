import asyncio
import aiounittest, unittest
from jsonschema import validate, ValidationError
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import  FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from alphawave.JSONResponseValidator import JSONResponseValidator

class TestJSONResponseValidator(aiounittest.AsyncTestCase):
    memory = VolatileMemory()
    functions = FunctionRegistry()
    tokenizer = GPT3Tokenizer()
    schema = {
        "type": "object",
        "properties": {
            "foo": {
                "type": "string"
            }
        },
        "required": ["foo"]
    }

    def test_constructor(self):
        validator = JSONResponseValidator()
        assert validator is not None

        validator_with_schema = JSONResponseValidator(self.schema)
        assert validator_with_schema is not None

    async def test_validate_response(self):
        validator = JSONResponseValidator()
        response = validator.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': '{"foo":"bar"}' }, 3)
        assert response is not None
        assert response['valid'] is True
        assert response['value'] == { 'foo': 'bar' }

        response = validator.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': { 'role': 'assistant', 'content': '{"foo":"bar"}' } }, 3)
        assert response is not None
        assert response['valid'] is True
        assert response['value'] == { 'foo': 'bar' }

        validator_with_schema = JSONResponseValidator(self.schema)
        response = validator_with_schema.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': '{"foo":"bar"}' }, 3)
        assert response is not None
        assert response['valid'] is True
        assert response['value'] == { 'foo': 'bar' }

        response = validator.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': '' }, 3)
        assert response is not None
        assert response['valid'] is False
        #assert response['feedback'] == 'Invalid JSON. Revise your previous response and return valid JSON as per the earlier schema.'
        assert ('value' in response) is False

        response = validator.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': { 'role': 'assistant', 'content': None } }, 3)
        assert response is not None
        assert response['valid'] is False
        print(response['feedback'])
        #assert response['feedback'] == 'Invalid JSON. Revise your previous response and return valid JSON as per the earlier schema.'
        assert ('value' in response) is False

    async def test_validate_response6(self):
        validator_with_schema = JSONResponseValidator(self.schema)
        response = validator_with_schema.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': '{"foo":7}' }, 3)
        assert response is not None
        assert response['valid'] is False


    async def test_validate_response7(self):
        validator_with_schema = JSONResponseValidator(self.schema)
        response = validator_with_schema.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': '{"foo":"taco"}\n{"foo":"bar"}' }, 3)
        assert response is not None
        assert response['valid'] is True
        assert response['value'] == { 'foo': 'bar' }
    """ # weird one, I think it should fail
    async def test_validate_response8(self):
        validator_with_schema = JSONResponseValidator(self.schema)
        response = validator_with_schema.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': '{"foo":1}\n{"foo":"bar"}\n{"foo":3}' }, 3)
        assert response is not None
        assert response['valid'] is True
        assert response['value'] == { 'foo': 'bar' }
    """
    async def test_validate_response9(self):
        validator_with_schema = JSONResponseValidator(self.schema)
        response = validator_with_schema.validate_response(self.memory, self.functions, self.tokenizer, { 'status': 'success', 'message': '{"bar":"foo"}\n{"foo":3}' }, 3)
        assert response is not None
        assert response['valid'] is False
        #assert response['feedback'] == 'The JSON returned had errors. Apply these fixes:\nconvert "foo" value to astring'
if __name__ == "__main__":
    unittest.main()
