import unittest
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import  FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from alphawave.PythonResponseValidator import PythonResponseValidator

code=\
"""
random text
```python
class Foo:
    def me(self):
        print('Hello world')
```
"""
class PythonResponseValidatorTest():

    def setUp(self):
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()

    def test_constructor(self):
        
        validator = PythonResponseValidator()
        assert validator is not None

        validator_with_schema = PythonResponseValidator()
        assert validator_with_schema is not None

    def test_validate_response(self):
        validator = PythonResponseValidator()
        response = validator.validate_response(self.memory, self.functions, self.tokenizer,
                                               { 'status': 'success', 'message': { 'role': 'assistant', 'content': code } }, 3)
        assert response is not None
        print(response)
        assert response['valid'] is True

    def test_validate_response2(self):
        validator = PythonResponseValidator()
        with open('../../autopgm/TicTacToeGame.py', 'r') as r:
            code = r.read()
        response = validator.validate_response(self.memory, self.functions, self.tokenizer,
                                               { 'status': 'success', 'message': { 'role': 'assistant', 'content': code } }, 3)
        assert response is not None
        print(response)
        assert response['valid'] is True


if __name__ == "__main__":
    p = PythonResponseValidatorTest()
    p.setUp()
    p.test_constructor()
    p.test_validate_response()
    p.test_validate_response2()
    
