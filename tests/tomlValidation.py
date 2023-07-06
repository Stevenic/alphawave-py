import json
import toml
from cerberus import Validator



test_toml = "[RESPONSE]\nreasoning=\"This is a simple math problem that can be easily solved using the math command.\"\ncommand=\"math\"\ninputs.code=\"35 * 64\"\n[STOP]"

try:
    toml_as_dict = toml.loads(test_toml)
    print(f'### toml_as_dict\n{toml_as_dict}\n')
except Exception as e:
    print(str(e))

dict_schema={
    "reasoning": {
        "type":"string",
        "meta":"I want this field",
        "required": True
    },
    "command": {
        "type":"string",
        "required":True
    },
    "inputs":{
        "type":"dict",
        "keysrules": {"type": "string"}
    }
}

v = Validator(dict_schema)

print('##### test test_toml')
try:
    print(f"*** validation response loads(test_toml) {v.validate(toml_as_dict['RESPONSE'])}")
    print(v._errors)
except Exception as e:
    print(str(e))
print('\n##########\n')


AgentThoughtSchemaTOML = {
    "reasoning": {
        "type":"string",
        "meta":"<reasoning about user task>",
        "required": True
    },
    "command": {
        "type":"string",
        "meta":"<selected command>",
        "required":True
    },
    "inputs":{
        "type":"dict",
        "keysrules": {"type": "string"}
    }
}

v = Validator(AgentThoughtSchemaTOML)

response = {'RESPONSE': {'reasoning': 'The user has asked for the result of 35 multiplied by 64.', 'command': 'math', 'inputs': {'code': '35 * 64'}}}
try:
    print(v.validate(response['RESPONSE']))
    print(v._errors)
except Exception as e:
    print(str(e))
