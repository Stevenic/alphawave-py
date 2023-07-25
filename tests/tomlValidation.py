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

toml_str = "best=\"C\"\nsecond=\"B\"\nthird=\"D\"\nreasoning=\"Option C accurately describes the origin and significance of the triskeles symbol. It mentions that the symbol represents a triple goddess reconstructed by the rulers of Syracuse, which is historically accurate. It also correctly states that the symbol represents the Greek name for Sicily, Trinacria, which contains the element 'tria' meaning three. The head of Medusa at the center of the Sicilian triskeles representing the three headlands is also mentioned correctly. Option B is the second best choice as it correctly mentions that the triskeles symbol is a representation of three interlinked spirals and was adopted as an emblem by the rulers of Syracuse. It also correctly states that the usage of the symbol in modern flags of Sicily has its origins in the ancient Greek name for the island, Trinacria. However, it incorrectly states that the head of Medusa represents the island's rich cultural heritage. Option D is the third best choice as it correctly states that the triskeles symbol represents three interlocked spiral arms and became an emblem for the rulers of Syracuse. It also correctly mentions that the usage of the symbol in modern flags of Sicily is due to the"

answer_schema={
    "reasoning": {
        "type":"string",
        "meta": "<step-by-step reasoning>"
    },
    "best": {
        "type":"string",
        "meta": "<single letter answer A or B or C or D or E>",
        "required": True
    },
    "second": {
        "type":"string",
        "meta": "<single letter answer A or B or C or D or E>",
        "required": True
    },
    "third": {
        "type":"string",
        "meta": "<single letter answer A or B or C or D or E>",
        "required": True
    }
}
