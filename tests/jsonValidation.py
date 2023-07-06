import json
import traceback
import ast
import re
from alphawave.JSONResponseValidator import JSONResponseValidator
from jsonschema import validate, ValidationError

j = JSONResponseValidator()
planning_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "plan": {
            "type":"string",
            "description": "<actions in plan>"
        }
    },
    "required": ["reasoning", "plan"]
}
r1 = "{\"reasoning\":\"The user is asking for the square root of 2. I can use a math function to calculate it.\", \"plan\":\"math\"}"
r2 = "{'reasoning': 'The user is asking for the square root of 2. I can use a math function to calculate it.', 'plan': 'math'}"
r3 = '{"reasoning":"The user is asking for the square root of 2. I can use a math function to calculate it.", "plan":"math"}'
#s = j.parse_dict(response)
#print(s)
#print('\n#######\n')
#print(j.parse_dict(r2))
print('\n#######\n')
s = j.parse_dict(r3)
print(s)
print('\n#######\n')


t = validate(s, planning_validation_schema)
print(t)
