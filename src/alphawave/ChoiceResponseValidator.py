from promptrix.promptrixTypes import PromptFunctions, PromptMemory, Tokenizer
from alphawave.alphawaveTypes import PromptResponse, Validation, PromptResponseValidator
import json
import traceback

class ChoiceResponseValidator(PromptResponseValidator):
    def __init__(self, choices=None, missing_choice_feedback="Response did not contain a choice from "):
        self.choices = []
        if choices is None or type(choices) != list:
            print(f' ChoiceResponseValidator init must be provided a list of string choices')
        else:
            for choice in choices:
                self.choices.append(choice)
        self.missing_choice_feedback = missing_choice_feedback+str(self.choices)
        
    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts) -> Validation:
        message = response['message']['content']
        if type(message) != str:
            text = str(message)
        else:
            text = message
        min_find = 99999
        found_choice = ''
        # find first choice in first few chars of returned text. first look for matched case
        for choice in self.choices:
            c = choice
            i = text.find(c)
            if i >=0:
                if i < min_find or i == min_find and len(c) > len(found_choice):
                    # this will make sure 'does not' beats 'does' in same start pos!
                    min_find = i
                    found_choice = choice # return choice as capitalized in spec
        if min_find < 15:
            return {
                'type': 'Validation',
                'valid': True,
                'value': found_choice
            }
                
        min_find = 99999
        text = text.lower()
        for choice in self.choices:
            c = choice.lower()
            if c in text:
                if text.find(c) < min_find or (text.find(c) == min_find and len(c) > len(found_choice)):
                    # this will make sure 'does not' beats 'does' in same start pos!
                    min_find = text.find(c) # ignore case in match
                    found_choice = choice # return choice as capitalized in spec
        if min_find < 9999:
            return {
                'type': 'Validation',
                'valid': True,
                'value': found_choice
            }
                
        return {
            'type': 'Validation',
            'valid': False,
            'feedback': self.missing_choice_feedback 
        }
