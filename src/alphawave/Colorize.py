import json
from termcolor import colored

class Colorize:
    @staticmethod
    def error(error):
        if isinstance(error, str):
            return colored(error, 'red')
        else:
            return colored(error.message, 'red')

    @staticmethod
    def output(output, quote='', units=''):
        if isinstance(output, str):
            return colored(f"{quote}{output}{quote}", 'green')
        elif isinstance(output, dict):
            # Python doesn't have a built-in library for colorizing JSON.
            # So, we just convert the object to a string and colorize it.
            return colored(json.dumps(output, indent=4), 'green')
        elif isinstance(output, int) or isinstance(output, float):
            return colored(f"{output}{units}", 'blue')
        else:
            return colored(str(output), 'blue')

    @staticmethod
    def success(message):
        return colored(message, 'green')

    @staticmethod
    def title(title):
        return colored(title, 'magenta')

    @staticmethod
    def value(field, value, units=''):
        return f"{field}: "+Colorize.output(value, '\"')

    @staticmethod
    def trace(trace):
        return colored(trace, 'grey')

    @staticmethod
    def warning(warning):
        return colored(warning, 'yellow')

    @staticmethod
    def error(error):
        return colored(error, 'red')

    
