import json

class Response:
    @staticmethod
    def parse_all_objects(text):
        objects = []
        lines = text.split('\n')
        if len(lines) > 1:
            for line in lines:
                obj = Response.parse_json(line)
                if obj:
                    objects.append(obj)

        if len(objects) == 0:
            obj = Response.parse_json(text)
            if obj:
                objects.append(obj)

        return objects

    @staticmethod
    def parse_json(text):
        start_brace = text.find('{')
        if start_brace >= 0:
            obj_text = text[start_brace:]
            nesting = ['}']
            cleaned = '{'
            in_string = False
            i = 1
            while i < len(obj_text) and len(nesting) > 0:
                ch = obj_text[i]
                if in_string:
                    cleaned += ch
                    if ch == '\\':
                        i += 1
                        if i < len(obj_text):
                            cleaned += obj_text[i]
                        else:
                            return None
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '{':
                        nesting.append('}')
                    elif ch == '[':
                        nesting.append(']')
                    elif ch == '}':
                        close_object = nesting.pop()
                        if close_object != '}':
                            return None
                    elif ch == ']':
                        close_array = nesting.pop()
                        if close_array != ']':
                            return None
                    elif ch == '<':
                        ch = '"<'
                    elif ch == '>':
                        ch = '>"'
                    cleaned += ch
                i += 1

            if len(nesting) > 0:
                cleaned += ''.join(reversed(nesting))

            try:
                obj = json.loads(cleaned)
                return obj if len(obj.keys()) > 0 else None
            except json.JSONDecodeError:
                return None
        else:
            return None
