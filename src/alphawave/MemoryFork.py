from promptrix.promptrixTypes import PromptMemory
from promptrix.VolatileMemory import VolatileMemory

class MemoryFork(PromptMemory):
    def __init__(self, memory):
        self._fork = VolatileMemory()
        self._memory = memory

    def has(self, key):
        return self._fork.has(key) or self._memory.has(key)

    def get(self, key):
        if self._fork.has(key):
            value = self._fork.get(key)
            if value is not None and isinstance(value, dict):
                return json.loads(json.dumps(value))
            else:
                return value
        else:
            value = self._memory.get(key)
            if value is not None and isinstance(value, dict):
                return json.loads(json.dumps(value))
            else:
                return value

    def set(self, key, value):
        if value is not None and isinstance(value, dict):
            clone = json.loads(json.dumps(value))
            self._fork.set(key, clone)
        else:
            self._fork.set(key, value)

    def delete(self, key):
        if self._fork.has(key):
            self._fork.delete(key)

    def clear(self):
        self._fork.clear()
