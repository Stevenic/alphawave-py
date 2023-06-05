class MemoryFork:
    def __init__(self, memory):
        self._fork = dict()
        self._memory = memory

    def has(self, key):
        return key in self._fork or self._memory.has(key)

    def get(self, key):
        if key in self._fork:
            return self._fork[key]
        else:
            return self._memory.get(key)

    def set(self, key, value):
        self._fork[key] = value

    def delete(self, key):
        if key in self._fork:
            self._fork.pop(key, None)

    def clear(self):
        self._fork.clear()
