import unittest

#import promptrix.VolatileMemory as VolatileMemory, MemoryFork
from alphawave.MemoryFork import MemoryFork
from promptrix.VolatileMemory import VolatileMemory
class TestMemoryFork(unittest.TestCase):

    def setUp(self):
        self.memory = VolatileMemory({
            "input": "I'd like to book a flight to London",
            "name": "John Doe"
        })

    def test_constructor(self):
        fork = MemoryFork(self.memory)
        self.assertIsNotNone(fork)
        self.assertFalse(fork ==self.memory)
        
    def test_has(self):
        fork = MemoryFork(self.memory)
        fork.set("output", "I can help with that")

        self.assertTrue(fork.has('output'))
        self.assertTrue(fork.has('name'))
        self.assertFalse(fork.has('age'))
        self.assertFalse(self.memory.has('output'))
        
    def test_get(self):
        fork = MemoryFork(self.memory)

        self.assertEqual(fork.get('input'), "I'd like to book a flight to London")
        self.assertEqual(fork.get('name'), "John Doe")
        self.assertIsNone(fork.get('age'))

    def test_set(self):
        fork = MemoryFork(self.memory)
        fork.set('input', "I'd like first class please")

        self.assertEqual(fork.get('input'), "I'd like first class please")
        self.assertEqual(self.memory.get('input'), "I'd like to book a flight to London")

    def test_delete(self):
        fork = MemoryFork(self.memory)
        fork.delete('input')

        self.assertEqual(fork.get('input'), "I'd like to book a flight to London")
        self.assertEqual(self.memory.get('input'), "I'd like to book a flight to London")

        fork.delete('name')

        self.assertEqual(fork.get('name'), "John Doe")
        self.assertEqual(self.memory.get('name'), "John Doe")

    def test_clear(self):
        fork = MemoryFork(self.memory)
        fork.clear()

        self.assertEqual(fork.get('output'), self.memory.get('output'))
        self.assertEqual(fork.get('input'), "I'd like to book a flight to London")


if __name__ == '__main__':
    unittest.main()
