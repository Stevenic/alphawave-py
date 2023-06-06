import unittest
import json
from alphawave.Response import Response

class TestResponse(unittest.TestCase):
    def test_parseJSON(self):
        self.assertEqual(Response.parse_json('{ "foo": "bar" }'), { "foo": "bar" })
        self.assertEqual(Response.parse_json('{\n"foo": "bar"\n}'), { "foo": "bar" })
        self.assertEqual(Response.parse_json('{ "foo": "bar\\"baz" }'), { "foo": 'bar"baz' })
        self.assertEqual(Response.parse_json('{ "foo": "bar\\\\baz" }'), { "foo": 'bar\\baz' })
        self.assertEqual(Response.parse_json('{ "foo": "bar\\/baz" }'), { "foo": 'bar/baz' })
        self.assertEqual(Response.parse_json('Hello { "foo": "bar" } World'), { "foo": 'bar' })
        self.assertEqual(Response.parse_json('Hello{}World'), None)
        self.assertEqual(Response.parse_json('{'), None)
        self.assertEqual(Response.parse_json('Plan: {"foo":"bar","baz":[1,2,3],"qux":{"quux":"corge"}}'), { "foo": 'bar', "baz": [1, 2, 3], "qux": { "quux": 'corge' } })
        self.assertEqual(Response.parse_json('Plan: "foo":"bar"}'), None)
        self.assertEqual(Response.parse_json('Plan: {"foo":"bar","baz":{"qux":[1,2,'), None)
        self.assertEqual(Response.parse_json('Plan: {"foo":"bar\\'), None)
        self.assertEqual(Response.parse_json('Plan: {"foo": ["bar"}'), None)
        self.assertEqual(Response.parse_json('Plan: {"foo":]"bar"}'), None)

    def test_parseAllObjects(self):
        self.assertEqual(Response.parse_all_objects('{ "foo": "bar" }'), [{ "foo": 'bar' }])
        self.assertEqual(Response.parse_all_objects('{"foo":"bar"}\n{"baz":"qux"}'), [{ "foo": 'bar' }, { "baz": 'qux' }])
        self.assertEqual(Response.parse_all_objects('{"foo":"bar"}\nHello World\nPlan: {"baz":"qux"}'), [{ "foo": 'bar' }, { "baz": 'qux' }])
        self.assertEqual(Response.parse_all_objects('{"foo":"bar"} {"bar":"foo"}\nHello World\nPlan: {"baz":"qux"}'), [{ "foo": 'bar' }, { "baz": 'qux' }])
        self.assertEqual(Response.parse_all_objects('{\n"foo": "bar"\n}'), [{ "foo": 'bar' }])
        self.assertEqual(Response.parse_all_objects('Hello\nWorld'), [])

if __name__ == '__main__':
    unittest.main()
