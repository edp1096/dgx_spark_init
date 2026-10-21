import unittest
import tf_structure as f
from tokenizers import Tokenizer

MODEL='/model'
class Masks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tok=Tokenizer.from_file(MODEL+'/tokenizer.json')
    def ids(self,text):return self.tok.encode(text,add_special_tokens=False).ids
    def test_integer_schema_and_eos(self):
        schema='{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}'
        state=f.create(MODEL,'json_schema',schema)
        first=f.mask(state,[])
        self.assertGreater(len(first),0)
        f.mask(state,self.ids('{"answer":391}'))
    def test_invalid_type_is_refused(self):
        state=f.create(MODEL,'json_schema','{"type":"integer"}')
        with self.assertRaises(ValueError):f.mask(state,self.ids('"bad"'))
    def test_unicode_enum(self):
        state=f.create(MODEL,'json_schema','{"type":"string","enum":["한국어","日本語"]}')
        f.mask(state,self.ids('"한국어"'))
    def test_fresh_request_has_fresh_state(self):
        state=f.create(MODEL,'json_schema','{"type":"integer"}')
        fresh=f.mask(state,[])
        f.mask(state,self.ids('391'))
        other=f.create(MODEL,'json_schema','{"type":"integer"}')
        self.assertEqual(f.mask(other,[]),fresh)
    def test_regex_and_choices(self):
        for kind,spec,text in [('regex','[A-Z]{3}[0-9]{2}','ABC12'),('choice','["red","blue"]','blue')]:
            f.mask(f.create(MODEL,kind,spec),self.ids(text))
    def test_bad_schema(self):
        with self.assertRaises(Exception):f.create(MODEL,'json_schema','{bad')
if __name__=='__main__':unittest.main()
