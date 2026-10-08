import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from expert_preload import read_profile

class ProfileTest(unittest.TestCase):
    def test_reject_stale_or_invalid_profile(self):
        profile={'schema':1,'revision':'r','format':'f','layers':{str(i):[1,2,3] for i in range(40)}}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'profile.json'
            p.write_text(json.dumps(profile)); layers,digest=read_profile(p,'r','f')
            self.assertEqual(layers['39'],[1,2,3]);self.assertEqual(len(digest),64)
            for revision,layout in [('stale','f'),('r','wrong')]:
                with self.assertRaises(ValueError):read_profile(p,revision,layout)
            for ids in [[1,1],[-1],[384],[True],[]]:
                profile['layers']['0']=ids;p.write_text(json.dumps(profile))
                with self.assertRaises(ValueError):read_profile(p,'r','f')
            del profile['layers']['0'];p.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):read_profile(p,'r','f')

if __name__=='__main__':unittest.main()
