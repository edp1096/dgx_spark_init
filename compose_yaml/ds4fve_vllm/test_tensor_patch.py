import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from tensor_patch import assemble

class PatchTest(unittest.TestCase):
    def test_header_change_patch_hash_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            layout={'x':{'dtype':'U8','shape':[4],'data_offsets':[0,4]}}
            def header(value):
                data=json.dumps(value).encode(); return struct.pack('<Q',len(data))+data
            original=header(layout)+b'abcd'; (root/'original').write_bytes(original)
            donor_header=header(dict(layout,__metadata__={}))
            expected=donor_header+b'aXYd'; (root/'patch').write_bytes(b'XY')
            assemble(root/'original',root/'target',donor_header,[(1,root/'patch')],hashlib.sha256(expected).hexdigest())
            self.assertEqual((root/'target').read_bytes(),expected)
            self.assertEqual((root/'original').read_bytes(),original)
            with self.assertRaises(ValueError):
                assemble(root/'original',root/'bad',donor_header,[(1,root/'patch')],'0'*64)
            self.assertFalse((root/'bad').exists())

if __name__=='__main__':unittest.main()
