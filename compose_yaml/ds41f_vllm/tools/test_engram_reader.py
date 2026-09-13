"""Native reader integrity/error/reuse tests; no model or GPU needed."""
import concurrent.futures
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engram_reader import Reader

class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.data=bytes((i*31+i//7)%256 for i in range(131072))
        p=Path(self.tmp.name)/'rows';p.write_bytes(self.data)
        self.fd=os.open(p,os.O_RDONLY);self.reader=Reader(4)
    def tearDown(self):
        self.reader.close();os.close(self.fd);self.tmp.cleanup()
    def test_mixed_rows_and_concurrent_callers(self):
        def one(seed):
            rng=random.Random(seed);jobs=[];outputs=[];expected=[]
            for width,base,n in ((256,17,200),(8,1025,213),(17,83,17),(1,1,0)):
                rows=[rng.randrange(350) for _ in range(n)]
                buf=bytearray(n*width);jobs.append((self.fd,base,rows,width,buf));outputs.append(buf)
                expected.append(b''.join(self.data[base+i*width:base+(i+1)*width] for i in rows))
            self.reader.read(jobs,chunk=1+seed%16)
            self.assertEqual(outputs,expected)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(one,range(50)))
    def test_errors_finish_all_reads_and_allow_reuse(self):
        for fd,base in ((-1,0),(self.fd,len(self.data)-2)):
            with self.assertRaises(OSError):
                self.reader.read([(fd,base,list(range(30)),8,bytearray(240))],chunk=1)
            out=bytearray(8);self.reader.read([(self.fd,0,[1],8,out)])
            self.assertEqual(out,self.data[8:16])
    def test_bounds_empty_and_closed(self):
        self.reader.read([]);self.reader.read([(self.fd,0,[],8,bytearray())])
        for base,rows,width,buf in ((0,[-1],8,bytearray(8)),(2**63-1,[0],8,bytearray(8)),
                                   (0,[0],8,bytearray(7)),(0,[0],8,bytes(8))):
            with self.assertRaises(ValueError):self.reader.read([(self.fd,base,rows,width,buf)])
        self.reader.close();self.reader.close()
        with self.assertRaises(RuntimeError):self.reader.read([])

if __name__=='__main__':unittest.main()
