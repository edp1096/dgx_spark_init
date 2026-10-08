import ctypes
import unittest
from xet_ranges import LZ4,decode_chunks

LZ4.LZ4F_compressFrameBound.argtypes=[ctypes.c_size_t,ctypes.c_void_p]
LZ4.LZ4F_compressFrameBound.restype=ctypes.c_size_t
LZ4.LZ4F_compressFrame.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
LZ4.LZ4F_compressFrame.restype=ctypes.c_size_t


def chunk(raw,mode):
    data=raw if mode!=2 else b''.join(raw[i::4] for i in range(4))
    if mode:
        capacity=LZ4.LZ4F_compressFrameBound(len(data),None);out=ctypes.create_string_buffer(capacity)
        n=LZ4.LZ4F_compressFrame(out,capacity,data,len(data),None)
        if LZ4.LZ4F_isError(n):raise ValueError('Test compression failed')
        data=out.raw[:n]
    return b'\0'+len(data).to_bytes(3,'little')+bytes([mode])+len(raw).to_bytes(3,'little')+data


class XetCodecTests(unittest.TestCase):
    def test_all_modes_and_odd_group_sizes_are_lossless(self):
        for mode in [0,1,2]:
            for size in [1,3,4,127,65537,131071]:
                raw=(bytes(range(251))*600)[:size]
                self.assertEqual(decode_chunks(chunk(raw,mode),7,8),{7:raw})

    def test_multiple_chunks_preserve_order(self):
        self.assertEqual(decode_chunks(chunk(b'abc',2)+chunk(b'xyz',1),4,6),{4:b'abc',5:b'xyz'})

    def test_truncation_and_extra_data_are_rejected(self):
        valid=chunk(b'abc'*100,2)
        for corrupt in [valid[:-1],valid+b'x',b'\1'+valid[1:]]:
            with self.assertRaises(ValueError):decode_chunks(corrupt,0,1)


if __name__=='__main__':unittest.main()
