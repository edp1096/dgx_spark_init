"""Lossless range reconstruction from compressed Xet chunks.

Uses the documented read-only CAS v1 reconstruction representation. It never
changes tensor values and rejects wrong byte ranges, chunk sizes and lengths.
Signed URLs and read tokens remain in memory and are not written to reports.
"""
import ctypes
import ctypes.util
import time
import requests
from xet_audit import Remote

LZ4 = ctypes.CDLL(ctypes.util.find_library('lz4'))
LZ4.LZ4_decompress_safe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
LZ4.LZ4_decompress_safe.restype = ctypes.c_int
LZ4.LZ4F_createDecompressionContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
LZ4.LZ4F_createDecompressionContext.restype = ctypes.c_size_t
LZ4.LZ4F_freeDecompressionContext.argtypes = [ctypes.c_void_p]
LZ4.LZ4F_freeDecompressionContext.restype = ctypes.c_size_t
LZ4.LZ4F_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
LZ4.LZ4F_decompress.restype = ctypes.c_size_t
LZ4.LZ4F_isError.argtypes = [ctypes.c_size_t]
LZ4.LZ4F_isError.restype = ctypes.c_uint


def decompress(src,size):
    buf=ctypes.create_string_buffer(size)
    if not src.startswith(b'\x04\x22\x4d\x18'):
        n=LZ4.LZ4_decompress_safe(src,buf,len(src),size)
        if n!=size:raise ValueError('LZ4 block size mismatch')
        return buf.raw
    context=ctypes.c_void_p()
    status=LZ4.LZ4F_createDecompressionContext(ctypes.byref(context),100)
    if LZ4.LZ4F_isError(status):raise ValueError('Could not create LZ4 frame decoder')
    source=ctypes.create_string_buffer(src);read=written=0
    try:
        while True:
            ns=ctypes.c_size_t(len(src)-read);nd=ctypes.c_size_t(size-written)
            status=LZ4.LZ4F_decompress(context,ctypes.addressof(buf)+written,ctypes.byref(nd),
                ctypes.addressof(source)+read,ctypes.byref(ns),None)
            if LZ4.LZ4F_isError(status):raise ValueError('Invalid LZ4 frame')
            read+=ns.value;written+=nd.value
            if status==0:break
            if not ns.value+nd.value or read>len(src) or written>size:raise ValueError('Incomplete LZ4 frame')
        if read!=len(src) or written!=size:raise ValueError('LZ4 frame size mismatch')
        return buf.raw
    finally:
        LZ4.LZ4F_freeDecompressionContext(context)


def decode_chunks(data, first, end):
    pos=0;chunks={}
    for index in range(first,end):
        if pos+8>len(data):raise ValueError('Truncated Xet chunk header')
        h=data[pos:pos+8];pos+=8
        version=h[0];compressed=int.from_bytes(h[1:4],'little');mode=h[4];size=int.from_bytes(h[5:8],'little')
        if version!=0 or mode not in (0,1,2) or not 0<size<=131072 or not 0<compressed<=131072 or pos+compressed>len(data):
            raise ValueError('Invalid Xet chunk descriptor')
        src=data[pos:pos+compressed];pos+=compressed
        if mode==0:
            if compressed!=size:raise ValueError('Invalid uncompressed chunk length')
            raw=src
        else:
            raw=decompress(src,size)
            if mode==2:
                output=bytearray(size);cursor=0
                for i in range(4):
                    group=(size+3-i)//4
                    output[i::4]=raw[cursor:cursor+group];cursor+=group
                raw=bytes(output)
        chunks[index]=raw
    if pos!=len(data):raise ValueError('Trailing unindexed Xet bytes')
    return chunks


class XetRanges:
    def __init__(self,root,side):
        self.remote=Remote(root,side)
        self.network_bytes=0

    def range(self,name,start,end):
        if not 0<=start<end<=self.remote.files[name]['size'] or end-start>64<<20:
            raise ValueError('Bad or oversized reconstruction range')
        info=self.remote.file(name);cas,auth=self.remote.auth()
        auth['Range']=f'bytes={start}-{end-1}'
        r=self.remote.get(cas+'/v1/reconstructions/'+info['xet_hash'],headers=auth).json()
        terms=r['terms'];fetched={}
        requested=sum(term['unpacked_length'] for term in terms)
        skip=r['offset_into_first_range']
        if not 0<=skip<=requested or requested-skip<end-start or requested>(end-start)+(2<<20):
            raise ValueError('Unbounded or incomplete reconstruction')
        # Read each unique compressed interval once, even if terms reuse it.
        entries={}
        for h,items in r['fetch_info'].items():
            for item in items:
                key=(h,item['range']['start'],item['range']['end'])
                entries[key]=item
        compressed_total=sum(x['url_range']['end']-x['url_range']['start']+1 for x in entries.values())
        if compressed_total>2*(end-start)+(2<<20):raise ValueError('Refusing excessive compressed overfetch')
        for (h,first,last),item in entries.items():
            lo,hi=item['url_range']['start'],item['url_range']['end']
            if not 0<=lo<=hi or hi-lo+1>64<<20:raise ValueError('Invalid compressed interval')
            # Do not allow requests exceptions to expose signed URL query strings.
            for attempt in range(5):
                try:
                    response=self.remote.session().get(item['url'],headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(30,120))
                    with response:
                        if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {lo}-{hi}/'):
                            raise RuntimeError(f'Compressed range rejected: HTTP {response.status_code}')
                        data=response.raw.read(hi-lo+2)
                    self.network_bytes+=len(data)
                    if len(data)!=hi-lo+1:raise ValueError('Compressed range length mismatch')
                    break
                except requests.RequestException as exc:
                    if attempt==4:raise RuntimeError('Compressed range transport failed: '+type(exc).__name__) from None
                    time.sleep(min(20,2**attempt))
            for index,raw in decode_chunks(data,first,last).items():
                key=(h,index)
                if key in fetched and fetched[key]!=raw:raise ValueError('Conflicting repeated chunk')
                fetched[key]=raw
        output=bytearray()
        for term in terms:
            seq=[fetched[(term['hash'],i)] for i in range(term['range']['start'],term['range']['end'])]
            if sum(map(len,seq))!=term['unpacked_length']:raise ValueError('Xet term length mismatch')
            for raw in seq:
                if skip>=len(raw):skip-=len(raw);continue
                if skip:raw=raw[skip:];skip=0
                remaining=end-start-len(output)
                if remaining:output.extend(raw[:remaining])
        if skip or len(output)!=end-start:raise ValueError('Reconstruction length mismatch')
        return bytes(output)
