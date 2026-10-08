"""Shared BF16 encoding; no model or network dependencies."""
import numpy as np

def bf16_to_float(data):
 return (np.frombuffer(data,dtype='<u2').astype(np.uint32)<<16).view(np.float32)

def float_to_bf16(values):
 if not np.isfinite(values).all():raise ValueError('Non-finite transferred weights')
 bits=np.ascontiguousarray(values,dtype=np.float32).view(np.uint32)
 rounded=((bits+(0x7fff+((bits>>16)&1)))>>16).astype('<u2')
 if not np.isfinite(bf16_to_float(rounded.tobytes())).all():raise ValueError('BF16 overflow')
 return rounded.tobytes()

def norm2(x):return float(np.einsum('i,i->',x,x,dtype=np.float64))
