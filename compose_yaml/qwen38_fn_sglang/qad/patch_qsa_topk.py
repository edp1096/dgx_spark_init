"""Preserve radix candidates when a coarse bin exceeds shared-memory capacity.

The common path stays unchanged. Overflow uses bounded-memory full-key radix
selection instead of discarding candidates. No host synchronization or allocation.
"""
from pathlib import Path

HELPER = r'''
// QAD exact overflow path: rescan scores rather than truncate a radix bin.
template <int kTopK>
SGL_DEVICE void qad_exact_overflow_topk(
    const float* input, int* output, int row_start, int length) {
  __shared__ int hist[256];
  __shared__ uint32_t prefix;
  __shared__ int remaining, greater_count, greater_written, ties_written;
  const int tx = threadIdx.x;
  if (tx == 0) { prefix = 0; remaining = kTopK; }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tx < 256) hist[tx] = 0;
    __syncthreads();
    const uint32_t mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
    for (int i = tx; i < length; i += blockDim.x) {
      const uint32_t key = convert_to_uint32(input[row_start + i]);
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
    }
    __syncthreads();
    if (tx == 0) {
      for (int bin = 255; bin >= 0; --bin) {
        if (hist[bin] >= remaining) {
          prefix |= uint32_t(bin) << shift;
          break;
        }
        remaining -= hist[bin];
      }
    }
    __syncthreads();
  }
  if (tx == 0) {
    greater_count = kTopK - remaining;
    greater_written = 0;
    ties_written = 0;
  }
  __syncthreads();
  for (int i = tx; i < length; i += blockDim.x) {
    const uint32_t key = convert_to_uint32(input[row_start + i]);
    if (key > prefix) {
      output[atomicAdd(&greater_written, 1)] = i;
    } else if (key == prefix) {
      const int pos = atomicAdd(&ties_written, 1);
      if (pos < remaining) output[greater_count + pos] = i;
    }
  }
  __syncthreads();
}

'''


def patched(source):
    if 'void qad_exact_overflow_topk(' in source:
        return source
    anchor = '// When length <= kTopK, write the indices directly.'
    stage = '  // stage 2: refine with 8bit radix passes'
    if source.count(anchor) != 1 or source.count(stage) != 1:
        raise RuntimeError('Unexpected pinned fast_topk source')
    source = source.replace(anchor, HELPER + anchor)
    return source.replace(stage, '''  if (s_num_input[0] > int(SMEM_INPUT_SIZE)) {
    qad_exact_overflow_topk<kTopK>(input, index, row_start, length);
    return;
  }

''' + stage)


if __name__ == '__main__':
    path = Path('/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh')
    path.write_text(patched(path.read_text()))
