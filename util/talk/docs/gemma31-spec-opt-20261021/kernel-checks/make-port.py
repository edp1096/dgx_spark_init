"""Narrow opt-in port on the pinned runtime, after SM121 unit qualification."""
from pathlib import Path
import hashlib,json
r=Path(__file__).parent
s=(r.parent/'source-baseline/triton_backend.py').read_text()
old='''        self.use_verify_splitkv = (
            is_gfx95_supported()
            and envs.SGLANG_ENABLE_SPLITKV_VERIFY.get()
            and self.topk == 1
        )'''
new='''        # Private experiment: only the validated Gemma31/SM121 target.
        text_config = getattr(model_runner.model_config.hf_config, "text_config", model_runner.model_config.hf_config)
        gemma31_sm121 = (
            get_bool_env_var("GEMMA31_VERIFY_SPLITKV", "false")
            and _is_cuda
            and torch.cuda.get_device_capability() == (12, 1)
            and get_parallel().attn_tp_size == 1
            and get_parallel().attn_dcp_size == 1
            and getattr(text_config, "model_type", None) == "gemma4_text"
            and getattr(text_config, "hidden_size", None) == 5376
        )
        self.use_verify_splitkv = (
            (is_gfx95_supported() and envs.SGLANG_ENABLE_SPLITKV_VERIFY.get())
            or gemma31_sm121
        ) and self.topk == 1
        if gemma31_sm121:
            print("GEMMA31_SPLITKV_ENABLED topk=" + str(self.topk), flush=True)'''
assert s.count(old)==1
(r/'triton_backend_sm121.py').write_text(s.replace(old,new))
s=(r/'verify_splitkv.py').read_text()
needle='\n\ndef cuda_splitkv_tile('
addition='''

# Local qualification: SM121, Gemma31 BF16, causal verify widths 4/6/8/10/16.
_CUDA_SPLITKV_TILES.update({
    (32, hkv, dim, width, (12, 1)): tile
    for hkv, dim, tile in [(16, 256, (32, 4)), (4, 512, (16, 4))]
    for width in [4, 6, 8, 9, 10, 16]
})
'''
assert s.count(needle)==1;s=s.replace(needle,addition+needle)
needle='    key = (h_q, h_kv, head_dim, v_head_dim, l_ext, str(device), n_splits)'
addition='''    sm121 = (not _IS_HIP and torch.device(device).type == "cuda"
             and torch.cuda.get_device_capability(device) == (12, 1))
    if sm121:
        # Static per-shape policy remains stable under graph replay. The global
        # KV graph buffer is max-context sized even for short requests.
        n_splits = 4 if head_dim == 256 else 8
'''
assert s.count(needle)==1;s=s.replace(needle,addition+needle)
needle='        _VK_CACHE[key] = vk'
replacement='''        _VK_CACHE[key] = vk
        if sm121:
            print("GEMMA31_SPLITKV_BUILD", dict(hq=h_q, hkv=h_kv, dim=head_dim,
                  width=l_ext, splits=n_splits, block_n=block_n, warps=num_warps), flush=True)'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
needle='    # No exotic features.\n'
addition='''    if not _IS_HIP and q_extend.device.type == "cuda":
        if torch.cuda.get_device_capability(q_extend.device) == (12, 1):
            if q_extend.dtype != torch.bfloat16 or k_buffer.dtype != torch.bfloat16 or v_buffer.dtype != torch.bfloat16:
                return False
            if q_extend.ndim != 3 or k_extend.ndim != 3:
                return False
            tile = cuda_splitkv_tile(q_extend.shape[1], k_extend.shape[1],
                                     q_extend.shape[2], max_len_extend, (12, 1))
            if tile is None:
                return False
'''
assert s.count(needle)==1;s=s.replace(needle,addition+needle)
(r/'verify_splitkv_sm121.py').write_text(s)
files=['triton_backend_sm121.py','verify_splitkv_sm121.py']
(r/'port-sha256.json').write_text(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n')
