from pathlib import Path
p=Path('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py')
if not p.exists():
 import importlib.util
 spec=importlib.util.find_spec('sglang');p=Path(spec.submodule_search_locations[0])/'srt/layers/quantization/modelopt_quant.py'
s=p.read_text();start=s.index('class ModelOptNvFp4FusedMoEMethod');head,body=s[:start],s[start:]
anchor='''    def apply(
        self,
        layer: FusedMoE,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
'''
assert body.count(anchor)==1
body=body.replace(anchor,anchor+'''        import os
        if os.environ.get("SGLANG_B12X_PROBE_MOE", "0") == "1":
            from sglang_b12x_probe import apply
            return apply(layer, dispatch_output, self.moe_runner_config)
''',1)
p.write_text(head+body)
