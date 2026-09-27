"""CPU regression checks for bounded auditing and GLM tensor mapping."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_original
from build import decode,targets,encode_best,encode
from materialize import Original
from inspect_headers import SOURCES
from weights_core.safetensors_io import Checkpoint

class AdapterTests(unittest.TestCase):
    def test_glm_mapping(self):
        self.assertEqual(targets({'name':'blk.0.attn_output.weight'}),['model.language_model.layers.0.self_attn.o_proj'])
        self.assertEqual(targets({'name':'blk.3.ffn_down_shexp.weight'}),['model.language_model.layers.3.mlp.shared_experts.down_proj'])
        self.assertEqual(len(targets({'name':'blk.3.ffn_down_exps.weight','shape':[2048,4096,288]})),288)
        for name in ['blk.45.attn_output.weight','blk.3.attn_q.weight','output.weight']:
            with self.assertRaises(ValueError):targets({'name':name})
    def test_high_precision_decode_preserves_bits(self):
        for dtype in [torch.bfloat16,torch.float16,torch.float32]:
            with tempfile.TemporaryDirectory() as td:
                w=torch.linspace(-1,1,64,dtype=dtype).reshape(4,16)
                save_file({'p.weight':w},str(Path(td)/'model.safetensors'))
                actual,kind=decode(Checkpoint(td),'p')
                torch.testing.assert_close(actual,w.float(),rtol=0,atol=0)
    def test_stream_hashes_and_resume(self):
        data=bytes(range(128));calls=[]
        def fetch(repo,rev,file,offset,length):calls.append((offset,length));return data[offset:offset+length]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);header=root/'header.json'
            header.write_text(json.dumps({'revision':'fixed','repo':'test','file':'test.gguf','size':128,'tensors':[{'name':'a','shape':[4],'type':0,'offset':32},{'name':'b','shape':[8],'type':0,'offset':64}]}))
            with patch.object(audit_original,'get_range',side_effect=fetch):
                audit_original.audit_file(header,root)
                report=json.loads((root/'header.hashes.json').read_text())
                self.assertEqual(report['status'],'complete')
                self.assertEqual(report['tensors'][0]['sha256'],hashlib.sha256(data[32:48]).hexdigest())
                self.assertEqual(report['tensors'][1]['sha256'],hashlib.sha256(data[64:96]).hexdigest())
                n=len(calls);audit_original.audit_file(header,root);self.assertEqual(len(calls),n)
    def test_proof_reconstruction_crosses_shared_and_changed_ranges(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);headers=root/'headers';headers.mkdir();proof=root/'proof';proof.mkdir();donor=root/'donor';donor.mkdir()
            data=bytes(range(128));d=bytearray(132);d[36:132]=data[32:128];d[68:100]=b'X'*32
            (donor/'test.gguf').write_bytes(d)
            (headers/'original-test.gguf.json').write_text(json.dumps({'data_start':32,'size':128}))
            plan={'files':[{'file':'test.gguf','file_index':2,'shift':4,'shared':[[32,64]],'gaps':[[64,128]]}]}
            p=proof/'proof-plan.json';p.write_text(json.dumps(plan))
            (proof/'complete.json').write_text(json.dumps({'status':'complete','plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
            gap=proof/'2-64-128.bin';gap.write_bytes(data[64:128])
            gap.with_suffix('.json').write_text(json.dumps({'file':'test.gguf','start':64,'end':128,'sha256':hashlib.sha256(data[64:128]).hexdigest()}))
            original=Original(headers,proof,donor)
            self.assertEqual(b''.join(original.blocks('test.gguf',40,80)),data[40:120])
            gap.write_bytes(b'Z'*64)
            with self.assertRaises(ValueError):Original(headers,proof,donor)
    def test_full_original_checksum_validates_shared_bytes_and_padding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);headers=root/'headers';headers.mkdir();proof=root/'proof';proof.mkdir();donor=root/'donor';donor.mkdir()
            data=bytes(range(128));d=bytearray(132);d[36:132]=data[32:128];d[68:100]=b'X'*32
            (donor/'test.gguf').write_bytes(d)
            h={'data_start':32,'size':128,'repo':SOURCES['original'][0],'revision':SOURCES['original'][1],'file':'test.gguf','tensors':[{'name':'a','offset':32,'shape':[4],'type':0},{'name':'b','offset':64,'shape':[8],'type':0}]}
            hp=headers/'original-test.gguf.json';hp.write_text(json.dumps(h));hp.with_suffix('.header.bin').write_bytes(data[:32])
            (root/'original-manifest.json').write_text(json.dumps({'sha':SOURCES['original'][1],'siblings':[{'rfilename':'test.gguf','lfs':{'sha256':hashlib.sha256(data).hexdigest()}}]}))
            p=proof/'proof-plan.json';p.write_text(json.dumps({'files':[{'file':'test.gguf','file_index':2,'shift':4,'shared':[[32,64]],'gaps':[[64,128]]}]}))
            (proof/'complete.json').write_text(json.dumps({'status':'complete','plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
            gap=proof/'2-64-128.bin';gap.write_bytes(data[64:128]);gap.with_suffix('.json').write_text(json.dumps({'file':'test.gguf','start':64,'end':128,'sha256':hashlib.sha256(data[64:128]).hexdigest()}))
            original=Original(headers,proof,donor);original.audit(root/'audit')
            result=json.loads((root/'audit/original-test.gguf.hashes.json').read_text())
            self.assertEqual(result['file_sha256'],hashlib.sha256(data).hexdigest())
            d[40]^=1;(donor/'test.gguf').write_bytes(d)
            with self.assertRaisesRegex(ValueError,'published SHA256'):original.audit(root/'bad-audit')
    def test_scale_selection_never_increases_error(self):
        torch.manual_seed(41)
        with tempfile.TemporaryDirectory() as td:
            base=torch.randn(8,64)*.03;parts,_=encode(base,'nvfp4')
            save_file({'p.'+k:v for k,v in parts.items()},str(Path(td)/'model.safetensors'))
            cp=Checkpoint(td);value,_=decode(cp,'p');target=value+torch.randn_like(value)*.0001
            out,restored,selected,fresh_error=encode_best(target,cp,'p')
            self.assertLessEqual(float((restored-target).double().square().sum()),fresh_error)
            self.assertEqual(set(out),{'weight','weight_scale','weight_scale_2'})
    def test_quantized_size(self):
        self.assertEqual(audit_original.nbytes({'type':8,'shape':[32,2]}),68)
        self.assertEqual(audit_original.nbytes({'type':13,'shape':[256,2]}),352)
        with self.assertRaises(ValueError):audit_original.nbytes({'type':8,'shape':[31]})
if __name__=='__main__':unittest.main()
