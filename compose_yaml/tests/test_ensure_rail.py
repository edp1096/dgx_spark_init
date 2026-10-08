import importlib.util
import json
import pathlib
import shlex
import subprocess
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]

class RailTest(unittest.TestCase):
    def scenario(self, recipe, mode):
        spec=importlib.util.spec_from_file_location('rail',ROOT/recipe/'ensure_rail.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        calls=[]
        def execute(command, **kwargs):
            rank=1 if command[0]=='ssh' else 0
            args=shlex.split(command[-1]) if rank else command
            calls.append((rank,args))
            data=''
            if args[:2]==['ip','-j']:
                address='10.222.3.'+str(rank+1)
                existing=[] if mode=='missing' else [{'family':'inet','local':address,'prefixlen':24}]
                if mode=='conflict' and rank==1:existing[0]['local']='10.222.3.99'
                data=json.dumps([{'flags':[] if mode=='down' else ['UP','LOWER_UP'],'addr_info':existing}])
            return subprocess.CompletedProcess(command,0,data,'')
        with patch.object(module.subprocess,'run',side_effect=execute):
            if mode in ('conflict','down'):
                with self.assertRaises(RuntimeError):module.ensure_rail('user@192.0.2.2','10.222.3.1','10.222.3.2','rail0','rail1','10.222.3.0/24')
            else:module.ensure_rail('user@192.0.2.2','10.222.3.1','10.222.3.2','rail0','rail1','10.222.3.0/24')
        mutations=[(r,a) for r,a in calls if a[0]=='docker']
        self.assertEqual(len(mutations),2 if mode=='missing' else 0)
        if mode=='missing':
            self.assertEqual([a[-3] for _,a in mutations],['10.222.3.1/24','10.222.3.2/24'])
        if mode in ('existing','missing'):self.assertEqual(sum(a[0]=='ping' for _,a in calls),2)
    def test_both_recipes(self):
        for recipe in ('qwen38_fn_sglang','ds4fve_vllm'):
            for mode in ('missing','existing','conflict','down'):
                with self.subTest(recipe=recipe,mode=mode):self.scenario(recipe,mode)

if __name__=='__main__':unittest.main()
