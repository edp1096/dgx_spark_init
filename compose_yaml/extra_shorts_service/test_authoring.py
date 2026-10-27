import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import authoring
from spec import Captions, Headline


class BrowserContractTests(unittest.TestCase):
    def test_headline_size_overlap_and_display_interval_are_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            spec={'title':'편안한 60대','duration':6,'headline_rules':Headline(text='편안한\n60대').model_dump(),
                  'caption_rules':Captions().model_dump(),'scenes':[{'start':0,'duration':6,'kind':'typography','title':'설명','voice_duration':0,'voice_start':0,'segments':[]}],
                  'captions':[{'id':'caption-0-0','text':'차분한 설명','start':0,'duration':6}],'assets':[]}
            base='#project-title{position:absolute;left:72px;top:180px;width:540px;margin:0;font-size:80px;line-height:1.2;color:white}.caption{position:absolute;left:80px;top:740px;width:520px;color:white}.caption .line{font-size:44px;line-height:1.3}'
            for extra,code in [('',None),('#project-title{font-size:28px}','headline_too_small'),('#project-title{top:720px}','headline_caption_overlap')]:
                authoring.compile_composition(folder,spec,{'css':base+extra,'javascript':'','overlays_html':'','highlights':{}})
                subprocess.run(['node',str(Path(authoring.__file__).with_name('audit.mjs')),str(folder)],stdout=subprocess.DEVNULL,check=False)
                result=json.loads((folder/'audit.json').read_text())
                if code:self.assertIn(code,{x['code'] for x in result['issues']})
                else:
                    self.assertTrue(result['passed'],result['issues'])
                    self.assertTrue(any(p['time']<1 for p in result['previews']))
                    self.assertTrue(all(not s['headline'] for s in result['samples'] if s['time']>=3.5))

    def test_opening_title_exit_fade_is_not_a_hidden_headline(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            spec={'title':'같은 말','duration':15,'headline_rules':Headline(text='같은 말').model_dump(),
                  'caption_rules':Captions().model_dump(),'scenes':[{'start':0,'duration':6.958,'kind':'typography','title':'설명','voice_duration':0,'voice_start':0,'segments':[]},{'start':6.958,'duration':8.042,'kind':'typography','title':'설명','voice_duration':0,'voice_start':6.958,'segments':[]}],
                  'captions':[{'id':'caption-0-0','text':'차분한 설명','start':0,'duration':15}],'assets':[]}
            css='#project-title{position:absolute;left:72px;top:180px;width:540px;margin:0;font-size:80px;line-height:1.2;color:black}.headline-line{color:white}.caption{position:absolute;left:80px;top:740px;width:520px;color:white}.caption .line{font-size:44px;line-height:1.3}'
            for javascript,passed in [("tl.to('#project-title',{opacity:0,duration:.5},3);",True),("tl.to('#project-title',{opacity:0,duration:.2},1);",False),("tl.set('#project-title',{opacity:0},0);",False)]:
                authoring.compile_composition(folder,spec,{'css':css,'javascript':javascript,'overlays_html':'','highlights':{}})
                subprocess.run(['node',str(Path(authoring.__file__).with_name('audit.mjs')),str(folder)],stdout=subprocess.DEVNULL,check=False)
                audit=json.loads((folder/'audit.json').read_text())
                self.assertEqual(audit['passed'],passed,audit['issues'])
                if passed:self.assertTrue(all(x['headline']['color']=='rgb(255, 255, 255)' for x in audit['samples'] if x.get('headline')))
                self.assertTrue(any(abs(s['time']-3.479)<.001 for s in audit['samples']))
                if not passed:self.assertIn('headline_hidden',{x['code'] for x in audit['issues']})

    def test_actual_browser_accepts_layout_and_rejects_hidden_caption_and_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'accent.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="30" fill="red"/></svg>')
            spec = {'title':'검사', 'duration':2, 'caption_rules':Captions().model_dump(),
                    'scenes':[{'start':0,'duration':2,'kind':'typography','title':'검사','voice_duration':0,'voice_start':0,'segments':[]}],
                    'captions':[{'id':'caption-0-0','text':'읽을 수 있는 자막','start':0,'duration':2}],
                    'assets':[{'id':'test','kind':'svg','path':'accent.svg','scene_asset':False}]}
            artwork = {'css':'.visual{position:absolute;inset:0;background:#123}.caption{position:absolute;left:80px;top:600px;width:560px;font-size:16px;color:white}.caption .line{font-size:44px;line-height:1.3}.caption:before{content:"";position:absolute;inset:-20px;background:#000;z-index:-1}.decoration{position:absolute;left:100px;top:100px;width:100px;height:100px}.scene-title,.project-title{display:none}', 'javascript':''}
            for bad in (False, True):
                data = dict(artwork)
                if bad: data['css'] += '.caption,.decoration{display:none}'
                authoring.compile_composition(folder,spec,data)
                if not bad:
                    lint=subprocess.run(["hyperframes","lint",str(folder)],capture_output=True,text=True)
                    self.assertEqual(lint.returncode,0,lint.stdout+lint.stderr)
                subprocess.run(['node',str(Path(authoring.__file__).with_name('audit.mjs')),str(folder)],check=False,stdout=subprocess.DEVNULL)
                report=json.loads((folder/'audit.json').read_text())
                self.assertEqual(report['passed'],not bad,report['issues'])
                if bad:
                    codes={issue['code'] for issue in report['issues']}
                    self.assertIn('caption_hidden',codes)
                    self.assertIn('decoration_not_visible',codes)

    def test_lottie_reference_on_photo_is_a_decoration_not_the_photo(self):
        from pipeline import select_assets
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/"selected-assets.json").write_text(json.dumps([{"id":"heart","kind":"lottie","scene_asset":True}]))
            plan={"scenes":[{"visual":"image","asset_id":"heart"}]}
            records=select_assets(None,plan,folder,None)
            self.assertFalse(records[0]["scene_asset"])
            plan["scenes"][0]["visual"]="lottie"
            self.assertTrue(select_assets(None,plan,folder,None)[0]["scene_asset"])

    def test_template_rejects_network_and_caption_highlight_fabrication(self):
        with self.assertRaises(ValueError):
            authoring.validate_artwork({'css':'.caption{font-size:44px;color:white}', 'javascript':'fetch("https://example.com")'})
        with self.assertRaises(ValueError):
            authoring._caption_html({'id':'c','text':'실제 문장'}, {'c':'없는 단어'})


if __name__ == '__main__': unittest.main()
