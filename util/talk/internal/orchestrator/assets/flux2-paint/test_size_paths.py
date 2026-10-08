"""Run in the image runtime; inference is stubbed, geometry is exercised end to end."""
import asyncio,base64,io,unittest
from unittest.mock import patch
from PIL import Image
import api

class SizePaths(unittest.TestCase):
    def test_large_edit_paths(self):
        source=Image.new('RGB',(1600,864),(20,30,40))
        url='data:image/png;base64,'+base64.b64encode(api.png_bytes(source)).decode()
        async def execute(graph):
            v=graph['5']['inputs'];size=(v['width'],v['height'])
            self.assertTrue(all(256<=n<=1024 and n%16==0 for n in size))
            return base64.b64encode(api.png_bytes(Image.new('RGB',size,(200,100,50)))).decode()
        async def cutout(data):
            im=Image.open(io.BytesIO(data)).convert('RGBA');self.assertEqual(im.size,source.size)
            im.putalpha(0)
            return base64.b64encode(api.png_bytes(im)).decode()
        with patch.object(api.base,'execute_workflow',execute),patch.object(api,'cutout',cutout):
            for op in ['inpaint','object_remove','background_cleanup','background_remove','outpaint']:
                with self.subTest(operation=op):
                    args=dict(prompt='edit',operation=op)
                    if op in ('inpaint','object_remove'):args.update(anypaint_image=url,mask_box=[200,100,1400,800])
                    elif op=='outpaint':args.update(anypaint_image=url,outpaint_left=256,outpaint_right=256,preserve_source=True)
                    else:args.update(source_image=url)
                    if op=='background_remove':args['background_method']='lora_rembg'
                    result=asyncio.run(api.generate(api.PaintRequest(**args)))
                    im=Image.open(io.BytesIO(base64.b64decode(result['data'][0]['b64_json'])))
                    self.assertEqual(im.size,(2112,864) if op=='outpaint' else source.size)
                    if op in ('inpaint','object_remove'):self.assertEqual(im.getpixel((0,0)),(20,30,40))
                    if op=='outpaint':
                        self.assertEqual(im.getpixel((256,0)),(20,30,40))
                        self.assertEqual(im.getpixel((0,0)),(200,100,50))
                    if op=='background_remove':self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getpixel((0,0))[3],0)
    def test_canvas_limit(self):
        im=Image.new('RGB',(4096,4096));url='data:image/png;base64,'+base64.b64encode(api.png_bytes(im)).decode()
        with self.assertRaises(api.HTTPException):api.prepare_paint(api.PaintRequest(prompt='extend',operation='outpaint',anypaint_image=url,outpaint_right=16))

if __name__=='__main__':unittest.main()
