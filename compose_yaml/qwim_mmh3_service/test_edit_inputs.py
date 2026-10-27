import base64
import asyncio
import io
from pathlib import Path
import json
import threading
import time
import unittest

from PIL import Image, ImageChops

import api
import edit_inputs
from test_api import Contract
from test_image_inputs import inline_image


class EditingAdmission(unittest.TestCase):
    def test_mask_compositing_preserves_every_outside_pixel(self):
        source=inline_image(width=128,height=192,color='red')
        prepared=edit_inputs.prepare(api.Generate(prompt='fix hand',operation='inpaint',source_image=source,mask_box=[10,20,50,90]))
        self.assertEqual(len(prepared['references']),2)
        result=edit_inputs.finish(edit_inputs.png(Image.new('RGB',(64,64),'blue')),prepared)
        with Image.open(io.BytesIO(result)) as image:
            self.assertEqual(image.size,(128,192))
            self.assertEqual(image.getpixel((30,40)),(0,0,255))
            self.assertEqual(image.getpixel((0,0)),(255,0,0))
            diff=ImageChops.difference(image,prepared['source']);diff.paste((0,0,0),(10,20,50,90))
            self.assertIsNone(diff.getbbox())

    def test_invalid_edits_never_fall_back_to_generation(self):
        cases=[{'operation':'identity_edit'}, {'operation':'inpaint','source_image':inline_image()},
               {'operation':'generate','source_image':inline_image()},
               {'operation':'inpaint','source_image':inline_image(),'mask_box':[-1,0,99,100]},
               {'operation':'identity_edit','reference_images':[inline_image()]*4,'source_image':inline_image()}]
        for fields in cases:
            with self.subTest(fields=list(fields)),self.assertRaises(ValueError):
                edit_inputs.prepare(api.Generate(prompt='edit',**fields))


class EditingAPI(Contract):
    def test_request_cancellation_waits_for_worker_without_stopping_resident_models(self):
        seen={}
        def worker():
            for _ in range(300):
                canceled=list((api.ROOT/'cancel').glob('*'))
                if canceled:
                    seen['case']=canceled[0].name
                    (api.ROOT/'results'/(seen['case']+'.json')).write_text('{"status":"canceled"}')
                    return
                time.sleep(.01)
            seen['error']='cooperative cancellation was not requested'
        class Disconnected:
            headers={}
            async def is_disconnected(self):return True
        thread=threading.Thread(target=worker);thread.start()
        with self.assertRaises(api.HTTPException) as caught:
            asyncio.run(api.generate(api.Generate(prompt='wave'),'h3',Disconnected()))
        thread.join();self.assertEqual(caught.exception.status_code,499)
        self.assertIn('case',seen);self.assertFalse((api.ROOT/'stop').exists())
        self.assertTrue((api.ROOT/'ready.json').exists());self.assertFalse(api.active or api.queued)

    def worker_image(self,seen):
        for _ in range(200):
            files=list((api.ROOT/'requests').glob('*.json'))
            if files:break
            time.sleep(.01)
        else:
            seen['error']='No queued image request';return
        job=json.loads(files[0].read_text());seen['job']=job
        seen['references']=[Path(f).read_bytes() for f in job['reference_files']]
        (api.ROOT/'output'/(job['case']+'.png')).write_bytes(edit_inputs.png(Image.new('RGB',(256,256),'blue')))
        (api.ROOT/'results'/files[0].name).write_text(json.dumps({'status':'success'}))

    def test_multipart_reference_reaches_worker_with_hash_and_operation(self):
        raw=base64.b64decode(inline_image().split(',')[1]);seen={}
        worker=threading.Thread(target=self.worker_image,args=(seen,));worker.start()
        response=self.client.post('/v1/images/edits',data={'prompt':'watercolor','operation':'identity_edit'},files=[('image',('source.png',raw,'image/png'))])
        worker.join();self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(seen['job']['operation'],'identity_edit')
        self.assertEqual(len(seen['references']),1)
        import hashlib
        self.assertEqual(hashlib.sha256(seen['references'][0]).hexdigest(),seen['job']['reference_hashes'][0])
        self.assertEqual(response.json()['reference_sha256'],seen['job']['reference_hashes'])
        self.assertEqual(list((api.ROOT/'inputs').iterdir()),[])

    def test_json_inpaint_returns_composited_source_sized_image(self):
        seen={};worker=threading.Thread(target=self.worker_image,args=(seen,));worker.start()
        response=self.client.post('/v1/images/generations',json={'prompt':'fix fingers','operation':'inpaint','source_image':inline_image(width=128,height=192,color='red'),'mask_box':[10,20,50,90]})
        worker.join();self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()['outside_mask_preserved'])
        with Image.open(io.BytesIO(base64.b64decode(response.json()['data'][0]['b64_json']))) as image:
            self.assertEqual(image.size,(128,192));self.assertEqual(image.getpixel((0,0)),(255,0,0))
            self.assertEqual(image.getpixel((30,40)),(0,0,255))
        self.assertEqual(len(seen['references']),2)

    def test_edit_fields_rejected_on_video_endpoint(self):
        response=self.client.post('/v1/videos/generations',json={'prompt':'wave','source_image':inline_image(),'operation':'identity_edit'})
        self.assertEqual(response.status_code,400)
        self.assertEqual(list((api.ROOT/'requests').iterdir()),[])


if __name__=='__main__':unittest.main()
