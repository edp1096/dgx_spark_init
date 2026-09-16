import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('launch',Path(__file__).with_name('launch.py'));launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)

class LaunchTests(unittest.TestCase):
 def test_family_context_overrides_preserve_source(self):
  for family in ('ornith35','gemma26'):
   for context in (262144,524288,1048576):
    with self.subTest(family=family,context=context),tempfile.TemporaryDirectory() as folder:
     p=Path(folder)/'config.json'
     rope={'rope_type':'default','partial_rotary_factor':.25} if family=='ornith35' else {'full_attention':{'rope_type':'proportional','partial_rotary_factor':.25},'sliding_attention':{'rope_type':'default'}}
     original={'text_config':{'max_position_embeddings':262144,'hidden_size':2048,'rope_parameters':rope}}
     p.write_text(json.dumps(original));before=p.read_bytes()
     args=launch.command({'MODEL_FAMILY':family,'MODEL_PATH':folder,'SERVED_MODEL_NAME':'test','CONTEXT_LENGTH':str(context),'RUNTIME_VIEW_ROOT':str(Path(folder).parent/('views-'+Path(folder).name))})
     override=json.loads((Path(args[2])/'config.json').read_text())['text_config']
     self.assertEqual(override['max_position_embeddings'],context)
     self.assertEqual(override['hidden_size'],2048)
     self.assertEqual(p.read_bytes(),before)
     import shutil;shutil.rmtree(Path(args[2]).parent)
     if family=='ornith35' and context>262144:self.assertEqual(override['rope_parameters']['factor'],context/262144)
     elif family=='gemma26':
      self.assertEqual(override['rope_parameters']['full_attention']['factor'],context/262144)
      self.assertEqual(override['rope_parameters']['sliding_attention'],rope['sliding_attention'])
     self.assertEqual(args[args.index('--tensor-parallel-size')+1],'1')


 def test_assistant_context_and_opt_out(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);model=root/'model';draft=root/'draft'
   model.mkdir();draft.mkdir()
   cfg={'text_config':{'max_position_embeddings':262144,'hidden_size':2816,'rope_parameters':{'full_attention':{'rope_type':'proportional'},'sliding_attention':{'rope_type':'default'}}}}
   raw=json.dumps(cfg);(model/'config.json').write_text(raw);(draft/'config.json').write_text(raw)
   env={'MODEL_FAMILY':'gemma26','MODEL_PATH':str(model),'SERVED_MODEL_NAME':'test','CONTEXT_LENGTH':'1048576','RUNTIME_VIEW_ROOT':str(root/'views'),'RUNTIME_DRAFT_VIEW_ROOT':str(root/'draft-views'),'DRAFT_MODEL_PATH':str(draft),'MTP_TOKENS':'1','DRAFT_VOCAB':'ko64k'}
   args=launch.command(env);spec=json.loads(args[args.index('--speculative-config')+1])
   self.assertEqual(spec['num_speculative_tokens'],1);self.assertEqual(spec['max_model_len'],1048576)
   view=json.loads((Path(spec['model'])/'config.json').read_text())
   self.assertEqual(view['text_config']['rope_parameters']['full_attention']['factor'],4)
   self.assertEqual((draft/'config.json').read_text(),raw)
   env['MTP_TOKENS']='0';env['DRAFT_MODEL_PATH']=str(root/'missing')
   self.assertNotIn('--speculative-config',launch.command(env))
   env['MTP_TOKENS']='2'
   with self.assertRaises(ValueError):launch.command(env)
   env['MTP_TOKENS']='1';env['DRAFT_CONTEXT_LENGTH']='2097152'
   with self.assertRaises(ValueError):launch.command(env)

 def test_tokenizer_specific_draft_vocab(self):
  import hashlib
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);model=root/'model';vocab=root/'vocab';model.mkdir();vocab.mkdir()
   for name,raw in [('gemma-ko64k.json',b'original'),('gemma-qat-ko64k.json',b'qat')]:
    (vocab/name).write_text(json.dumps({'tokenizer_sha256':hashlib.sha256(raw).hexdigest()}))
    (model/'tokenizer.json').write_bytes(raw)
    self.assertEqual(launch.draft_vocab_path('gemma26',model,vocab).name,name)
   (model/'tokenizer.json').write_bytes(b'unknown')
   with self.assertRaises(ValueError):launch.draft_vocab_path('gemma26',model,vocab)
   (model/'tokenizer.json').write_bytes(b'qat')
   (vocab/'gemma-duplicate-ko64k.json').write_bytes((vocab/'gemma-qat-ko64k.json').read_bytes())
   with self.assertRaises(ValueError):launch.draft_vocab_path('gemma26',model,vocab)

if __name__=='__main__':unittest.main()
