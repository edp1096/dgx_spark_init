import json,pathlib,shutil,subprocess
meta=json.loads(subprocess.check_output(['cargo','metadata','--locked','--format-version=1','--filter-platform','wasm32-unknown-unknown']))
packages={p['id']:p for p in meta['packages']};nodes={n['id']:n for n in meta['resolve']['nodes']}
start=next(p['id'] for p in meta['packages'] if p['name']=='rhwp')
seen=set();todo=[start]
while todo:
 key=todo.pop()
 if key in seen:continue
 seen.add(key)
 for dep in nodes[key]['deps']:
  if any(kind['kind']!='dev' for kind in dep['dep_kinds']):todo.append(dep['pkg'])
root=pathlib.Path('/hwp-licenses');root.mkdir()
manifest=[]
for key in sorted(seen):
 p=packages[key];name=p['name']+'-'+p['version'];directory=pathlib.Path(p['manifest_path']).parent
 manifest.append({'name':p['name'],'version':p['version'],'license':p['license']})
 destination=root/name;destination.mkdir(exist_ok=True)
 for pattern in ['LICENSE*','COPYING*','NOTICE*']:
  for file in directory.glob(pattern):
   if file.is_file():shutil.copy2(file,destination/file.name)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
