"""Publish the qualified local artifact in one atomic Hugging Face commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '0')
os.environ.setdefault('HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION', '1')
os.environ.setdefault('HF_XET_FIXED_UPLOAD_CONCURRENCY', '2')

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--publication', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.publication/'plan.json').read_text())
    manifest = json.loads((a.model/'transfer-manifest.json').read_text())
    qualification = json.loads((a.model/'runtime-qualification.json').read_text())
    assert plan['repo_id'] == 'edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD'
    assert plan['commit_message'] == 'qad step-5500'
    assert manifest['runtime_validated'] and qualification['status'] == 'passed'
    assert manifest['base_revision'] == plan['source_revision']
    assert hashlib.sha256((a.model/'README.md').read_bytes()).hexdigest() == plan['readme_sha256']
    api = HfApi()
    assert api.whoami()['name'] == 'edp1096'
    assert api.model_info(plan['repo_id']).sha == plan['parent_commit']

    def status(phase, **values):
        report = {'phase': phase, 'time': time.time(), 'repo_id': plan['repo_id'], **values}
        tmp = a.publication/'publication-status.tmp'
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(a.publication/'publication-status.json')

    names = list(manifest['output_shard_hashes'])
    names += ['README.md', 'LICENSE', 'config.json', 'generation_config.json',
        'hf_quant_config.json', 'model.safetensors.index.json', 'tokenizer.json',
        'tokenizer_config.json', 'vocab.json', 'merges.txt', 'chat_template.jinja',
        'preprocessor_config.json', 'video_preprocessor_config.json',
        'transfer-manifest.json', 'bf16-audit.json', 'runtime-qualification.json',
        'upstream-config.json', 'upstream-HYBRID.json', 'upstream-export-manifest.json',
        'docs/conversion-summary.json', 'docs/source-audit-summary.json',
        'docs/source-alignment.json', 'docs/numeric-probe.json']
    assert len(names) == len(set(names))
    operations = []
    for n in names:
        op = CommitOperationAdd(path_in_repo=n, path_or_fileobj=a.model/n)
        if n in manifest['output_shard_hashes']:
            assert op.upload_info.sha256.hex() == manifest['output_shard_hashes'][n], n
        operations.append(op)
        status('hashing', completed=len(operations), total=len(names))
        print('HASHED', n, flush=True)
    stale = sorted(set(plan['previous_files'])-set(names)-{'.gitattributes'})
    operations += [CommitOperationDelete(path_in_repo=n) for n in stale]
    status('uploading', files=len(names), replaced_old_files=stale)
    print('UPLOADING', len(names), 'files; commit:', plan['commit_message'], flush=True)
    result = api.create_commit(repo_id=plan['repo_id'], repo_type='model',
        operations=operations, commit_message=plan['commit_message'],
        parent_commit=plan['parent_commit'], num_threads=2)
    status('verifying_remote', revision=result.oid)
    info = api.model_info(plan['repo_id'], revision=result.oid, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    for n, expected in manifest['output_shard_hashes'].items():
        f = remote[n]
        assert f.size == (a.model/n).stat().st_size, n
        if f.lfs is not None:
            assert f.lfs.sha256 == expected, n
        else:
            # The three tensorless placeholders are ordinary Git blobs.
            raw = (a.model/n).read_bytes()
            assert f.blob_id == hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), n
    assert set(remote) == set(names)|{'.gitattributes'}
    status('complete', revision=result.oid, verified_weight_shards=len(manifest['output_shard_hashes']),
        commit_message=plan['commit_message'], url=result.commit_url)
    print('PUBLISHED', result.commit_url, flush=True)


if __name__ == '__main__':
    main()
