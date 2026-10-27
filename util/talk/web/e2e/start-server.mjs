import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';

const projectRoot = resolve(import.meta.dirname, '../..');
const port = Number(process.env.SPARKTALK_E2E_PORT || 18585);
const workDir = await mkdtemp(join(tmpdir(), 'sparktalk-e2e-'));
const binary = join(workDir, 'sparktalk-e2e');

// Managed-mode UI tests must not discover or reconfigure the host's real models.
const isolatedCatalog = process.env.SPARKTALK_E2E_MINIMAL_CATALOG ? {
  hosts: { local: {} },
  components: [
    { id: 'chat-test-llm', name: 'Test model', role: 'llm', controller: 'external', endpoint: 'http://127.0.0.1:9', health_url: 'http://127.0.0.1:9/health', model: 'test-model' },
    ...['shorts', 'fish'].map(key => ({ id: `extra-${key}`, name: key, role: key === 'fish' ? 'tts' : 'tool', controller: 'external', endpoint: 'http://127.0.0.1:9', health_url: 'http://127.0.0.1:9/health', compose_asset: `compose.extra-${key}.yaml` })),
  ],
  bundles: [{ id: 'shorts-chat-test', name: 'Test', model_id: 'test-model', model_type: 'qwen3.8', context_tokens: 65536, components: ['chat-test-llm'] }],
} : null;

await new Promise((resolveBuild, rejectBuild) => {
  const build = spawn('go', ['build', '-o', binary, './cmd/chat'], { cwd: projectRoot, stdio: 'inherit' });
  build.once('error', rejectBuild);
  build.once('exit', (code) => code === 0 ? resolveBuild() : rejectBuild(new Error(`go build exited with ${code}`)));
});

await writeFile(join(workDir, 'sparktalk.yaml'), `version: 2
runtime:
  data_dir: ${join(workDir, 'runtime-data')}
  mode: external
  bundle: flash-next
  auto_start: false
  memory_reserve_gib: 8
${isolatedCatalog ? `  builtin_revision: 52\n  catalog: ${JSON.stringify(isolatedCatalog)}\n` : ''}\
server:
  listen_addr: 127.0.0.1:${port}
  database: ${join(workDir, 'sparktalk.db')}
model:
  endpoint: http://127.0.0.1:9
  default_model: test-model
  model_type: qwen3.8
tools:
  enabled: false
appearance:
  assistant_avatar: preset:spark
  user_avatar: preset:person-blue
`, { mode: 0o600 });

const server = spawn(binary, [], { cwd: workDir, stdio: 'inherit' });

async function shutdown() {
  if (!server.killed) server.kill('SIGTERM');
  await rm(workDir, { recursive: true, force: true });
  process.exit(0);
}

process.once('SIGTERM', shutdown);
process.once('SIGINT', shutdown);
server.once('exit', async (code) => {
  await rm(workDir, { recursive: true, force: true });
  process.exit(code ?? 0);
});

await new Promise(() => {});
