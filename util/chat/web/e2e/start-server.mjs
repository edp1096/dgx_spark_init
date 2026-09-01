import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';

const projectRoot = resolve(import.meta.dirname, '../..');
const workDir = await mkdtemp(join(tmpdir(), 'sparktalk-e2e-'));
const binary = join(workDir, 'sparktalk-e2e');

await new Promise((resolveBuild, rejectBuild) => {
  const build = spawn('go', ['build', '-o', binary, './cmd/chat'], { cwd: projectRoot, stdio: 'inherit' });
  build.once('error', rejectBuild);
  build.once('exit', (code) => code === 0 ? resolveBuild() : rejectBuild(new Error(`go build exited with ${code}`)));
});

await writeFile(join(workDir, 'sparktalk.yaml'), `version: 2
runtime:
  mode: external
  bundle: flash-next
  auto_start: false
  memory_reserve_gib: 8
server:
  listen_addr: 127.0.0.1:18585
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
