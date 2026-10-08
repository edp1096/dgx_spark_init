import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveSetMembers, setDeploymentValue, resetDeployment, hostIsUsed } from './runtime-sets.js';

test('set deployment edits and resets do not mutate shared definitions or other sets', () => {
  const catalog = { components: [{ id: 'extra', host: 'local', port: 8695, endpoint: 'http://localhost:8695' }], bundles: [{ id: 'local', components: ['extra'] }, { id: 'remote', components: ['extra'] }] };
  const remote = catalog.bundles[1];
  setDeploymentValue(remote, 'extra', 'host', 'worker');
  setDeploymentValue(remote, 'extra', 'endpoint', 'http://worker:18695');
  setDeploymentValue(remote, 'extra', 'port', 0);
  assert.equal(resolveSetMembers(catalog, remote)[0].port, 0);
  assert.equal(resolveSetMembers(catalog, catalog.bundles[0])[0].host, 'local');
  assert.equal(catalog.components.length, 1);
  assert.equal(catalog.components[0].endpoint, 'http://localhost:8695');
  assert.equal(hostIsUsed(catalog, 'worker'), true);
  const copy = structuredClone(remote);
  setDeploymentValue(copy, 'extra', 'host', 'another');
  assert.equal(remote.bindings.extra.host, 'worker');
  resetDeployment(remote, 'extra');
  assert.equal(resolveSetMembers(catalog, remote)[0].host, 'local');
  assert.equal(hostIsUsed(catalog, 'worker'), false);
});
