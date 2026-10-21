import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for frontend contract tests")
class DepthCacheFrontendTests(unittest.TestCase):
    def run_contract(self, body):
        script = f"""
const assert = require('node:assert/strict');
const contract = require({json.dumps(str(ROOT / 'static' / 'app.js'))});
(async () => {{
{body}
}})().catch((error) => {{ console.error(error); process.exitCode = 1; }});
"""
        result = subprocess.run([NODE, "-e", script], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_old_settings_default_cache_off_and_new_settings_are_strict(self):
        self.run_contract("""
const old = contract.performancePreset('balanced');
assert.equal(old.use_depth_cache, false);
const { use_depth_cache, ...legacy } = old;
assert.equal(contract.validateSettings(legacy).use_depth_cache, false);
for (const value of [1, 0, 'true', 'false', null]) {
  assert.throws(() => contract.validateSettings({ ...old, use_depth_cache: value }), /use_depth_cache.*boolean/);
}
assert.equal(contract.validateSettings({ ...old, use_depth_cache: true }).use_depth_cache, true);
""")

    def test_queue_cache_state_and_inventory_contract(self):
        self.run_contract("""
assert.equal(contract.executionSummary({ depth_cache_hit: true, active_provider: 'DmlExecutionProvider', device: 'gpu' }), 'Cached depth · CPU rendering');
const status = { state: 'hit', frames: 37, bytes: 2048, warning: null };
assert.equal(contract.depthCacheStatusLabel(status, (n) => `${n} B`), 'Depth cache hit · 37 frames · 2048 B');
assert.match(contract.depthCacheStatusLabel({ ...status, state: 'writing', estimated_bytes: 4096 }, (n) => `${n} B`), /estimated total 4096 B/);
assert.throws(() => contract.validateDepthCacheStatus({ ...status, estimated_bytes: NaN }), /invalid depth cache estimate/i);
assert.throws(() => contract.validateDepthCacheStatus({ ...status, state: 'unknown' }), /invalid depth cache status/i);
const inventory = contract.validateDepthCacheInventory({
  quota_bytes: 2048, bytes: 1024,
  entries: [
    { key: 'a'.repeat(64), bytes: 1024, frames: 37, active: false },
    { key: 'b'.repeat(64), bytes: 0, frames: 0, active: true }
  ], errors: [{ key: '', error: 'directory could not be scanned' }]
});
assert.deepEqual(inventory.errors, ['Inventory: directory could not be scanned']);
assert.equal(contract.depthCacheSelection(inventory.entries, ['a'.repeat(64), 'b'.repeat(64)]).length, 1);
assert.deepEqual(contract.depthCacheCleanupRequest(inventory.entries, ['a'.repeat(64), 'b'.repeat(64)]), { keys: ['a'.repeat(64)] });
const partial = contract.depthCacheCleanupSummary(['a'.repeat(64), 'c'.repeat(64)], {
  deleted: ['a'.repeat(64)], errors: [{ key: 'c'.repeat(64), error: 'file is busy' }]
}, true);
assert.match(partial, /Removed 1 of 2 selected/);
assert.match(partial, /c{64}: file is busy/);
assert.match(contract.depthCacheCleanupSummary(['a'.repeat(64)], { deleted: [], errors: [] }, false), /inventory could not be refreshed/i);
assert.throws(() => contract.validateDepthCacheInventory({ ...inventory, entries: [{ ...inventory.entries[0], key: 'UPPER' }] }), /invalid depth cache entry/i);
""")


if __name__ == "__main__":
    unittest.main()
