import hashlib
import importlib.util
import json
import os
import unittest
import uuid
from pathlib import Path

from onnx_optimization import prepare_directml_model, sha256, TRANSFORM_VERSION


@unittest.skipUnless(importlib.util.find_spec('onnx'), 'ONNX graph parser is unavailable')
class GraphOptimizationTests(unittest.TestCase):
    def setUp(self):
        import onnx
        from onnx import helper, TensorProto
        self.directory = Path(__file__).resolve().parents[1] / '.runtime' / 'onnx-optimization-tests' / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.source = self.directory / 'source.onnx'
        x = helper.make_tensor_value_info('image', TensorProto.FLOAT, ['batch', 3, 'height', 'width'])
        y = helper.make_tensor_value_info('depth', TensorProto.FLOAT, ['batch', 3, 'height', 'width'])
        scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 1, 1])
        node = helper.make_node('Resize', ['image', '', 'scales'], ['depth'], name='cubic', mode='cubic', coordinate_transformation_mode='half_pixel')
        model = helper.make_model(helper.make_graph([node], 'test', [x], [y], [scales]), opset_imports=[helper.make_opsetid('', 17)])
        onnx.save(model, self.source)
        self.source_hash = sha256(self.source)

    def tearDown(self):
        target = self.directory.parents[1] / '_quarantine' / 'onnx-optimization-tests' / self.directory.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.directory, target)

    def test_patch_preserves_source_records_semantic_change_and_reuses_verified_cache(self):
        import onnx
        target, info = prepare_directml_model(self.source, self.source_hash, self.directory / 'cache')
        self.assertEqual(sha256(self.source), self.source_hash)
        self.assertEqual(info['cubic_resize_nodes_changed'], 1)
        self.assertEqual(info['input']['shape'], ['batch', 3, 'height', 'width'])
        attrs = {a.name: a.s for a in onnx.load(target).graph.node[0].attribute}
        self.assertEqual(attrs['mode'], b'linear')
        self.assertEqual(attrs['coordinate_transformation_mode'], b'half_pixel')
        self.assertIn('cubic replaced with linear', info['semantic_change'])
        self.assertIn(self.source_hash, target.name)
        self.assertIn(TRANSFORM_VERSION, target.name)
        mtime = target.stat().st_mtime_ns
        again, metadata = prepare_directml_model(self.source, self.source_hash, self.directory / 'cache')
        self.assertEqual(again, target)
        self.assertEqual(metadata, info)
        self.assertEqual(target.stat().st_mtime_ns, mtime)

    def test_corrupt_cache_is_quarantined_and_regenerated(self):
        target, info = prepare_directml_model(self.source, self.source_hash, self.directory / 'cache')
        target.write_bytes(b'corrupt')
        again, metadata = prepare_directml_model(self.source, self.source_hash, self.directory / 'cache')
        self.assertEqual(again, target)
        self.assertEqual(sha256(target), info['derived_sha256'])
        self.assertTrue(list((target.parent / '_quarantine').rglob('*.onnx')))

    def test_rejects_source_hash_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            prepare_directml_model(self.source, '0' * 64, self.directory / 'cache')
        self.assertFalse(list((self.directory / 'cache').glob('*.onnx')))


if __name__ == '__main__':
    unittest.main()
