"""Reproducible, local DirectML graph variant; upstream model bytes stay intact.

Changing cubic Resize to linear changes inference semantics. This helper records
both model hashes and the exact transformation version instead of claiming parity.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

TRANSFORM_VERSION = 'directml-linear-resize-v1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _quarantine(path: Path, directory: Path) -> None:
    if path.exists():
        target = directory / '_quarantine' / uuid.uuid4().hex / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)


def prepare_directml_model(source: Path, source_sha256: str, directory: Path) -> tuple[Path, dict]:
    """Validate or atomically derive a cache identified by source hash and policy."""
    import onnx
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{source_sha256}-{TRANSFORM_VERSION}.onnx'
    receipt = target.with_suffix('.json')
    try:
        info = json.loads(receipt.read_text(encoding='utf-8'))
        if (info['source_sha256'] == source_sha256 and info['transform_version'] == TRANSFORM_VERSION
                and target.stat().st_size == info['derived_size'] and sha256(target) == info['derived_sha256']):
            return target, info
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if sha256(source) != source_sha256:
        raise RuntimeError('Source ONNX failed SHA256 verification before graph optimization.')
    model = onnx.load(str(source), load_external_data=False)
    # These pinned single-file models must not resolve arbitrary external files.
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise RuntimeError('External ONNX tensor data is unsupported for cached graph optimization.')
    inputs = [item for item in model.graph.input if item.name not in {t.name for t in model.graph.initializer}]
    if len(inputs) != 1:
        raise RuntimeError('Depth model must have one image input.')
    item = inputs[0]
    tensor = item.type.tensor_type
    shape = [d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in tensor.shape.dim]
    type_name = {onnx.TensorProto.FLOAT: 'tensor(float)', onnx.TensorProto.FLOAT16: 'tensor(float16)'}.get(tensor.elem_type)
    if len(shape) != 4 or type_name is None:
        raise RuntimeError('Depth model must accept one float32 or float16 NCHW image tensor.')
    changed = []
    def patch_graph(graph):
        for node in graph.node:
            if node.op_type == 'Resize' and node.domain in ('', 'ai.onnx'):
                for attr in node.attribute:
                    if attr.name == 'mode' and attr.s == b'cubic':
                        attr.s = b'linear'
                        changed.append(node.name or node.output[0])
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    patch_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child in attr.graphs:
                        patch_graph(child)
    patch_graph(model.graph)
    onnx.checker.check_model(model)
    partial = target.with_name(f'{target.name}.{uuid.uuid4().hex}.part')
    receipt_partial = receipt.with_name(f'{receipt.name}.{uuid.uuid4().hex}.part')
    try:
        with partial.open('xb') as output:
            output.write(model.SerializeToString())
            output.flush()
            os.fsync(output.fileno())
        info = {'transform_version': TRANSFORM_VERSION, 'onnx_version': onnx.__version__,
                'opsets': {item.domain or 'ai.onnx': item.version for item in model.opset_import},
                'source_sha256': source_sha256,
                'derived_sha256': sha256(partial), 'derived_size': partial.stat().st_size,
                'cubic_resize_nodes_changed': len(changed), 'changed_nodes': changed,
                'semantic_change': 'ONNX Resize cubic replaced with linear' if changed else None,
                'input': {'name': item.name, 'shape': shape, 'type': type_name}}
        with receipt_partial.open('x', encoding='utf-8') as output:
            json.dump(info, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        _quarantine(target, directory)
        _quarantine(receipt, directory)
        os.replace(partial, target)
        os.replace(receipt_partial, receipt)
        return target, info
    finally:
        _quarantine(partial, directory)
        _quarantine(receipt_partial, directory)
