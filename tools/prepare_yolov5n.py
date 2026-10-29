"""Retarget the official v7.0 fixed-shape ONNX; preserve every learned tensor."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper as nh


def prepare(source, output, size):
    if size % 32 or not 32 <= size <= 640:
        raise ValueError('Size must be a multiple of 32, between 32 and 640')
    model = onnx.load(source)
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if shape != [1, 3, 640, 640]:
        raise ValueError(f'Expected official v7.0 static export, got {shape}')
    changed = []
    for node in model.graph.node:
        if node.op_type != 'Constant' or not node.name.startswith('/model.24/'):
            continue
        attr = next(a for a in node.attribute if a.name == 'value')
        value = nh.to_array(attr.t)
        if value.ndim == 1 and value.size == 5:
            stride = 640 // int(value[3])
            value = np.array([1, 3, 85, size // stride, size // stride], dtype=value.dtype)
        elif value.ndim == 1 and value.size == 3:
            stride = int(round(640 / np.sqrt(value[1] / 3)))
            value = np.array([1, 3 * (size // stride)**2, 85], dtype=value.dtype)
        elif value.ndim == 5 and value.shape[-1] == 2:
            side = size // (640 // value.shape[2])
            value = value[:, :, :side, :side, :].copy()
        else:
            continue
        attr.t.CopyFrom(nh.from_array(value))
        changed.append(node.name)
    if len(changed) != 12:
        raise ValueError(f'Unexpected head topology: {len(changed)} shape/grid constants')
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = size
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = size
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = sum(3*(size//s)**2 for s in (8,16,32))
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    original = onnx.load(source)
    assert all(a.SerializeToString() == b.SerializeToString()
               for a,b in zip(original.graph.initializer, model.graph.initializer))
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    output.with_suffix('.source.json').write_text(json.dumps({
        'source_url': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'size': size, 'learned_tensors_unchanged': True,
        'changed_constants': changed,
        'method': 'Fully convolutional backbone; resize static head shapes and crop XY grids / broadcast anchor grids'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--size', type=int, default=224)
    a = p.parse_args()
    prepare(a.source, a.output, a.size)
