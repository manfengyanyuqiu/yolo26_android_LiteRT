"""Keep YOLO class Sigmoid in NHWC heads for MediaTek Neuron INT8 execution.

Elementwise Sigmoid commutes with reshape, transpose and concatenation. Move it
to each final class convolution, preserving all input/output quantization and
the public model contract. Unsupported/shared graph paths fail explicitly.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


def stabilize_head_sigmoid(content):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(content, 0))
    if len(model.subgraphs) != 1:
        raise ValueError('Expected one fixed YOLO subgraph')
    graph = model.subgraphs[0]
    if len(graph.outputs) != 1:
        raise ValueError('Expected one decoded YOLO output')
    shape = list(graph.tensors[graph.outputs[0]].shape)
    if len(shape) == 4 and shape[1] == 1:
        shape = [shape[0], shape[2], shape[3]]
    elif len(shape) == 4 and shape[3] == 1:
        shape = shape[:3]
    codes = [code.builtinCode for code in model.operatorCodes]
    targets = []
    for op in graph.operators:
        if codes[op.opcodeIndex] != schema.BuiltinOperator.LOGISTIC:
            continue
        tensor = graph.tensors[op.inputs[0]]
        dims = list(tensor.shape)
        if (tensor.type == schema.TensorType.INT8 and len(dims) == 3
                and len(shape) == 3 and dims[0] == shape[0] == 1
                and dims[1] + 4 == shape[1] and dims[2] == shape[2]):
            targets.append(op)
    if not targets:
        return content, {'applied': False, 'reason': 'No decoded rank-3 INT8 class Sigmoid'}
    if len(targets) != 1:
        raise ValueError('Ambiguous class Sigmoid topology')
    target = targets[0]
    source, destination = int(target.inputs[0]), int(target.outputs[0])
    quantization = graph.tensors[destination].quantization
    source_quantization = graph.tensors[source].quantization
    producers = {int(index): op for op in graph.operators for index in op.outputs}
    source_op = producers[source]
    if codes[source_op.opcodeIndex] == schema.BuiltinOperator.STRIDED_SLICE:
        # Combined box/class heads have different quantization boundaries. They
        # are outside this separate-class-head transform, not a repair failure.
        return content, {'applied': False, 'reason':
                         'Combined box/class slice topology; separate class-head transform not applicable'}
    leaves, paths = {}, set()

    def trace(index):
        op = producers[index]
        code = codes[op.opcodeIndex]
        if code == schema.BuiltinOperator.CONV_2D:
            leaf = graph.tensors[index]
            if (len(leaf.shape) != 4 or leaf.shape[0] != 1
                    or leaf.shape[3] != graph.tensors[source].shape[1]
                    or not np.array_equal(leaf.quantization.scale, source_quantization.scale)
                    or not np.array_equal(leaf.quantization.zeroPoint, source_quantization.zeroPoint)):
                raise ValueError('Class convolution layout/quantization differs from merged logits')
            leaves[index] = op
            return
        if code not in (schema.BuiltinOperator.RESHAPE, schema.BuiltinOperator.TRANSPOSE,
                        schema.BuiltinOperator.CONCATENATION):
            raise ValueError(f'Unsupported class Sigmoid path operator: {code}')
        paths.add(index)
        inputs = op.inputs if code == schema.BuiltinOperator.CONCATENATION else op.inputs[:1]
        for input_index in inputs:
            trace(int(input_index))

    trace(source)
    if len(leaves) != 3:
        raise ValueError(f'Expected three class heads, found {len(leaves)}')
    for op in graph.operators:
        if op is target or any(int(index) in paths for index in op.outputs):
            continue
        if any(int(index) in paths or int(index) in leaves for index in op.inputs):
            raise ValueError('Class logits are shared with another graph branch')
    mapped = {}
    for index in leaves:
        tensor = copy.deepcopy(graph.tensors[index])
        tensor.name += b'_neuron_head_sigmoid'
        tensor.buffer = 0
        tensor.quantization = copy.deepcopy(quantization)
        mapped[index] = len(graph.tensors)
        graph.tensors.append(tensor)
    for index in paths:
        graph.tensors[index].quantization = copy.deepcopy(quantization)
    operators = []
    for op in graph.operators:
        if op is target:
            continue
        op.inputs = np.array([mapped.get(int(index), source if int(index) == destination else int(index))
                              for index in op.inputs], dtype=np.int32)
        operators.append(op)
        for index, producer in leaves.items():
            if op is producer:
                sigmoid = copy.deepcopy(target)
                sigmoid.inputs = np.array([index], dtype=np.int32)
                sigmoid.outputs = np.array([mapped[index]], dtype=np.int32)
                operators.append(sigmoid)
    graph.operators = operators
    builder = flatbuffers.Builder(0)
    offset = model.Pack(builder)
    builder.Finish(offset, file_identifier=b'TFL3')
    fixed = bytes(builder.Output())
    return fixed, {'applied': True, 'method': 'Sigmoid before class-head layout and concatenation',
                   'head_count': len(leaves), 'output_shape': shape,
                   'source_sha256': hashlib.sha256(content).hexdigest(),
                   'sha256': hashlib.sha256(fixed).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.model.resolve():
        raise ValueError('Choose a separate output file to preserve the original model')
    content, report = stabilize_head_sigmoid(args.model.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    args.output.with_suffix('.neuron.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
