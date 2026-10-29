"""Lower fixed BatchMatMul to TensorFlow broadcast multiply + sum, max rank 4.

This is an offline graph transformation; execution remains in the NPU model.
The source graph is frozen using TensorFlow, and its public signature is kept.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('destination', type=Path)
a = p.parse_args()
loaded = tf.saved_model.load(str(a.source))
original = loaded.signatures['serving_default']
frozen = convert_variables_to_constants_v2(original)
graph = frozen.graph.as_graph_def(add_shapes=True)
replacement, reports = [], []
for node in graph.node:
    if node.op not in ('BatchMatMulV2', 'BatchMatMul'):
        replacement.append(node)
        continue
    tensor = frozen.graph.get_tensor_by_name(node.name + ':0')
    left = frozen.graph.get_tensor_by_name(node.input[0] if ':' in node.input[0] else node.input[0] + ':0')
    right = frozen.graph.get_tensor_by_name(node.input[1] if ':' in node.input[1] else node.input[1] + ':0')
    ls, rs, shape = left.shape.as_list(), right.shape.as_list(), tensor.shape.as_list()
    if not all(isinstance(d, int) and d > 0 for s in (ls, rs, shape) for d in s):
        raise ValueError(f'Dynamic attention shape: {node.name}')
    if ls[:-2] != rs[:-2]:
        raise ValueError(f'Broadcast batch dimensions need explicit export: {ls}, {rs}')
    subgraph = tf.Graph()
    with subgraph.as_default():
        x = tf.compat.v1.placeholder(left.dtype, ls, name='left')
        y = tf.compat.v1.placeholder(right.dtype, rs, name='right')
        with tf.name_scope(node.name + '/fydnn_reduce'):
            if node.attr['adj_x'].b:
                x = tf.linalg.matrix_transpose(x)
            if node.attr['adj_y'].b:
                y = tf.linalg.matrix_transpose(y)
            batch = int(np.prod(shape[:-2]))
            m, k, n = x.shape[-2], x.shape[-1], y.shape[-1]
            if k != y.shape[-2]:
                raise ValueError(f'MatMul contraction mismatch: {node.name}')
            x = tf.reshape(x, [batch, m, k, 1])
            y = tf.reshape(y, [batch, 1, k, n])
            # Neuron 9.0.11's MVPU broadcast kernel aborts on these two
            # simultaneous broadcast axes. Materialize equal shapes explicitly.
            # Gather repeats singleton dimensions without the TFLite optimizer
            # folding Tile + Mul straight back into a broadcasting Mul.
            x = tf.gather(x, tf.zeros([n], tf.int32), axis=3)
            y = tf.gather(y, tf.zeros([m], tf.int32), axis=1)
            z = tf.reduce_sum(tf.multiply(x, y), axis=2)
        tf.reshape(z, shape, name=node.name)
    for added in subgraph.as_graph_def(add_shapes=True).node:
        if added.op == 'Placeholder':
            continue
        for i, name in enumerate(added.input):
            if name == 'left':
                added.input[i] = node.input[0]
            elif name == 'right':
                added.input[i] = node.input[1]
        replacement.append(added)
    reports.append({'name': node.name, 'left': ls, 'right': rs, 'output': shape,
                    'adj_x': node.attr['adj_x'].b, 'adj_y': node.attr['adj_y'].b})
if not reports:
    raise ValueError('Source contains no BatchMatMul')
del graph.node[:]
graph.node.extend(replacement)
wrapped = tf.compat.v1.wrap_function(lambda: tf.import_graph_def(graph, name=''), [])
pruned = wrapped.prune([wrapped.graph.get_tensor_by_name(t.name) for t in frozen.inputs],
                       [wrapped.graph.get_tensor_by_name(t.name) for t in frozen.outputs])
spec = tf.TensorSpec(frozen.inputs[0].shape, frozen.inputs[0].dtype, name='images')

class Model(tf.Module):
    @tf.function(input_signature=[spec])
    def serve(self, images):
        return {'output0': pruned(images)[0]}

model = Model()
sample = np.random.default_rng(26).random(spec.shape, dtype=np.float32)
ref = list(original(tf.constant(sample)).values())[0].numpy()
actual = model.serve(tf.constant(sample))['output0'].numpy()
error = float(np.max(np.abs(actual - ref)))
if not np.isfinite(actual).all() or error > .001:
    raise ValueError(f'Attention lowering parity failed: {error}')
tf.saved_model.save(model, str(a.destination), signatures={'serving_default': model.serve})
(a.destination / 'attention-lowering.json').write_text(json.dumps({
    'source': str(a.source.resolve()), 'transformation': 'BatchMatMul -> fixed reshape, multiply, reduce_sum',
    'max_runtime_rank': 4, 'deterministic_input_max_absolute_error': error,
    'operators': reports, 'device_validation': 'separate'}, indent=2), encoding='utf-8')
print(json.dumps({'replaced': len(reports), 'max_error': error, 'destination': str(a.destination)}), flush=True)
