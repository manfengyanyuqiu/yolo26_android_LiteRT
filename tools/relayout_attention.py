"""Produce an equivalent channel-last attention output for device diagnosis."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
root = Path(__file__).resolve().parent.parent
loaded = tf.saved_model.load(str(root / 'models/yolo26n-320-mediatek'))
frozen = convert_variables_to_constants_v2(loaded.signatures['serving_default'])
graph = frozen.graph.as_graph_def(add_shapes=True)
prefix = 'StatefulPartitionedCall/model_38/'
targets = {'tf.compat.v1.transpose_56/transpose': 'tf.linalg.matmul_3/MatMul',
           'tf.compat.v1.transpose_149/transpose': 'tf.linalg.matmul_7/MatMul'}
replacement = []
for node in graph.node:
    source = targets.get(node.name.removeprefix(prefix))
    if source is None:
        replacement.append(node); continue
    matmul = frozen.graph.get_operation_by_name(prefix + source)
    subgraph = tf.Graph()
    with subgraph.as_default():
        left = tf.compat.v1.placeholder(tf.float32, matmul.inputs[0].shape, name='left')
        right = tf.compat.v1.placeholder(tf.float32, matmul.inputs[1].shape, name='right')
        with tf.name_scope(node.name + '/channel_last'):
            product = tf.matmul(right, left, transpose_a=True, transpose_b=True)
            spatial = tf.transpose(product, [0, 2, 1, 3])
        tf.reshape(spatial, [1, 10, 10, 128], name=node.name)
    for added in subgraph.as_graph_def(add_shapes=True).node:
        if added.op == 'Placeholder': continue
        for index, value in enumerate(added.input):
            if value == 'left': added.input[index] = matmul.inputs[0].name
            elif value == 'right': added.input[index] = matmul.inputs[1].name
        replacement.append(added)
del graph.node[:]
graph.node.extend(replacement)
wrapped = tf.compat.v1.wrap_function(lambda: tf.import_graph_def(graph, name=''), [])
pruned = wrapped.prune([wrapped.graph.get_tensor_by_name(t.name) for t in frozen.inputs],
                       [wrapped.graph.get_tensor_by_name(t.name) for t in frozen.outputs])
class Model(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 320, 320, 3], tf.float32, name='images')])
    def serve(self, images):
        return {'output0': pruned(images)[0]}
model = Model()
sample = np.random.default_rng(26).random([1,320,320,3], dtype=np.float32)
expected = frozen(tf.constant(sample))[0].numpy()
actual = model.serve(tf.constant(sample))['output0'].numpy()
error = float(np.max(np.abs(actual - expected)))
if not np.isfinite(actual).all() or error > .001: raise ValueError(error)
destination = root / '.deps/yolo26-relayout-saved'
tf.saved_model.save(model, str(destination), signatures={'serving_default': model.serve})
converter = tf.lite.TFLiteConverter.from_saved_model(str(destination))
content = converter.convert()
(destination / 'yolo26-relayout-fp32.tflite').write_bytes(content)
(destination / 'relayout.json').write_text(json.dumps({'max_error':error,'bytes':len(content),'device_status':'PENDING'}),encoding='utf-8')
print(json.dumps({'max_error': error,'bytes':len(content)}))
