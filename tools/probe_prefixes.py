"""Export unchanged YOLO26 graph prefixes to locate device execution failure."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
root = Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('--source',default='models/yolo26n-320-mediatek')
p.add_argument('--tag',default='yolo26-prefix-probes')
p.add_argument('--parity',action='store_true')
a=p.parse_args()
destination = root / '.deps' / a.tag
destination.mkdir(exist_ok=True)
loaded = tf.saved_model.load(str(root / a.source))
frozen = convert_variables_to_constants_v2(loaded.signatures['serving_default'])
graph = frozen.graph.as_graph_def(add_shapes=True)
wrapped = tf.compat.v1.wrap_function(lambda: tf.import_graph_def(graph, name=''), [])
reports = []
for name, suffix in [('backbone','tf.math.add_48/Add'),('qk','tf.linalg.matmul_1/MatMul'),
                     ('softmax','tf.nn.softmax_1/wa/model.10/m/m.0/attn/Softmax'),
                     ('attention', 'tf.linalg.matmul_3/MatMul'), ('position', 'tf.math.add_49/Add'), ('combine', 'tf.math.add_51/Add')]:
    candidates = [op for op in frozen.graph.get_operations() if
                  (op.name == suffix if suffix == 'Identity' else op.name.endswith('/' + suffix))]
    if len(candidates) != 1: raise ValueError((name, [op.name for op in candidates]))
    operation = candidates[0]
    tensor = operation.outputs[0]
    channels = int(np.prod(tensor.shape.as_list())) // 100
    pruned = wrapped.prune(wrapped.graph.get_tensor_by_name(frozen.inputs[0].name),
                           wrapped.graph.get_tensor_by_name(tensor.name))
    def serve(images, pruned=pruned, channels=channels):
        values = tf.reshape(pruned(images), [-1])
        if a.parity: values=tf.math.sigmoid(values)
        return tf.reshape(values, [1, 1, channels, 100])
    fn = tf.function(serve, input_signature=[tf.TensorSpec([1, 320, 320, 3], tf.float32, name='images')])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], fn)
    content = converter.convert()
    name = 'fullprefix-' + name
    (destination / f'{name}.tflite').write_bytes(content)
    sample = np.full([1, 320, 320, 3], 114.0/255, np.float32)
    fn(sample).numpy().tofile(destination / f'{name}-reference.bin')
    reports.append({'name': name, 'node': operation.name, 'shape': tensor.shape.as_list(), 'bytes': len(content), 'classes': channels-4})
(destination / 'prefixes.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
print(json.dumps(reports))
