"""Export isolated YOLO26 attention operations for NPU device diagnosis."""
import json
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf

root = Path(__file__).resolve().parent.parent
destination = root / '.deps/yolo26-operator-probes'
destination.mkdir(exist_ok=True)
specs = [
    ('qk', [1, 2, 100, 64], lambda x: tf.matmul(x[..., :32], x[..., 32:], transpose_b=True)),
    ('softmax', [1, 2, 100, 100], lambda x: tf.nn.softmax(x, axis=-1)),
    ('av', [1, 2, 164, 100], lambda x: tf.matmul(x[:, :, :64, :], x[:, :, 64:, :])),
]
reports = []
for name, shape, operation in specs:
    fn = tf.function(operation, input_signature=[tf.TensorSpec(shape, tf.float32, name='images')])
    data = np.random.default_rng(26).uniform(-0.2, 0.2, shape).astype(np.float32)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], fn)
    content = converter.convert()
    model = destination / f'{name}.tflite'
    model.write_bytes(content)
    data.tofile(destination / f'{name}-input.bin')
    fn(data).numpy().tofile(destination / f'{name}-reference.bin')
    interpreter = tf.lite.Interpreter(model_content=content)
    reports.append({'name': name, 'input': shape, 'output': interpreter.get_output_details()[0]['shape'].tolist(),
                    'ops': [op['op_name'] for op in interpreter._get_ops_details()]})
    # Use the already-installed detector instrumentation without updating APKs.
    # Fixed RGB input and YOLO-shaped output only adapt the test boundary;
    # the operation between them retains the original attention matrix shapes.
    def detector_probe(rgb, shape=shape, operation=operation):
        matrix = tf.reshape(tf.reshape(rgb, [-1])[:int(np.prod(shape))], shape)
        result = operation(matrix)
        return tf.reshape(tf.reshape(result, [-1])[:4200], [1, 1, 84, 50])
    wrapped = tf.function(detector_probe, input_signature=[tf.TensorSpec([1, 128, 128, 3], tf.float32, name='images')])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([wrapped.get_concrete_function()], wrapped)
    content = converter.convert()
    (destination / f'{name}-detector.tflite').write_bytes(content)
    neutral = np.full([1, 128, 128, 3], 114.0/255, np.float32)
    wrapped(neutral).numpy().tofile(destination / f'{name}-detector-reference.bin')
(destination / 'probes.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
print(json.dumps(reports))
