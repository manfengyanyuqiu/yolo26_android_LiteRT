"""Full-integer PTQ with real RGB images and the app's letterbox convention."""
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


def letterbox(path, size):
    image = tf.io.decode_image(Path(path).read_bytes(), channels=3, expand_animations=False)
    h, w = int(image.shape[0]), int(image.shape[1])
    ratio = min(size / w, size / h)
    rw, rh = max(1, int(w*ratio+.5)), max(1, int(h*ratio+.5))
    px, py = (size-rw)//2, (size-rh)//2
    resized = tf.image.resize(tf.cast(image, tf.float32)/255., [rh, rw])[None]
    return tf.pad(resized, [[0,0],[py,size-rh-py],[px,size-rw-px],[0,0]],
                  constant_values=114/255.).numpy(), (w,h,rw,rh,px,py)


def dataset_split(directory):
    images = sorted(p for p in Path(directory).glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if len(images) < 32:
        raise ValueError('At least 32 real calibration/evaluation images required')
    # Deterministic disjoint split: 96 calibration / 32 evaluation for COCO128.
    evaluation = images[::4]
    calibration = [p for p in images if p not in set(evaluation)]
    return calibration, evaluation


def quantize(saved_model, destination, directory, size, disable_per_channel=False):
    calibration, evaluation = dataset_split(directory)
    def representative():
        for image in calibration:
            yield [letterbox(image, size)[0]]
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Some MediaTek Neuron versions reject per-channel convolution scales.
    # Keep this opt-in so general-purpose exports retain the accuracy benefit.
    converter._experimental_disable_per_channel = disable_per_channel
    converted = converter.convert()
    # Reject float islands in the actual flatbuffer (int32 shape/bias tensors are normal).
    interpreter = tf.lite.Interpreter(model_content=converted, num_threads=1)
    interpreter.allocate_tensors()
    float_tensors = [t['name'] for t in interpreter.get_tensor_details()
                     if np.issubdtype(t['dtype'], np.floating)]
    if float_tensors:
        raise ValueError(f'Full INT8 conversion left float tensors: {float_tensors}')
    destination.write_bytes(converted)
    report = {
        'file': destination.name, 'sha256': hashlib.sha256(converted).hexdigest(),
        'method': ('TFLite full integer post-training quantization; per-tensor weights'
                   if disable_per_channel else
                   'TFLite full integer post-training quantization; per-channel weights'),
        'weight_quantization_granularity': 'per_tensor' if disable_per_channel else 'per_channel',
        'dataset': str(Path(directory).resolve()),
        'calibration_images': [{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in calibration],
        'evaluation_images': [p.name for p in evaluation], 'float_tensor_count':len(float_tensors),
    }
    for key, tensors in [('input', interpreter.get_input_details()), ('output', interpreter.get_output_details())]:
        report[key] = [{'shape':t['shape'].tolist(),'dtype':str(t['dtype']),
                        'scale':t['quantization'][0], 'zero_point':t['quantization'][1]} for t in tensors]
    report['int8_tensor_count'] = sum(
        t['dtype'] == np.int8 for t in interpreter.get_tensor_details())
    report['per_channel_quantized_tensor_count'] = sum(
        len(t['quantization_parameters']['scales']) > 1
        for t in interpreter.get_tensor_details())
    destination.with_suffix('.quantization.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('calibration_images','evaluation_images')}), flush=True)
