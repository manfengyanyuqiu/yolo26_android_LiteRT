"""Export validated attention-merge rewrite as FP32, FP16 weights and full INT8."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tensorflow as tf
from quantize_yolo import quantize

root = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--saved-model', type=Path, default=root / '.deps/yolo26-nchw-add-saved')
p.add_argument('--output', type=Path, default=root / 'models/yolo26n-320-neuronfix-v061')
p.add_argument('--source-artifacts', type=Path, default=root / 'models/yolo26n-320-mediatek')
p.add_argument('--model', choices=['n','s','m','l','x'], default='n')
p.add_argument('--size', type=int, default=320)
p.add_argument('--formats', nargs='+', choices=['fp32','fp16','int8'], default=['fp32','fp16','int8'])
p.add_argument('--calibration-dir', type=Path, default=root / '.deps/yolov5/coco128/images/train2017')
a = p.parse_args()
source, destination = a.saved_model.resolve(), a.output.resolve()
destination.mkdir(parents=True, exist_ok=True)
stem = f'yolo26{a.model}_{a.size}_4d_norm_channels_mergefix'
for half, suffix in [(False, 'fp32'), (True, 'fp32io_fp16weights')]:
    if ('fp16' if half else 'fp32') not in a.formats: continue
    converter = tf.lite.TFLiteConverter.from_saved_model(str(source))
    if half:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    content = converter.convert()
    file = destination / f'{stem}_{suffix}.tflite'
    file.write_bytes(content)
    print(json.dumps({'file':file.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}),flush=True)
if 'int8' in a.formats:
    quantize(source, destination / f'{stem}_int8.tflite', a.calibration_dir.resolve(),
             a.size,disable_per_channel=True)
for name in [f'yolo26{a.model}.pt',f'yolo26{a.model}_{a.size}.onnx',f'yolo26{a.model}_{a.size}.source.json','onnx2tf-replacements.json']:
    original = a.source_artifacts.resolve() / name
    if original != destination / name: shutil.copy2(original, destination / name)
shutil.copy2(source / 'rewrite.json', destination / 'attention-merge-rewrite.json')
