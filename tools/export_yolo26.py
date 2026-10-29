"""Use the pinned official Ultralytics YOLOv8 / YOLO26 dense exporter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--family',choices=['v8','26'],default='26')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--size', type=int, default=320)
p.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], default='n')
p.add_argument('--weights', type=Path, help='Optional local YOLOv8 / YOLO26 detection .pt')
a = p.parse_args()
if a.size < 32 or a.size % 32:
    raise ValueError('Fixed input must be a multiple of 32')
output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
weights = a.weights.resolve() if a.weights else None
if weights and not weights.is_file():
    raise FileNotFoundError(weights)
os.chdir(output)
torch.set_num_threads(4)
stem = f'yolo{a.family}{a.model}'
model = YOLO(str(weights) if weights else stem + '.pt')
if model.task != 'detect':
    raise ValueError('This exporter requires an object detection model')
architecture = model.model.yaml
expected_architecture=yaml_model_load(stem+'.yaml')
if architecture.get('backbone')!=expected_architecture['backbone'] or architecture.get('head')!=expected_architecture['head']:
    raise ValueError('Checkpoint topology does not match the selected official family')
if 'scales' in architecture:
    if architecture.get('scale')!=a.model or architecture['scales'].get(a.model)!=expected_architecture['scales'][a.model]:
        raise ValueError('Selected model scale does not match checkpoint architecture')
elif a.family=='v8':
    # Official older v8 checkpoints encode n/s with explicit width/depth,
    # rather than the newer yaml_file + scale metadata.
    depth,width,_=expected_architecture['scales'][a.model]
    if architecture.get('depth_multiple')!=depth or architecture.get('width_multiple')!=width:
        raise ValueError('Selected v8 scale does not match checkpoint depth/width')
else:
    raise ValueError('Expected YOLO26 checkpoint scale metadata')
weights_path = Path(model.ckpt_path).resolve()
weights_target = output / (stem + '.pt')
if weights_path != weights_target:
    shutil.copy2(weights_path, weights_target)
# Current official exporter uses nms=None for the dense one-to-many head.
# nms=False opts into the six-column one-to-one head with embedded TopK.
exported = Path(model.export(format='onnx', imgsz=a.size, batch=1, dynamic=False,
                             simplify=False, opset=17, device='cpu', nms=None if a.family=='26' else False, verbose=False))
target = output / f'{stem}_{a.size}.onnx'
if exported.resolve() != target:
    shutil.copy2(exported, target)
graph = onnx.load(target); onnx.checker.check_model(graph)
shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
assert len(graph.graph.input) == len(graph.graph.output) == 1
assert shape(graph.graph.input[0]) == [1, 3, a.size, a.size]
assert shape(graph.graph.output[0]) == [1, len(model.names)+4, sum((a.size // s) ** 2 for s in (8, 16, 32))]
session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
sample = np.random.default_rng(26).random((1, 3, a.size, a.size), dtype=np.float32)
with torch.no_grad():
    model.model.eval()
    if a.family=='26': model.model.end2end = False
    expected = model.model(torch.from_numpy(sample))[0].numpy()
actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
coord_error = float(np.max(np.abs(expected[:, :4] - actual[:, :4])))
score_error = float(np.max(np.abs(expected[:, 4:] - actual[:, 4:])))
if coord_error > .05 or score_error > .001:
    raise ValueError(f'PyTorch/ONNX mismatch: {coord_error}, {score_error}')
metadata = {
    'model': f'Ultralytics {stem.upper()} detection ({len(model.names)} classes)', 'size': [a.size, a.size],
    'weights_source': str(weights) if weights else f'official {stem}.pt',
    'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
    'weights_sha256': hashlib.sha256(weights_target.read_bytes()).hexdigest(),
    'onnx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'export': {'format': 'onnx', 'nms': None if a.family=='26' else False, 'head': 'dense decoded', 'opset': 17, 'dynamic': False},
    'input_shape': shape(graph.graph.input[0]), 'output_shape': shape(graph.graph.output[0]),
    'pytorch_onnx_max_coordinate_error_pixels': coord_error,
    'pytorch_onnx_max_probability_error': score_error, 'classes': model.names,
    'validation': 'Deterministic synthetic input; TFLite and dataset evaluation are separate',
    'license': 'AGPL-3.0',
}
target.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(json.dumps(metadata, ensure_ascii=True), flush=True)
