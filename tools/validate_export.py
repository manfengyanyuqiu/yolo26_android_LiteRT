"""Check final outputs against ONNX; report low-score box errors separately."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from quantize_yolo import letterbox

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--models',type=Path,required=True)
p.add_argument('--onnx',type=Path,required=True)
p.add_argument('--image',type=Path,required=True)
p.add_argument('--size',type=int,required=True)
p.add_argument('--family',choices=['v5','v8','26'],default='26')
p.add_argument('--classes',type=int,default=80)
a=p.parse_args()
x,_=letterbox(a.image,a.size)
session=ort.InferenceSession(str(a.onnx),providers=['CPUExecutionProvider'])
offset=5 if a.family=='v5' else 4
channels=a.classes+offset
def channel_layout(value):
    if value.ndim==4:
        if value.shape[1]==1: value=value[:,0]
        elif value.shape[-1]==1: value=value[...,0]
        else: raise ValueError(f'Unexpected four dimensional output: {value.shape}')
    if value.ndim!=3 or value.shape[0]!=1:
        raise ValueError(f'Unexpected output: {value.shape}')
    if value.shape[1]==channels:return value.astype(np.float32)
    if value.shape[2]==channels:return value.transpose(0,2,1).astype(np.float32)
    raise ValueError(f'Expected {channels} output channels, got {value.shape}')
dtype=np.float16 if session.get_inputs()[0].type=='tensor(float16)' else np.float32
y=channel_layout(session.run(None,{session.get_inputs()[0].name:x.transpose(0,3,1,2).astype(dtype)})[0])
scores=y[:,5:]*y[:,4:5] if a.family=='v5' else y[:,4:]
active=np.max(scores,axis=1) >= .1
reports=[]
for path in sorted(a.models.glob('*.tflite')):
    it=tf.lite.Interpreter(model_content=path.read_bytes(),num_threads=1); it.allocate_tensors()
    ins,outs=it.get_input_details()[0],it.get_output_details()[0]
    candidates=sum((a.size//s)**2 for s in (8,16,32))*(3 if a.family=='v5' else 1)
    expected=([1,1,channels,candidates],[1,1,candidates,channels],[1,channels,candidates,1],[1,candidates,channels,1])
    if ins['shape'].tolist()!=[1,a.size,a.size,3] or outs['shape'].tolist() not in expected:
        raise ValueError(f'Unexpected input/output: {path}')
    sample=x
    if np.issubdtype(ins['dtype'],np.integer):
        scale,zero=ins['quantization']; info=np.iinfo(ins['dtype'])
        sample=np.clip(np.rint(x/scale)+zero,info.min,info.max).astype(ins['dtype'])
    it.set_tensor(ins['index'],sample); it.invoke()
    z=it.get_tensor(outs['index']).astype(np.float32)
    if np.issubdtype(outs['dtype'],np.integer):
        scale,zero=outs['quantization']; z=(z-zero)*scale
    z=channel_layout(z)
    if z.shape!=y.shape or not np.isfinite(z).all(): raise ValueError(f'Invalid output: {path}')
    z[:,:4]*=a.size
    error=np.abs(z-y)
    row={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'family':a.family,'classes':a.classes,'output_shape':outs['shape'].tolist(),
         'coordinate_max_error_pixels':float(error[:,:4].max()),
         'coordinate_mean_error_pixels':float(error[:,:4].mean()),
         'score_max_error':float(error[:,4:].max()),'active_threshold':.1,
         'active_reference_candidates':int(active.sum()),
         'active_coordinate_max_error_pixels':float(error[:,:4,active[0]].max()) if active.any() else None,
         'validation_scope':'Single real image host comparison; not dataset accuracy or NPU validation'}
    if 'int8' in path.name:
        row['status']='HOST_EXECUTION_PASS; quantized accuracy requires separate dataset evaluation'
    elif 'fp16weights' in path.name or dtype==np.float16:
        if row['score_max_error']>.05 or (active.any() and row['active_coordinate_max_error_pixels']>2):
            raise ValueError(f'FP16 parity failed: {row}')
        row['source_precision']='FP16' if dtype==np.float16 else 'FP32'
        row['comparison_policy']='FP16 source/weights: scores <=0.05 and active boxes <=2 pixels'
        row['status']='HOST_PARITY_PASS'
    else:
        if row['coordinate_max_error_pixels']>.05 or row['score_max_error']>.001:
            raise ValueError(f'FP32 parity failed: {row}')
        row['status']='HOST_PARITY_PASS'
    reports.append(row); print(json.dumps(row),flush=True)
(a.models/'final-parity.json').write_text(json.dumps({'image':a.image.name,
    'image_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),'reports':reports},indent=2),encoding='utf-8')
