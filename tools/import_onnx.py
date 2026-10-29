"""Validate a fixed decoded YOLO ONNX source, then stage an unmodified copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import onnx
import onnxruntime as ort

def inspect(source, family, size, model=None):
    model=onnx.load(source) if model is None else model
    onnx.checker.check_model(model)
    if len(model.graph.input)!=1 or len(model.graph.output)!=1:
        raise ValueError('Expected one RGB input and one decoded output without NMS')
    shape=lambda t:[d.dim_value if not d.dim_param else None for d in t.type.tensor_type.shape.dim]
    ins,outs=shape(model.graph.input[0]),shape(model.graph.output[0])
    if ins != [1,3,size,size]:
        raise ValueError(f'ONNX input is {ins}; select the matching fixed size. Re-export dynamic or differently sized models first.')
    if model.graph.input[0].type.tensor_type.elem_type not in (onnx.TensorProto.FLOAT,onnx.TensorProto.FLOAT16):
        raise ValueError('Source ONNX input must be floating point RGB [0,1]')
    candidates=sum((size//s)**2 for s in (8,16,32))
    if family=='v5':
        if len(outs)!=3 or outs[:2]!=[1,3*candidates] or not outs[2] or outs[2]<=5:
            raise ValueError(f'Expected legacy YOLOv5 decoded [1,{3*candidates},5+classes], got {outs}')
        classes=outs[2]-5; layout='ROWS'
        if not any(n.name.startswith('/model.24/') for n in model.graph.node):
            raise ValueError('This YOLOv5 adapter requires the upstream v7-style /model.24/ anchor head')
    else:
        if len(outs)!=3 or outs[0]!=1 or outs[2]!=candidates or not outs[1] or outs[1]<=4:
            raise ValueError(f'Expected dense [1,4+classes,{candidates}], got {outs}; embedded NMS/end-to-end output is not accepted')
        classes=outs[1]-4; layout='CHANNELS'
        if family=='26' and not all(any(n.name==name for n in model.graph.node) for name in ('/model.23/Sub','/model.23/Mul_2','/model.23/Concat_3')):
            raise ValueError('YOLO26 source must retain the supported upstream decoded-head names')
        if family=='26' and not any('attn/pe/conv/Conv' in n.name for n in model.graph.node):
            raise ValueError('YOLO26 Attention topology not present; check selected family')
    if any(n.op_type in ('NonMaxSuppression','EfficientNMS_TRT','BatchedNMSDynamic_TRT') for n in model.graph.node):
        raise ValueError('Export a dense decoded output without embedded NMS')
    metadata={e.key:e.value for e in model.metadata_props}
    description=metadata.get('description','').lower()
    if description and (('yolo26' in description and family!='26') or ('yolov8' in description and family!='v8')):
        raise ValueError('ONNX family metadata conflicts with the selected family')
    session=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider'])
    dtype=np.float16 if session.get_inputs()[0].type=='tensor(float16)' else np.float32
    output=session.run(None,{session.get_inputs()[0].name:np.full(ins,114/255,dtype=dtype)})[0]
    if list(output.shape)!=outs or not np.isfinite(output).all():
        raise ValueError('ONNX host execution produced an unexpected shape or non-finite output')
    return {'family':family,'classes':classes,'source_layout':layout,'input_shape':ins,'output_shape':outs,
            'source_input_dtype':str(dtype),'source_coordinates':'pixel cxcywh; RGB [0,1] input',
            'onnx_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_path':str(source.resolve()),'metadata':metadata,
            'validation':'ONNX checker + host execution of neutral image; not PyTorch parity or NPU validation'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--family',choices=['v5','v8','26'],required=True)
    p.add_argument('--size',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--name',required=True);a=p.parse_args()
    model=onnx.load(a.source)
    dims=model.graph.input[0].type.tensor_type.shape.dim
    dynamic=any(d.dim_param or not d.dim_value for d in dims)
    specialization=None
    if dynamic:
        if len(dims)!=4 or (dims[0].dim_value not in (0,1)) or (dims[1].dim_value not in (0,3)):
            raise ValueError('Dynamic source must have one NCHW RGB input')
        if any(d.dim_value and d.dim_value!=a.size for d in dims[2:]):
            raise ValueError('Fixed ONNX spatial dimensions must match the selected size')
        from onnxsim import simplify
        input_shape=[1,3,a.size,a.size]
        original=ort.InferenceSession(str(a.source),providers=['CPUExecutionProvider'])
        dtype=np.float16 if original.get_inputs()[0].type=='tensor(float16)' else np.float32
        sample=np.random.default_rng(8).random(input_shape).astype(dtype)
        expected=original.run(None,{original.get_inputs()[0].name:sample})[0]
        model,valid=simplify(model,overwrite_input_shapes={model.graph.input[0].name:input_shape})
        if not valid:raise ValueError('Dynamic ONNX specialization failed onnxsim validation')
        static=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider'])
        actual=static.run(None,{static.get_inputs()[0].name:sample})[0]
        if actual.shape!=expected.shape:raise ValueError('Dynamic ONNX specialization changed output shape')
        error=float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32))))
        if error>.001:raise ValueError(f'Dynamic ONNX specialization changed output values: {error}')
        specialization={'method':'onnxsim overwrite_input_shapes','shape':input_shape,'max_output_error':error}
    report=inspect(a.source.resolve(),a.family,a.size,model)
    report['source_file_sha256']=hashlib.sha256(a.source.read_bytes()).hexdigest()
    report['specialization']=specialization
    a.output.mkdir(parents=True,exist_ok=True)
    destination=a.output/(a.name+'.onnx')
    if dynamic:
        if destination.resolve()==a.source.resolve():raise ValueError('Specialized output must not overwrite the source')
        shutil.copy2(a.source,destination.with_suffix('.original.onnx'))
        onnx.save(model,destination)
    elif destination.resolve()!=a.source.resolve():shutil.copy2(a.source,destination)
    report['onnx_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
    report_path=destination.with_suffix('.source.json')
    if report_path.is_file():
        previous=json.loads(report_path.read_text(encoding='utf-8'))
        report['upstream_export_report']=previous
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)
