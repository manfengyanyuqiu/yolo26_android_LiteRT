"""Check NPU exports against host execution of the exact device input."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import tensorflow as tf

root = Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--models',type=Path,default=root/'models/yolo26n-320-mergefix-v061')
p.add_argument('--report',type=Path,default=root/'.deps/yolo26-mergefix-device')
p.add_argument('--image',type=Path,default=root/'.deps/yolov5/coco128/images/train2017/000000000009.jpg')
p.add_argument('--benchmark',action='store_true')
p.add_argument('--manifest', help='Optional model list with name and classes for graph probes')
p.add_argument('--pattern',default='*.tflite')
a=p.parse_args()
a.report.mkdir(parents=True,exist_ok=True)
adb=str(root/'.toolchain/sdk/platform-tools/adb.exe')
def run(*args,timeout=45):
    completed=subprocess.run([adb,*args],capture_output=True,timeout=timeout)
    if completed.returncode: raise RuntimeError((completed.stdout+completed.stderr).decode('utf-8',errors='replace'))
    return completed.stdout
run('push',str(a.image),'/data/local/tmp/y26-verify.jpg')
run('shell','run-as','com.fydnn.dimensity','cp','/data/local/tmp/y26-verify.jpg','files/y26-verify.jpg')
reports=[]
specs=json.loads((a.models/a.manifest).read_text(encoding='utf-8')) if a.manifest else None
models=[(a.models/(s['name']+'.tflite'),s.get('classes',80)) for s in specs] if specs else [(m,80) for m in sorted(a.models.glob(a.pattern))]
for model,classes in models:
    tag=model.stem
    directory=a.report/tag
    directory.mkdir(exist_ok=True)
    run('push',str(model),'/data/local/tmp/'+model.name)
    run('shell','run-as','com.fydnn.dimensity','cp','/data/local/tmp/'+model.name,'files/'+model.name)
    try:
        output=run('shell','am','instrument','-w','-r','-e','model',model.name,'-e','family','YOLO26',
                   '-e','layout','CHANNELS','-e','classes',str(classes),'-e','normalized','true','-e','image','y26-verify.jpg',
                   '-e','benchmark',str(a.benchmark).lower(),
                   'com.fydnn.dimensity.test/com.fydnn.dimensity.NpuDeviceInstrumentation').decode('utf-8',errors='replace')
    except subprocess.TimeoutExpired:
        run('shell','am','force-stop','com.fydnn.dimensity')
        output='TIMEOUT after 45 seconds; target stopped'
    (directory/'instrumentation.txt').write_text(output,encoding='utf-8')
    report=run('exec-out','run-as','com.fydnn.dimensity','cat','files/device-test.txt').decode('utf-8',errors='replace')
    (directory/'device-test.txt').write_text(report,encoding='utf-8')
    if 'NPU compile/run PASS' not in output or not report.rstrip().endswith('PASS'):
        item={'model':model.name,'npu_execution':'FAIL','output':output,'report':report}
        reports.append(item);print(json.dumps(item),flush=True)
        (a.report/'device-validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        continue
    interpreter=tf.lite.Interpreter(model_path=str(model),num_threads=1)
    interpreter.allocate_tensors()
    inp,out=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
    raw_input=run('exec-out','run-as','com.fydnn.dimensity','cat','files/inputBuffer.bin')
    raw_output=run('exec-out','run-as','com.fydnn.dimensity','cat','files/outputBuffer.bin')
    (directory/'input.bin').write_bytes(raw_input)
    (directory/'npu-output.bin').write_bytes(raw_output)
    x=np.frombuffer(raw_input,dtype=inp['dtype']).reshape(inp['shape']).copy()
    actual=np.frombuffer(raw_output,dtype=out['dtype']).reshape(out['shape']).copy()
    interpreter.set_tensor(inp['index'],x); interpreter.invoke()
    expected=interpreter.get_tensor(out['index'])
    expected.tofile(directory/'host-output.bin')
    npu=actual.astype(np.float32); host=expected.astype(np.float32)
    if np.issubdtype(out['dtype'],np.integer):
        scale,zero=out['quantization']; npu=(npu-zero)*scale; host=(host-zero)*scale
    if not np.isfinite(npu).all(): raise ValueError('Non-finite NPU output')
    coord=np.abs(npu[:,:,:4,:]-host[:,:,:4,:])*320
    score=np.abs(npu[:,:,4:,:]-host[:,:,4:,:])
    def statistics(label):
        lines=[line.split('=',1)[1] for line in report.splitlines() if line.startswith(label+'=')]
        return json.loads(lines[0]) if lines else None
    item={'model':model.name,'sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
          'image':a.image.name,'image_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),
          'npu_execution':'PASS','coordinate_max_error_pixels':float(coord.max()),
          'coordinate_mean_error_pixels':float(coord.mean()),
          'score_max_error':float(score.max()),'score_mean_error':float(score.mean()),
          'score_p99_error':float(np.quantile(score,.99)),
          'output_max_absolute_error':float(np.max(np.abs(npu-host))),
          'output_mean_absolute_error':float(np.mean(np.abs(npu-host))),
          'reference_mean_magnitude':float(np.mean(np.abs(host))),
          'backend_avg_p50_p95_ms':statistics('benchmark_avg_p50_p95_ms'),
          'image_end_to_end_avg_p50_p95_ms':statistics('image_end_to_end_avg_p50_p95_ms')}
    reports.append(item);print(json.dumps(item),flush=True)
    (a.report/'device-validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
(a.report/'device-validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
