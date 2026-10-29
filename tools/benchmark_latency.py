"""Run same-output decoder and local JPEG-to-detections A/B tests on MediaTek."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent.parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--family',choices=['YOLOV5','YOLOV8','YOLO26'],required=True)
    p.add_argument('--layout',choices=['ROWS','CHANNELS'],required=True)
    p.add_argument('--classes',type=int,default=80)
    p.add_argument('--normalized',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--image',type=Path,default=ROOT/'.deps/yolov5/coco128/images/train2017/000000000009.jpg')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device-only',action='store_true',
                   help='Run the NPU and optimized image path without the slow legacy decoder A/B')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    adb=[str(ROOT/'.toolchain/sdk/platform-tools/adb.exe')]
    if a.serial:adb+=['-s',a.serial]
    def run(*args,timeout=60):
        r=subprocess.run(adb+list(map(str,args)),capture_output=True,timeout=timeout)
        if r.returncode:raise RuntimeError((r.stdout+r.stderr).decode('utf-8','replace'))
        return r.stdout
    if run('get-state').strip()!=b'device':raise RuntimeError('Connect one USB debugging device')
    for source,name in [(a.model,'latency.tflite'),(a.image,'latency.jpg')]:
        run('push',source,'/data/local/tmp/fydnn-'+name)
        run('shell','run-as','com.fydnn.dimensity','cp','/data/local/tmp/fydnn-'+name,'files/'+name)
    args=['-e','model','latency.tflite','-e','image','latency.jpg','-e','family',a.family,
          '-e','layout',a.layout,'-e','classes',str(a.classes),'-e','normalized',str(a.normalized).lower()]
    tests=[('device',['-e','benchmark','true'])] if a.device_only else [
        ('local-ab',['-e','decoderBenchmark','true','-e','pipelineBenchmark','true'])]
    for name,options in tests:
        run('shell','run-as','com.fydnn.dimensity','rm','-f','files/device-test.txt')
        result=run('shell','am','instrument','-w','-r',*args,*options,
                   'com.fydnn.dimensity.test/com.fydnn.dimensity.NpuDeviceInstrumentation')
        (a.output/(name+'-instrumentation.txt')).write_bytes(result)
        if b'INSTRUMENTATION_CODE: -1' not in result:
            raise RuntimeError(name+' instrumentation failed: '+result.decode('utf-8','replace'))
        report=run('exec-out','run-as','com.fydnn.dimensity','cat','files/device-test.txt')
        (a.output/(name+'.txt')).write_bytes(report)
        text=report.decode('utf-8','replace')
        if not text.rstrip().endswith('PASS') or b'INSTRUMENTATION_CODE: -1' not in result:
            raise RuntimeError(name+' failed: '+text)
        if name=='local-ab' and ('decoder_exact_match=true' not in text or 'local_pipeline_boxes_exact_match=true' not in text):
            raise RuntimeError('Missing exact-output comparisons')
        print(text,flush=True)
    (a.output/'metadata.json').write_text(json.dumps({
        'model':str(a.model.resolve()),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'image_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),
        'family':a.family,'layout':a.layout,'classes':a.classes,'normalized':a.normalized,
        'decoder_warmup_pairs':0 if a.device_only else 10,
        'decoder_timed_pairs':0 if a.device_only else 60,
        'pipeline_warmup_pairs':0 if a.device_only else 10,
        'pipeline_timed_pairs':0 if a.device_only else 60,'order':'alternating',
        'local_ab_scope':None if a.device_only else 'JPEG decode + letterbox/RGB + NPU/sync readback + postprocessing; excludes capture and display',
        'device_scope':'NPU execution and image end-to-end path; excludes camera capture and display'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
