"""FYDNN YOLOv5 / YOLOv8 / YOLO26 exporter: ONNX or weights -> corrected TFLite."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_FILE = ROOT / 'exporter-runtime.json'
RUNTIME = json.loads(RUNTIME_FILE.read_text(encoding='utf-8')) if RUNTIME_FILE.is_file() else {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['v5','v8','26'],default='26')
    p.add_argument('--model', choices=['n','s'], default='s')
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--formats', nargs='+', choices=['fp16','int8','fp32'], default=['fp16','int8','fp32'])
    source=p.add_mutually_exclusive_group()
    source.add_argument('--weights', type=Path, help='Local YOLOv8 / YOLO26 detection .pt')
    source.add_argument('--onnx',type=Path,help='NCHW decoded ONNX; dynamic input specialized and checked; pixel cxcywh, no NMS')
    p.add_argument('--calibration-dir', type=Path, default=Path(RUNTIME.get('calibration_dir',ROOT / '.deps/yolov5/coco128/images/train2017')))
    p.add_argument('--test-image', type=Path)
    p.add_argument('--output', type=Path, required=True, help='New or empty destination directory')
    p.add_argument('--work-dir',type=Path,default=Path(RUNTIME.get('work_dir',Path(tempfile.gettempdir())/'fydnn-export-jobs')),
                   help='ASCII staging directory for Windows TensorFlow SavedModel paths')
    p.add_argument('--export-python', type=Path, default=Path(RUNTIME.get('export_python',ROOT / '.toolchain/yolo26-env/Scripts/python.exe')))
    p.add_argument('--convert-python', type=Path, default=Path(RUNTIME.get('convert_python',ROOT / '.toolchain/model-env/Scripts/python.exe')))
    p.add_argument('--evaluate', action='store_true', help='COCO128-style images/train2017 + labels/train2017 evaluation')
    return p

def main(argv=None):
    a = parser().parse_args(argv)
    if a.size < 32 or a.size % 32:
        raise ValueError('Input size must be a positive multiple of 32')
    a.output = a.output.resolve()
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        raise ValueError('Destination must be empty; existing models are never overwritten')
    if a.family=='v5' and not a.onnx:
        raise ValueError('YOLOv5 uses an upstream v7 decoded ONNX source; select a .onnx file')
    if os.name=='nt' and not str(a.work_dir.resolve()).isascii():
        raise ValueError('Windows TensorFlow requires an ASCII staging path; set --work-dir or exporter-runtime.json work_dir')
    for py in ((a.convert_python,) if a.onnx else (a.export_python,a.convert_python)):
        if not py.is_file(): raise FileNotFoundError(f'Python environment: {py}')
    if a.weights and not a.weights.is_file(): raise FileNotFoundError(a.weights)
    if a.onnx and not a.onnx.is_file(): raise FileNotFoundError(a.onnx)
    images = sorted(p for p in a.calibration_dir.glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if 'int8' in a.formats and len(images) < 32:
        raise ValueError('INT8 requires at least 32 real calibration/evaluation images')
    image = a.test_image.resolve() if a.test_image else (images[0].resolve() if images else None)
    if image is None or not image.is_file():
        raise ValueError('Select a real validation image or calibration directory')
    dataset = a.calibration_dir.resolve().parent.parent
    if a.evaluate and (not (dataset / 'labels/train2017').is_dir() or a.calibration_dir.name != 'train2017'):
        raise ValueError('Evaluation requires COCO128-style images/train2017 and labels/train2017')
    a.output.mkdir(parents=True, exist_ok=True)
    logs = a.output / 'logs'; logs.mkdir()
    job = a.work_dir.resolve() / uuid.uuid4().hex
    source, converted, corrected = (job / part for part in ('source','converted','corrected'))
    source.mkdir(parents=True)
    model_name=f'yolo{a.family}{a.model}'
    state = {'status':'running', 'started_at_local':datetime.now().isoformat(timespec='seconds'),
             'family':a.family,'model':model_name, 'size':a.size, 'formats':a.formats,
             'source_type':'onnx' if a.onnx else 'pt',
             'work_directory':str(job), 'device_validation':'NOT RUN: host export only', 'stages':[]}
    state_file = a.output / 'export-job.json'
    save_json(state_file,state)
    def run(stage, python, script, args):
        command = [str(python.resolve()), str(ROOT / 'tools' / script), *map(str,args)]
        entry = {'stage':stage,'command':command,'status':'running'}
        state['stages'].append(entry); save_json(state_file,state)
        print(f'\n[STAGE] {stage}',flush=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='3')
        with (logs / (stage + '.log')).open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace') as process:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line,end='',flush=True)
                code = process.wait()
        entry.update(exit_code=code,status='passed' if code == 0 else 'failed')
        save_json(state_file,state)
        if code: raise subprocess.CalledProcessError(code,command)
    try:
        stem=f'{model_name}_{a.size}'
        if a.onnx:
            run('01-validate-onnx',a.convert_python,'import_onnx.py',
                [a.onnx.resolve(),'--family',a.family,'--size',a.size,'--output',source,'--name',stem])
        else:
            args=['--family',a.family,'--model',a.model,'--size',a.size,'--output',source]
            if a.weights: args+=['--weights',a.weights.resolve()]
            run('01-official-onnx',a.export_python,'export_yolo26.py',args)
            run('01b-validate-onnx',a.convert_python,'import_onnx.py',
                [source/(stem+'.onnx'),'--family',a.family,'--size',a.size,'--output',source,'--name',stem])
        source_report=json.loads((source/(stem+'.source.json')).read_text(encoding='utf-8'))
        classes=source_report['classes']
        if a.evaluate and classes!=80:
            raise ValueError('COCO evaluation expects 80 classes; disable evaluation for a custom class model')
        run('02-onnx2tf',a.convert_python,'convert_model.py',[source / (stem+'.onnx'),
            '--family',a.family,'--output',converted,'--size',a.size,'--classes',classes,'--test-image',image,
            '--mediatek-4d','--normalize-coordinates','--name',stem,'--keep-saved-model',
            '--fp32-intermediate-only'])
        saved_model=converted
        conversion=json.loads((converted/'validation.json').read_text(encoding='utf-8'))
        layout=conversion[0]['layout']
        export_name=stem+'_4d_norm'+('_channels' if layout=='CHANNELS' else '')
        if a.family=='26':
            run('03-attention-layout',a.convert_python,'rewrite_attention_merge.py',
                ['--source',converted,'--output',corrected,'--mode','nchw-add'])
            saved_model=corrected;export_name+='_mergefix'
        run('04-tflite-precisions',a.convert_python,'export_precisions.py',
            ['--saved-model',saved_model,'--output',a.output,'--name',export_name,
             '--size',a.size,'--calibration-dir',a.calibration_dir.resolve(),
             '--formats',*a.formats])
        for path in source.iterdir():
            if path.suffix in ('.pt','.onnx','.json'):shutil.copy2(path,a.output/path.name)
        if a.family=='26':shutil.copy2(corrected/'rewrite.json',a.output/'attention-merge-rewrite.json')
        replacements=converted/'onnx2tf-replacements.json'
        if replacements.is_file():shutil.copy2(replacements,a.output/replacements.name)
        shutil.copy2(converted/'validation.json',a.output/'conversion-validation.json')
        run('05-tensor-check',a.convert_python,'inspect_tflite.py',[a.output])
        run('06-final-parity',a.convert_python,'validate_export.py',
            ['--models',a.output,'--onnx',a.output/(stem+'.onnx'),'--image',image,'--size',a.size,
             '--family',a.family,'--classes',classes])
        if a.evaluate:
            run('07-coco-evaluation',a.convert_python,'evaluate_yolo.py',
                ['--models',a.output,'--dataset',dataset,'--onnx',a.output/(stem+'.onnx'),
                 '--size',a.size,'--family',a.family])
            accuracy=json.loads((a.output/'accuracy-report.json').read_text(encoding='utf-8'))
            baseline=accuracy['results'][0]['mAP50_95']
            for report in accuracy['results'][1:]:
                if 'int8' not in report['file'] and report['mAP50_95'] < baseline-.02:
                    raise ValueError(f'Floating point export mAP dropped more than 2 percentage points: {report}')
        models = sorted(a.output.glob('*.tflite'))
        if len(models) != len(set(a.formats)): raise ValueError('Missing requested precision outputs')
        final_parity=json.loads((a.output/'final-parity.json').read_text(encoding='utf-8'))
        save_json(a.output/'SHA256.json',{'model':state['model'],'size':[a.size,a.size],
            'input_shape':[1,a.size,a.size,3],
            'output_shape':final_parity['reports'][0]['output_shape'],
            'family':a.family,'layout':layout,'classes':classes,
            'normalized_coordinates':True,'device_validation':'NOT RUN: NPU compatibility and speed require device test',
            'fp16_note':'FP16 weight storage, FLOAT32 input/output; hardware arithmetic is driver-controlled',
            'files':[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in models]})
        state.update(status='complete',finished_at_local=datetime.now().isoformat(timespec='seconds'))
        save_json(state_file,state)
        print('\nEXPORT COMPLETE: '+str(a.output),flush=True)
    except Exception as error:
        state.update(status='failed',error=str(error)); save_json(state_file,state)
        raise

if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='replace')
        sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    main()
