"""Package exporter 0.6.3 and verified YOLOv5/v8 examples without runtimes."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent.parent
TOOL=ROOT.parent/'模型导出工具-v0.6.3'
MODELS=ROOT.parent/'YOLOv5-YOLOv8-320-v0.6.3'
CASES={
    'yolov5n-224':ROOT/'models/exporter-v063-v5n-224-r2',
    'yolov5s-320':ROOT/'models/exporter-v063-v5s-320',
    'yolov8n-320':ROOT/'models/exporter-v063-v8n-320',
    'yolov8s-320':ROOT/'models/exporter-v063-v8s-320',
    'yolo26s-320-onnx-regression':ROOT/'models/exporter-v063-26s-320',
    'yolo26s-320-pt-regression':ROOT/'models/exporter-v063-26s-pt320',
    'packaged-tool-chinese-path-regression':ROOT/'.deps/交付工具中文路径验证',
}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

verification={'version':'0.6.3','date':'2026-10-28',
    'device_validation':'NOT RUN; new models require phone NPU compilation and latency tests',
    'accuracy_scope':'COCO128: 96 calibration / 32 held-out evaluation images; not official COCO val2017',
    'cases':[]}
for name,directory in CASES.items():
    job=read(directory/'export-job.json')
    assert job['status']=='complete' and all(s['status']=='passed' for s in job['stages']),name
    manifest=read(directory/'SHA256.json')
    parity=read(directory/'final-parity.json')['reports']
    tensors=read(directory/'tensor-report.json')
    assert len(manifest['files'])==len(set(job['formats']))
    assert all(t['max_tensor_rank']<=4 for t in tensors)
    for entry in manifest['files']:
        path=directory/entry['file']
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
        report=next(p for p in parity if p['file']==path.name)
        assert report['sha256']==entry['sha256']
        assert report['status'].startswith('HOST_EXECUTION_PASS' if 'int8' in path.name else 'HOST_PARITY_PASS')
        if 'int8' in path.name:
            q=read(path.with_suffix('.quantization.json'))
            assert q['sha256']==entry['sha256'] and q['float_tensor_count']==q['per_channel_quantized_tensor_count']==0
            assert not {p['file'] for p in q['calibration_images']} & set(q['evaluation_images'])
    case={'name':name,'family':job['family'],'source_type':job['source_type'],
          'size':manifest['size'],'layout':manifest['layout'],'models':manifest['files'],
          'stages_passed':[s['stage'] for s in job['stages']]}
    source_report=read(directory/(job['model']+'_'+str(job['size'])+'.source.json'))
    if job['source_type']=='pt':
        upstream=source_report['upstream_export_report']
        assert upstream['pytorch_onnx_max_coordinate_error_pixels']<=.05
        assert upstream['pytorch_onnx_max_probability_error']<=.001
        case['pytorch_onnx_coordinate_error_pixels']=upstream['pytorch_onnx_max_coordinate_error_pixels']
        case['pytorch_onnx_probability_error']=upstream['pytorch_onnx_max_probability_error']
    case['onnx_specialization']=source_report.get('specialization')
    accuracy=directory/'accuracy-report.json'
    if accuracy.is_file():
        result=read(accuracy)
        baseline=result['results'][0]['mAP50_95']
        case['accuracy']=[{'file':r['file'],'mAP50_95':r['mAP50_95'],
             'drop_percentage_points':(baseline-r['mAP50_95'])*100} for r in result['results']]
        for r in result['results']:
            assert sha(directory/r['file'])==r['sha256']
    verification['cases'].append(case)
    if name in ('yolov5s-320','yolov8n-320','yolov8s-320'):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in ('.tflite','.json'):
                copy(path,MODELS/name/path.name)
        for path in (directory/'logs').glob('*.log'):copy(path,MODELS/name/'logs'/path.name)
        source=next(p for p in directory.glob('*.onnx') if '.original.' not in p.name)
        copy(source,MODELS/name/'source'/source.name)

scripts=['model_exporter.py','model_exporter_gui.py','export_yolo26.py','convert_model.py',
         'import_onnx.py','export_precisions.py','rewrite_attention_merge.py',
         'quantize_yolo.py','inspect_tflite.py','validate_export.py','evaluate_yolo.py']
for name in scripts:copy(ROOT/'tools'/name,TOOL/'tools'/name)
for name in ('launch_exporter.cmd','launch_exporter.ps1'):copy(ROOT/name,TOOL/name)
copy(ROOT/'MODEL_EXPORTER.md',TOOL/'使用说明.md')
for kind,source in [('export','yolo26-export-requirements.txt'),('convert','yolo26-conversion-requirements.txt')]:
    copy(ROOT/'.deps'/source,TOOL/f'requirements-{kind}.txt')
copy(ROOT/'.toolchain/yolo26-env/Lib/site-packages/ultralytics-8.4.190.dist-info/licenses/LICENSE',TOOL/'LICENSE-Ultralytics-AGPL-3.0.txt')
copy(ROOT/'.deps/yolov5/LICENSE-yolov5-v7.txt',TOOL/'LICENSE-YOLOv5-GPL-3.0.txt')
write(TOOL/'exporter-runtime.json',{
    'export_python':str(ROOT/'.toolchain/yolo26-env/Scripts/python.exe'),
    'convert_python':str(ROOT/'.toolchain/model-env/Scripts/python.exe'),
    'calibration_dir':str(ROOT/'.deps/yolov5/coco128/images/train2017'),
    'work_dir':str(ROOT/'.deps/export-jobs'),
    'note':'Local existing environments; update paths after moving the original project. Runtime not bundled.'})
write(TOOL/'verification.json',verification)
write(MODELS/'verification.json',verification)
copy(ROOT/'MODEL_EXPORTER.md',MODELS/'使用说明.md')
for name in ('LICENSE-Ultralytics-AGPL-3.0.txt','LICENSE-YOLOv5-GPL-3.0.txt'):
    copy(TOOL/name,MODELS/name)
for folder in (TOOL,MODELS):
    entries=[{'file':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SHA256-package.json' and '__pycache__' not in p.parts]
    write(folder/'SHA256-package.json',{'version':'0.6.3','files':entries})
archive=TOOL.parent/(TOOL.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(TOOL.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:z.write(path,TOOL.name+'/'+path.relative_to(TOOL).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'tool':str(TOOL),'archive':str(archive),'models':str(MODELS),
                  'validated_cases':len(verification['cases'])},ensure_ascii=False,indent=2))
