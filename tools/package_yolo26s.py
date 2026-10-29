"""Package FYDNN 0.6.2 / YOLO26s 320 / desktop export tool with hash checks."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent.parent
MODELS=ROOT/'models/yolo26s-320-neuronfix-v062'
DEST=ROOT.parent/'天玑9500-YOLO26s-320-v0.6.2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
metadata=read(ROOT/'app/build/outputs/apk/debug/output-metadata.json')['elements'][0]
assert (metadata['versionName'],metadata['versionCode'])==('0.6.2',10)
apk=ROOT/'app/build/outputs/apk/debug'/metadata['outputFile']
test_apk=ROOT/'app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk'
test_files=sorted((ROOT/'app/build/test-results/testDebugUnitTest').glob('TEST-*.xml'))
tests=[ET.parse(p).getroot().attrib for p in test_files]
assert sum(int(t['tests']) for t in tests)==37
assert all(int(t['failures'])==int(t['errors'])==int(t.get('skipped',0))==0 for t in tests)
lint=ROOT/'app/build/reports/lint-results-debug.xml'
assert not any(x.attrib['severity'] in ('Fatal','Error') for x in ET.parse(lint).getroot().findall('issue'))
assert 'BUILD SUCCESSFUL' in (ROOT/'.deps/yolo26s-062-build.log').read_text(encoding='utf-8')
assert 'Verified using v2 scheme (APK Signature Scheme v2): true' in (ROOT/'.deps/yolo26s-062-apk-signature.log').read_text(encoding='utf-8')
job=read(MODELS/'export-job.json')
assert job['status']=='complete' and len(job['stages'])==7
assert all(s['status']=='passed' and s['exit_code']==0 for s in job['stages'])
model_manifest=read(MODELS/'SHA256.json')
assert model_manifest['input_shape']==[1,320,320,3]
assert model_manifest['output_shape']==[1,1,84,2100]
assert len(model_manifest['files'])==3
for entry in model_manifest['files']:
    p=MODELS/entry['file']
    assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
accuracy=read(MODELS/'accuracy-report.json')
for entry in accuracy['results']:
    assert sha(MODELS/entry['file'])==entry['sha256']
baseline=accuracy['results'][0]['mAP50_95']
fp16=next(e for e in accuracy['results'] if 'fp16weights' in e['file'])
int8=next(e for e in accuracy['results'] if e['file'].endswith('_int8.tflite'))
assert fp16['mAP50_95']>=baseline-.02
quant=read(MODELS/Path(int8['file']).with_suffix('.quantization.json'))
assert quant['sha256']==int8['sha256']
assert quant['float_tensor_count']==quant['per_channel_quantized_tensor_count']==0
assert len(quant['calibration_images'])==96 and len(quant['evaluation_images'])==32
assert not {p['file'] for p in quant['calibration_images']} & set(quant['evaluation_images'])
assert all(t['max_tensor_rank']<=4 for t in read(MODELS/'tensor-report.json'))
rewrite=read(MODELS/'attention-merge-rewrite.json')
assert rewrite['mode']=='nchw-add' and len(rewrite['merges'])==2
assert all(m['channels']==256 and m['nhwc_shape']==[1,10,10,256] for m in rewrite['merges'])
assert rewrite['max_host_error']<.00001
parity=read(MODELS/'final-parity.json')
for row in parity['reports']:
    assert sha(MODELS/row['file'])==row['sha256']
    assert row['status'].startswith('HOST_EXECUTION_PASS' if 'int8' in row['file'] else 'HOST_PARITY_PASS')
files={
    'FYDNN-0.6.2-arm64.apk':apk,
    '测试组件/FYDNN-0.6.2-device-tests.apk':test_apk,
    '使用说明.md':ROOT/'YOLO26S.md',
    'LICENSE-YOLO26-AGPL-3.0.txt':ROOT/'.toolchain/yolo26-env/Lib/site-packages/ultralytics-8.4.190.dist-info/licenses/LICENSE',
    '验证日志/build.log':ROOT/'.deps/yolo26s-062-build.log',
    '验证日志/apk-signature.log':ROOT/'.deps/yolo26s-062-apk-signature.log',
    '验证日志/export-pipeline.log':ROOT/'.deps/yolo26s-062-export-tool.log',
    '验证日志/lint-results.xml':lint,
    '模型导出工具/使用说明.md':ROOT/'MODEL_EXPORTER.md',
    '模型导出工具/launch_exporter.cmd':ROOT/'launch_exporter.cmd',
    '模型导出工具/launch_exporter.ps1':ROOT/'launch_exporter.ps1',
    '模型导出工具/requirements-export.txt':ROOT/'.deps/yolo26-export-requirements.txt',
    '模型导出工具/requirements-convert.txt':ROOT/'.deps/yolo26-conversion-requirements.txt',
}
export_tools=['model_exporter.py','model_exporter_gui.py','export_yolo26.py','convert_model.py',
              'rewrite_attention_merge.py','export_yolo26_mergefix.py','quantize_yolo.py',
              'inspect_tflite.py','validate_export.py','evaluate_yolo.py']
for name in export_tools: files['模型导出工具/tools/'+name]=ROOT/'tools'/name
for p in MODELS.iterdir():
    if p.is_file():
        if p.suffix=='.tflite':
            relative=('实验模型-待真机验证/' if 'int8' in p.name else '')+p.name
        elif p.suffix in ('.pt','.onnx'): relative='原始模型/'+p.name
        else: relative='模型报告/'+p.name
        files[relative]=p
for p in (MODELS/'logs').glob('*.log'): files['验证日志/导出阶段/'+p.name]=p
for p in test_files: files['验证日志/主机测试/'+p.name]=p
for name in ['build.gradle.kts','settings.gradle.kts','gradle.properties','gradlew.bat','app/build.gradle.kts',
             'YOLO26S.md','MODEL_EXPORTER.md','CAMERA.md','CALL_OPTIMIZATION.md',
             'launch_exporter.cmd','launch_exporter.ps1']:
    files['源码/'+name]=ROOT/name
for p in (ROOT/'gradle/wrapper').glob('*'):
    if p.is_file(): files['源码/gradle/wrapper/'+p.name]=p
for p in (ROOT/'app/src').rglob('*'):
    if not p.is_file(): continue
    target_apk=test_apk if p.relative_to(ROOT/'app/src').parts[0]=='androidTest' else apk
    assert p.stat().st_mtime<=target_apk.stat().st_mtime,f'Source newer than APK: {p}'
    files['源码/'+p.relative_to(ROOT).as_posix()]=p
for p in (ROOT/'tools').glob('*.py'): files['源码/tools/'+p.name]=p
runtime={
    'export_python':str(ROOT/'.toolchain/yolo26-env/Scripts/python.exe'),
    'convert_python':str(ROOT/'.toolchain/model-env/Scripts/python.exe'),
    'calibration_dir':str(ROOT/'.deps/yolov5/coco128/images/train2017'),
    'note':'Local existing environments; update paths after moving the original project. Not a standalone bundled runtime.'
}
generated={
    '模型导出工具/exporter-runtime.json':json.dumps(runtime,ensure_ascii=False,indent=2),
    '源码/exporter-runtime.json':json.dumps(runtime,ensure_ascii=False,indent=2),
}
expected=set(files)|set(generated)|{'SHA256.json'}
if DEST.exists():
    unexpected={p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()}-expected
    assert not unexpected,f'Unexpected destination files: {unexpected}'
DEST.mkdir(parents=True,exist_ok=True)
for relative,source in files.items():
    target=DEST/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
for relative,content in generated.items():
    target=DEST/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(content,encoding='utf-8')
manifest=[{'file':relative,'bytes':(DEST/relative).stat().st_size,'sha256':sha(DEST/relative)} for relative in sorted(set(files)|set(generated))]
(DEST/'SHA256.json').write_text(json.dumps({
    'version':'0.6.2','version_code':10,'created_on':'2026-10-28','apk_type':'debug-signed arm64, Android API31+',
    'model':'Official YOLO26s COCO dense / normalized CHANNELS / fixed 320x320',
    'host_tests_passed':37,'lint_report_errors':0,
    'lint_analyzer_note':'LiteRT Kotlin metadata 2.4 vs analyzer 2.1 diagnostic; some dependency analysis may be limited',
    'device_validation':'NOT RUN for YOLO26s; no ADB device present',
    'exporter':'Tkinter desktop GUI + CLI; uses existing Python environments referenced in exporter-runtime.json',
    'int8_host_accuracy_drop_percentage_points':(baseline-int8['mAP50_95'])*100,
    'accuracy_scope':accuracy['dataset'],
    'files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
archive=DEST.parent/(DEST.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for relative in sorted(expected): z.write(DEST/relative,DEST.name+'/'+relative)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(expected)
    for entry in manifest:
        assert hashlib.sha256(z.read(DEST.name+'/'+entry['file'])).hexdigest()==entry['sha256']
print(json.dumps({'folder':str(DEST),'archive':str(archive),'archive_bytes':archive.stat().st_size,
                  'files_verified':len(manifest),'models':model_manifest['files']},ensure_ascii=True,indent=2))
