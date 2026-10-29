"""Package exactly the INT8 models exercised on the device, with evidence."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent.parent
destination = root.parent/'天玑9500-INT8修复-v0.4.1'
destination.mkdir(exist_ok=True)
comparison = json.loads((root/'.deps/npu-debug/int8-verified/device-output-comparison.json').read_text())
files = {
    'FYDNN-0.4.1-arm64.apk': root/'app/build/outputs/apk/debug/app-debug.apk',
    '修复说明.md': root/'INT8_FIX.md',
    'LICENSE-yolov5-v7.txt': root/'.deps/yolov5/LICENSE-yolov5-v7.txt',
    '验证报告/device-output-comparison.json': root/'.deps/npu-debug/int8-verified/device-output-comparison.json',
}
for tag, directory, stem in [
    ('n224', 'yolov5n-224-mediatek', 'yolov5n_224'),
    ('s320', 'yolov5s-320-mediatek', 'yolov5s_320'),
]:
    model_dir = root/'models'/directory
    model_name = stem+'_4d_norm_int8.tflite'
    quantization = json.loads((model_dir/(stem+'_4d_norm_int8.quantization.json')).read_text())
    digest = hashlib.sha256((model_dir/model_name).read_bytes()).hexdigest()
    assert quantization['sha256'] == digest
    assert quantization['float_tensor_count'] == 0
    assert quantization['per_channel_quantized_tensor_count'] == 0
    assert next(r for r in comparison if r['model'] == model_name)['sha256'] == digest
    accuracy = json.loads((model_dir/'accuracy-report.json').read_text())
    assert next(r for r in accuracy['results'] if r['file'] == model_name)['sha256'] == digest
    device = root/'.deps/npu-debug/int8-verified'/tag/'device-test.txt'
    device_report = device.read_text(encoding='utf-8')
    assert 'PASS' in device_report and 'INT8' in device_report
    files[stem+'_4d_norm_int8_per_tensor.tflite'] = model_dir/model_name
    for name in ('accuracy-report.json', 'tensor-report.json', stem+'_4d_norm_int8.quantization.json'):
        files['验证报告/'+tag+'/'+name] = model_dir/name
    files['验证报告/'+tag+'/device-test.txt'] = device
tests = [ET.parse(f).getroot().attrib for f in (root/'app/build/test-results/testDebugUnitTest').glob('TEST-*.xml')]
assert sum(int(t['tests']) for t in tests) == 21
assert all(int(t['failures']) == 0 and int(t['errors']) == 0 for t in tests)
manifest = []
for relative, source in files.items():
    target = destination/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append({'file': relative, 'bytes': target.stat().st_size,
                     'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(destination/'SHA256.json').write_text(json.dumps({'version':'0.4.1',
    'device':'V2548A / MT6993 / Android 17 / Neuron 9.0.11',
    'device_npu_compile_and_image_inference':'PASS for both included models',
    'host_tests':21, 'files':manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
archive = destination.parent/(destination.name+'.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for file in sorted(destination.rglob('*')):
        if file.is_file(): bundle.write(file, file.relative_to(destination.parent))
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
print(json.dumps({'destination':str(destination),'archive':str(archive),'files':manifest}, ensure_ascii=True, indent=2))
