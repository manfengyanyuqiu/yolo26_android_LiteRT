"""Package the verified offline YOLO26 320 artifacts and their source."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parent.parent
folder = root / 'models/yolo26n-320-neuronfix-v061'
build = json.loads((root / 'app/build/outputs/apk/debug/output-metadata.json').read_text(encoding='utf-8'))['elements'][0]
assert (build['versionName'], build['versionCode']) == ('0.6.1', 9)
apk = root / 'app/build/outputs/apk/debug' / build['outputFile']
test_apk = root / 'app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk'
build_log = root / '.deps/yolo26-061-final-build.log'
signature_log = root / '.deps/yolo26-061-final-signature.log'
assert 'BUILD SUCCESSFUL' in build_log.read_text(encoding='utf-8')
assert 'Verified using v2 scheme (APK Signature Scheme v2): true' in signature_log.read_text(encoding='utf-8')
tests = list((root / 'app/build/test-results/testDebugUnitTest').glob('TEST-*.xml'))
results = [ET.parse(p).getroot().attrib for p in tests]
assert sum(int(t['tests']) for t in results) == 37
assert all(int(t['failures']) == int(t['errors']) == int(t.get('skipped', 0)) == 0 for t in results)
lint = root / 'app/build/reports/lint-results-debug.xml'
assert not any(i.attrib['severity'] in ('Fatal', 'Error') for i in ET.parse(lint).getroot().findall('issue'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

quant = json.loads((folder / 'yolo26n_320_4d_norm_channels_mergefix_int8.quantization.json').read_text(encoding='utf-8'))
assert quant['float_tensor_count'] == quant['per_channel_quantized_tensor_count'] == 0
assert quant['weight_quantization_granularity'] == 'per_tensor'
assert len(quant['calibration_images']) == 96 and len(quant['evaluation_images']) == 32
assert not {p['file'] for p in quant['calibration_images']} & set(quant['evaluation_images'])
assert quant['input'][0]['shape'] == [1, 320, 320, 3]
assert quant['output'][0]['shape'] == [1, 1, 84, 2100]
assert sha(folder / quant['file']) == quant['sha256']
accuracy = json.loads((folder / 'accuracy-report.json').read_text(encoding='utf-8'))
assert accuracy['size'] == [320, 320] and accuracy['family'] == '26'
for report in accuracy['results']:
    assert sha(folder / report['file']) == report['sha256']
baseline = accuracy['results'][0]['mAP50_95']
fp16 = next(r for r in accuracy['results'] if 'fp16weights' in r['file'])
fp32 = next(r for r in accuracy['results'] if r['file'].endswith('_fp32.tflite'))
int8 = next(r for r in accuracy['results'] if r['file'] == quant['file'])
# Host accuracy is a separate check from device compatibility.
assert fp16['mAP50_95'] >= baseline - .01
assert int8['mAP50_95'] >= baseline - .08
assert all(r['max_tensor_rank'] <= 4 for r in json.loads((folder / 'tensor-report.json').read_text(encoding='utf-8')))
rewrite = json.loads((folder / 'attention-merge-rewrite.json').read_text(encoding='utf-8'))
assert rewrite['mode'] == 'nchw-add' and len(rewrite['merges']) == 2
assert rewrite['max_host_error'] < 2e-6
device_folder = root / '.deps/yolo26-neuronfix-device'
device = json.loads((device_folder / 'device-validation.json').read_text(encoding='utf-8'))
device_by_file = {r['model']: r for r in device}
for model in (fp32, fp16):
    report = device_by_file[model['file']]
    assert report['sha256'] == sha(folder / model['file'])
    assert report['npu_execution'] == 'PASS'
    # Bounds apply only to the recorded image / captured device input.
    assert report['coordinate_max_error_pixels'] < 4 and report['score_max_error'] < .02
    assert all(t > 0 for key in ('backend_avg_p50_p95_ms', 'image_end_to_end_avg_p50_p95_ms') for t in report[key])
    evidence = device_folder / Path(model['file']).stem
    for name, size in [('input.bin', 320 * 320 * 3 * 4), ('host-output.bin', 84 * 2100 * 4), ('npu-output.bin', 84 * 2100 * 4)]:
        assert (evidence / name).stat().st_size == size
int8_device = device_by_file[quant['file']]
assert int8_device['sha256'] == quant['sha256']
assert int8_device['npu_execution'].startswith('INCOMPLETE')
assert int8_device['backend_avg_p50_p95_ms'] is None
assert int8_device['image_end_to_end_avg_p50_p95_ms'] is None

destination = root.parent / '天玑9500-YOLO26-320-v0.6.1'
destination.mkdir(exist_ok=True)
files = {
    'FYDNN-0.6.1-arm64.apk': apk,
    'FYDNN-0.6.1-device-tests.apk': test_apk,
    '使用说明.md': root / 'YOLO26.md',
    fp16['file']: folder / fp16['file'],
    fp32['file']: folder / fp32['file'],
    '实验模型-待真机验证/' + quant['file']: folder / quant['file'],
    '原始模型/yolo26n.pt': folder / 'yolo26n.pt',
    '原始模型/yolo26n_320.onnx': folder / 'yolo26n_320.onnx',
    'LICENSE-YOLO26-AGPL-3.0.txt': root / '.toolchain/yolo26-env/Lib/site-packages/ultralytics-8.4.190.dist-info/licenses/LICENSE',
    '模型报告/accuracy-report.json': folder / 'accuracy-report.json',
    '模型报告/attention-merge-rewrite.json': folder / 'attention-merge-rewrite.json',
    '模型报告/tensor-report.json': folder / 'tensor-report.json',
    '模型报告/quantization.json': folder / 'yolo26n_320_4d_norm_channels_mergefix_int8.quantization.json',
    '模型报告/yolo26n_320.source.json': folder / 'yolo26n_320.source.json',
    '模型报告/onnx2tf-replacements.json': folder / 'onnx2tf-replacements.json',
    '验证日志/device-validation.md': root / '.deps/yolo26-device-validation.md',
    '验证日志/build.log': build_log,
    '验证日志/evaluation.log': root / '.deps/yolo26-neuronfix-evaluate.log',
    '验证日志/export.log': root / '.deps/yolo26-neuronfix-export.log',
    '验证日志/apk-signature.log': signature_log,
    '验证日志/lint-results.xml': lint,
    '源码/requirements-export.txt': root / '.deps/yolo26-export-requirements.txt',
    '源码/requirements-convert.txt': root / '.deps/yolo26-conversion-requirements.txt',
}
for name in ['build.gradle.kts', 'settings.gradle.kts', 'gradle.properties', 'app/build.gradle.kts', 'YOLO26.md', 'CAMERA.md', 'CALL_OPTIMIZATION.md']:
    files['源码/' + name] = root / name
for path in (root / 'app/src').rglob('*'):
    if path.is_file():
        matching_apk = test_apk if path.relative_to(root / 'app/src').parts[0] == 'androidTest' else apk
        assert path.stat().st_mtime <= matching_apk.stat().st_mtime, f'APK older than source: {path}'
        files['源码/' + path.relative_to(root).as_posix()] = path
for path in (root / 'tools').glob('*.py'):
    files['源码/tools/' + path.name] = path
for path in device_folder.rglob('*'):
    if path.is_file():
        files['验证报告/' + path.relative_to(device_folder).as_posix()] = path
for path in tests:
    files['主机测试/' + path.name] = path

unexpected = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} - set(files) - {'SHA256.json'}
assert not unexpected, f'Unexpected files in destination; review before packaging: {unexpected}'
manifest = []
for relative, source in files.items():
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append({'file': relative, 'bytes': target.stat().st_size, 'sha256': sha(target)})
(destination / 'SHA256.json').write_text(json.dumps({
    'version': '0.6.1', 'version_code': 9, 'apk_type': 'debug-signed arm64',
    'model': 'Official YOLO26n COCO, dense one-to-many, fixed 320x320',
    'host_tests_passed': 37, 'lint_report_errors': 0,
    'lint_analyzer_note': 'LiteRT Kotlin metadata 2.4 vs analyzer 2.1 diagnostic; build succeeded',
    'device': 'vivo V2548A / MT6993, Neuron 9.0.11; LiteRT 2.3.0 MediaTek NPU-only',
    'validated_on': '2026-10-28',
    'device_validation': device,
    'validation_scope': 'FP32/FP16: one real image, exact captured input host/NPU parity, 5 warmups and 50 measurements. Uncontrolled power/temperature/clocks; static-image test, not camera FPS. INT8: incomplete after ADB disconnect; stored under experimental directory.',
    'fp16_note': 'FP16 weight storage with FLOAT32 I/O; Neuron controls hardware arithmetic precision.',
    'host_accuracy_note': accuracy['dataset'],
    'int8_host_mAP50_95_drop_percentage_points': (baseline - int8['mAP50_95']) * 100,
    'files': manifest,
}, ensure_ascii=False, indent=2), encoding='utf-8')
archive = destination.parent / (destination.name + '.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for relative in list(files) + ['SHA256.json']:
        bundle.write(destination / relative, str(Path(destination.name) / relative))
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert len(bundle.namelist()) == len(files) + 1
    for entry in manifest:
        content = bundle.read(destination.name + '/' + entry['file'])
        assert hashlib.sha256(content).hexdigest() == entry['sha256']
print(json.dumps({'folder': str(destination), 'archive': str(archive),
                  'archive_bytes': archive.stat().st_size, 'files': len(files) + 1}, ensure_ascii=True))
