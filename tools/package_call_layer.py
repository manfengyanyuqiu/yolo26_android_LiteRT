"""Deliver the offline call-layer candidate without claiming device validation."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parent.parent
build = json.loads((root / 'app/build/outputs/apk/debug/output-metadata.json').read_text(encoding='utf-8'))['elements'][0]
assert (build['versionName'], build['versionCode']) == ('0.5.1', 7)
tests = [ET.parse(p).getroot().attrib for p in (root / 'app/build/test-results/testDebugUnitTest').glob('TEST-*.xml')]
assert sum(int(t['tests']) for t in tests) == 31
assert all(int(t['failures']) == 0 and int(t['errors']) == 0 for t in tests)
lint = ET.parse(root / 'app/build/reports/lint-results-debug.xml').getroot()
assert not any(i.attrib['severity'] in ('Fatal', 'Error') for i in lint.findall('issue'))

destination = root.parent / '天玑9500-调用层优化候选-v0.5.1'
destination.mkdir(exist_ok=True)
files = {
    'FYDNN-0.5.1-arm64.apk': root / 'app/build/outputs/apk/debug' / build['outputFile'],
    'FYDNN-0.5.1-device-tests.apk': root / 'app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk',
    '调用层优化说明.md': root / 'CALL_OPTIMIZATION.md',
    'build.log': root / '.deps/call-optimization-051-build.log',
    'LICENSE-yolov5-v7.txt': root / '.deps/yolov5/LICENSE-yolov5-v7.txt',
}
for directory, stem in [('yolov5n-224-mediatek', 'yolov5n_224'), ('yolov5s-320-mediatek', 'yolov5s_320')]:
    folder = root / 'models' / directory
    int8 = folder / (stem + '_4d_norm_int8.tflite')
    quant = json.loads((folder / (stem + '_4d_norm_int8.quantization.json')).read_text(encoding='utf-8'))
    assert hashlib.sha256(int8.read_bytes()).hexdigest() == quant['sha256']
    assert quant['per_channel_quantized_tensor_count'] == quant['float_tensor_count'] == 0
    files[stem + '_4d_norm_int8_per_tensor.tflite'] = int8
    files[stem + '_4d_norm_fp32io_fp16weights.tflite'] = folder / (stem + '_4d_norm_fp32io_fp16weights.tflite')
    files['模型报告/' + stem + '-quantization.json'] = folder / (stem + '_4d_norm_int8.quantization.json')
    files['模型报告/' + stem + '-accuracy.json'] = folder / 'accuracy-report.json'
for report in (root / 'app/build/test-results/testDebugUnitTest').glob('TEST-*.xml'):
    files['主机测试/' + report.name] = report

manifest = []
for relative, source in files.items():
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append({'file': relative, 'bytes': target.stat().st_size,
                     'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(destination / 'SHA256.json').write_text(json.dumps({
    'version': '0.5.1', 'version_code': 7, 'apk_type': 'debug-signed arm64',
    'host_tests_passed': 31, 'lint_errors': 0,
    'device_test': 'NOT RUN - device unavailable; installation did not complete',
    'performance_improvement': 'UNMEASURED',
    'models': 'Unmodified models from the 0.5.0 delivery; prior device results do not validate the new call path',
    'files': manifest,
}, ensure_ascii=False, indent=2), encoding='utf-8')
archive = destination.parent / (destination.name + '.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for relative in list(files) + ['SHA256.json']:
        bundle.write(destination / relative, str(Path(destination.name) / relative))
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert len(bundle.namelist()) == len(files) + 1
print(json.dumps({'folder': str(destination), 'archive': str(archive),
                  'archive_bytes': archive.stat().st_size, 'files': len(files) + 1}, ensure_ascii=True))
