"""Compare public/prepared LiteRT calls on the same device model and buffers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--serial')
parser.add_argument('--install', action='store_true', help='Install current main and instrumentation APKs')
parser.add_argument('--image', type=Path, default=root / '.deps/yolov5/coco128/images/train2017/000000000009.jpg')
variants = {
    'n224-int8': root / 'models/yolov5n-224-mediatek/yolov5n_224_4d_norm_int8.tflite',
    'n224-fp16': root / 'models/yolov5n-224-mediatek/yolov5n_224_4d_norm_fp32io_fp16weights.tflite',
    's320-int8': root / 'models/yolov5s-320-mediatek/yolov5s_320_4d_norm_int8.tflite',
    's320-fp16': root / 'models/yolov5s-320-mediatek/yolov5s_320_4d_norm_fp32io_fp16weights.tflite',
}
parser.add_argument('--models', choices=list(variants), nargs='+', default=list(variants))
args = parser.parse_args()
adb = [str(root / '.toolchain/sdk/platform-tools/adb.exe')]
if args.serial:
    adb += ['-s', args.serial]


def run(*command, timeout=180):
    result = subprocess.run(adb + list(command), capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr).decode('utf-8', errors='replace'))
    return result.stdout


assert run('get-state', timeout=15).strip() == b'device', 'Connect one USB-debug-enabled device or select --serial'
assert args.image.is_file(), args.image
for tag in args.models:
    assert variants[tag].is_file(), variants[tag]
if args.install:
    print('Install APKs; allow USB installation on the unlocked phone.', flush=True)
    run('install', '-r', str(root / 'app/build/outputs/apk/debug/app-debug.apk'))
    run('install', '-r', '-t', str(root / 'app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk'))
run('push', str(args.image), '/data/local/tmp/fydnn-callbench.jpg')
run('shell', 'run-as', 'com.fydnn.dimensity', 'cp', '/data/local/tmp/fydnn-callbench.jpg', 'files/callbench.jpg')
reports = root / '.deps/call-layer-device'
reports.mkdir(exist_ok=True)
for tag in args.models:
    model = variants[tag]
    print('Benchmark ' + tag, flush=True)
    run('push', str(model), '/data/local/tmp/fydnn-callbench.tflite')
    run('shell', 'run-as', 'com.fydnn.dimensity', 'cp', '/data/local/tmp/fydnn-callbench.tflite', 'files/callbench.tflite')
    result = run('shell', 'am', 'instrument', '-w', '-r', '-e', 'model', 'callbench.tflite',
                 '-e', 'image', 'callbench.jpg', '-e', 'normalized', 'true',
                 '-e', 'callBenchmark', 'true',
                 'com.fydnn.dimensity.test/com.fydnn.dimensity.NpuDeviceInstrumentation')
    (reports / (tag + '-instrumentation.txt')).write_bytes(result)
    report = run('exec-out', 'run-as', 'com.fydnn.dimensity', 'cat', 'files/device-test.txt')
    (reports / (tag + '.txt')).write_bytes(report)
    text = report.decode('utf-8')
    assert 'call_output_exact_match=true' in text and '\nPASS\n' in text.replace('\r\n', '\n'), text
    for line in text.splitlines():
        if line.startswith(('call_', 'public_call_', 'prepared_call_')):
            print(line, flush=True)
    (reports / (tag + '.json')).write_text(json.dumps({
        'model': str(model), 'sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'warmup_pairs': 20, 'timed_pairs': 200, 'order': 'alternating',
        'measurement': 'submit + inference completion + output readback; excludes preprocessing/NMS',
        'exact_output_match': True,
    }, indent=2), encoding='utf-8')
print('Reports: ' + str(reports), flush=True)
