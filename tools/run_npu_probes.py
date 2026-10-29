"""Run explicit diagnostic models with bounded device calls and save results."""
import argparse
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent.parent
adb = str(root / '.toolchain/sdk/platform-tools/adb.exe')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('manifest')
p.add_argument('--timeout', type=int, default=35)
p.add_argument('--names', nargs='*')
a = p.parse_args()
directory = a.directory.resolve()
specs = json.loads((directory / a.manifest).read_text(encoding='utf-8'))
records = []
def run(*args, timeout=15):
    return subprocess.run([adb, *args], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout, check=True).stdout
for spec in specs:
    name = spec['name']
    if a.names and name not in a.names: continue
    model = directory / (name + '.tflite')
    if not model.is_file(): raise FileNotFoundError(model)
    run('push', str(model), '/data/local/tmp/' + model.name)
    run('shell', 'run-as', 'com.fydnn.dimensity', 'cp', '/data/local/tmp/' + model.name, 'files/' + model.name)
    start = time.monotonic()
    try:
        output = run('shell','am','instrument','-w','-r','-e','model',model.name,
                     '-e','family','YOLO26','-e','layout','CHANNELS',
                     '-e','classes',str(spec.get('classes',80)),'-e','normalized','true',
                     '-e','benchmark','false',
                     'com.fydnn.dimensity.test/com.fydnn.dimensity.NpuDeviceInstrumentation',timeout=a.timeout)
    except subprocess.TimeoutExpired:
        run('shell','am','force-stop','com.fydnn.dimensity')
        output = f'TIMEOUT after {a.timeout}s; target process stopped'
    (directory / (name + '-device.txt')).write_text(output, encoding='utf-8')
    status = 'PASS' if 'NPU compile/run PASS' in output else 'FAIL'
    record = {'name':name, 'result':status, 'elapsed_seconds':round(time.monotonic()-start,2), 'output':output}
    records.append(record)
    print(json.dumps(record), flush=True)
(directory / 'device-results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
