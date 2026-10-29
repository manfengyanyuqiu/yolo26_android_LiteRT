"""Collect the tested build and exact validated models, with hashes and instructions."""
import hashlib
import json
import re
import sys
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
version = re.search(r'versionName = "([^"]+)"',(root/'app/build.gradle.kts').read_text(encoding='utf-8')).group(1)
camera = version == '0.4.0'
destination = root.parent / (f'天玑9500-YOLOv5n224-摄像头120FPS-v{version}' if camera else '天玑9500-YOLOv5n224')
destination.mkdir(exist_ok=True)
model_dir = root / 'models/yolov5n-224-mediatek'
items = {
    f'FYDNN-{version}-arm64.apk': root/'app/build/outputs/apk/debug/app-debug.apk',
    'yolov5n_224_4d_norm_fp32io_fp16weights.tflite': model_dir/'yolov5n_224_4d_norm_fp32io_fp16weights.tflite',
    'yolov5n_224_4d_norm_int8.tflite': model_dir/'yolov5n_224_4d_norm_int8.tflite',
    '详细说明.md': root/'MIGRATION.md',
    'LICENSE-yolov5-v7.txt': root/'.deps/yolov5/LICENSE-yolov5-v7.txt',
}
if camera:
    items['摄像头使用说明.md'] = root/'CAMERA.md'
    items['CAMERA.md'] = root/'CAMERA.md'
for name in ('accuracy-report.json','validation.json','tensor-report.json',
             'yolov5n_224_4d_norm_int8.quantization.json','yolov5n_224.source.json'):
    items['验证报告/'+name] = model_dir/name
manifest = []
for relative, source in items.items():
    target = destination/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    manifest.append({'file':relative,'bytes':target.stat().st_size,
                     'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
test_results = [ET.parse(p).getroot().attrib for p in (root/'app/build/test-results/testDebugUnitTest').glob('TEST-*.xml')]
test_count = sum(int(t['tests']) for t in test_results)
assert test_count == (21 if camera else 12)
assert all(int(t['failures']) == 0 and int(t['errors']) == 0 for t in test_results)
accuracy = json.loads((model_dir/'accuracy-report.json').read_text(encoding='utf-8'))
for item in manifest:
    if item['file'].endswith('.tflite'):
        validated = next(r for r in accuracy['results'] if r['file']==item['file'])
        assert validated['sha256']==item['sha256']
(destination/'SHA256.json').write_text(json.dumps({'app_version':version,'host_tests':test_count,
    'device_test':'Not performed for these new models: phone unavailable',
    'files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
(destination/'先看使用说明.txt').write_text('''FYDNN 0.3.0 · 天玑9500 · YOLOv5n 224×224

1. 安装 FYDNN-0.3.0-arm64.apk，可覆盖旧版。
2. 两个 .tflite 放到手机 Download；在应用中分别选择、编译和测试。
3. 配置：YOLOv5 / 输出 ROWS [1,N,C] / 80 类 / 勾选归一化坐标。
   保留文件名，导入带 _norm_ 的模型会自动勾选归一化坐标。
4. 先选测试图片，再点击检测或测速。换模型后使用同一图片比较。

FP16 权重版：约3.83MB，FLOAT32输入和输出。
INT8 版：约2.10MB，权重和激活全INT8，INT8输入和输出。
两版使用官方COCO80类权重，不是原.fymf的自定义类别模型。

已复用输入/截屏缓冲、量化查表、INT8直接解码及objectness筛选。
仅使用联发科NPU，失败显示实际错误，不自动切换CPU。
测速分别显示推理+同步读回、图片端到端耗时；不含录屏采集和显示。

96张COCO128图片用于校准，另32张用于CPU参考评估：
FP16权重版 mAP50–95 29.11%；INT8版 26.22%，相差2.90个百分点。
这不是正式COCO验证集指标，也不是用户实际检测场景精度。

本次手机不在身边，两个新模型尚未做真机NPU编译和测速。
FP16权重文件不代表已经验证NPU内部以FP16执行。
12项主机测试和APK构建通过。更多信息见“详细说明.md”和“验证报告”。

完整源码工程：C:\\Users\\Alienware\\Desktop\\fydnn\\android-dimensity9500
设备自动对比脚本：上述工程下 tools/benchmark_224.ps1
YOLOv5来源：https://github.com/ultralytics/yolov5/releases/tag/v7.0
YOLOv5 v7.0许可证见 LICENSE-yolov5-v7.txt。
''',encoding='utf-8-sig')
if camera:
    (destination/'先看使用说明.txt').write_text(f'''FYDNN {version} · YOLOv5n 224 · 摄像头实时检测

1. 安装 FYDNN-{version}-arm64.apk，可覆盖旧版。
2. 将 .tflite 放到手机 Download，在应用中选择模型。
3. 配置 YOLOv5 / ROWS / 80类 / 勾选归一化坐标。
4. 点击“摄像头检测：120 FPS 采集 + 实时画框”，授予相机权限。
   此入口自动编译模型。退出或切到后台会释放相机和NPU。

后置相机使用公开的固定120FPS模式；如未公开或配置失败，显示实际错误，
不会静默改成30/60FPS。相机原始尺寸按设备能力选择，模型输入仍为224×224。
推理忙时跳过旧帧，处理最新帧；画框叠加在实时预览上。
界面分别显示相机回报FPS、GPU取帧FPS、实际检测FPS和处理耗时。
120FPS采集请求不等于已经达到每秒120次检测。

FP16权重模型约3.83MB，INT8模型约2.10MB，均为官方COCO80类权重。
不是原.fymf的自定义类别权重。FP16权重存储不证明NPU内部执行精度。
两份模型与0.3.0一致，不用重新量化。

21项主机测试、构建通过。手机未连接，相机120FPS和实景检测尚待真机验证。
详细流程、计时边界和限制见“摄像头使用说明.md”。
模型精度报告见“验证报告”。源码位于：
C:/Users/Alienware/Desktop/fydnn/android-dimensity9500
YOLOv5 v7.0许可证见 LICENSE-yolov5-v7.txt。
''',encoding='utf-8-sig')
archive = destination.parent / (destination.name+'.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
    for file in sorted(destination.rglob('*')):
        if file.is_file(): bundle.write(file,file.relative_to(destination.parent))
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
print(str(destination))
print(str(archive))
print(json.dumps(manifest[:3],ensure_ascii=False,indent=2))
