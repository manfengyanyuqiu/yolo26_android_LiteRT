"""Compare ONNX / TFLite on images excluded from calibration, using COCOeval."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from quantize_yolo import dataset_split, letterbox


def prediction_rows(output, channels):
    if output.ndim == 4 and output.shape[0] == 1:
        output = np.squeeze(output, axis=1 if output.shape[1] == 1 else 3)
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f'Expected fixed decoded output, got {output.shape}')
    if output.shape[2] == channels: return output[0]
    if output.shape[1] == channels: return output[0].T
    raise ValueError(f'Output channels do not match family: {output.shape}, {channels}')


def decoded_boxes(output, geometry, size, normalized, family='v5'):
    prediction = prediction_rows(output, 85 if family == 'v5' else 84)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite output')
    scores = prediction[:,4,None] * prediction[:,5:] if family == 'v5' else prediction[:,4:]
    classes = scores.argmax(axis=1)
    confidence = scores[np.arange(len(scores)),classes]
    keep = confidence >= .001
    xywh = prediction[keep,:4].copy()
    scores, classes = confidence[keep], classes[keep]
    if normalized: xywh *= size
    xyxy = np.concatenate([xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2],axis=1)
    w,h,rw,rh,px,py = geometry
    xyxy[:,[0,2]] = np.clip((xyxy[:,[0,2]]-px)*w/rw,0,w)
    xyxy[:,[1,3]] = np.clip((xyxy[:,[1,3]]-py)*h/rh,0,h)
    valid = (xyxy[:,2] > xyxy[:,0]) & (xyxy[:,3] > xyxy[:,1])
    xyxy, scores, classes = xyxy[valid], scores[valid], classes[valid]
    # Standard class-offset NMS, TensorFlow's implementation (no custom NMS).
    boxes_for_nms = xyxy[:,[1,0,3,2]] + classes[:,None]*7680
    indices = tf.image.non_max_suppression(boxes_for_nms,scores,300,iou_threshold=.45).numpy()
    return [(int(classes[i]), float(scores[i]), [float(xyxy[i,0]),float(xyxy[i,1]),
             float(xyxy[i,2]-xyxy[i,0]),float(xyxy[i,3]-xyxy[i,1])]) for i in indices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--onnx', type=Path)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--family', choices=['v5', 'v8', '26'], default='v5')
    p.add_argument('--normalized-tflite', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    onnx_file = a.onnx or (a.models/'yolov5n_224.onnx')
    _, paths = dataset_split(a.dataset/'images/train2017')
    samples = [(path,*letterbox(path,a.size)) for path in paths]
    annotations, images = [], []
    for image_id,(path,_,g) in enumerate(samples):
        w,h = g[:2]
        images.append({'id':image_id,'width':w,'height':h,'file_name':path.name})
        label = a.dataset/'labels/train2017'/path.with_suffix('.txt').name
        for line in label.read_text().splitlines() if label.exists() else []:
            c,x,y,bw,bh = map(float,line.split())
            annotations.append({'id':len(annotations)+1,'image_id':image_id,'category_id':int(c),
                'bbox':[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h],'area':bw*w*bh*h,'iscrowd':0})
    truth = COCO()
    truth.dataset = {'images':images,'annotations':annotations,'categories':[{'id':i,'name':str(i)} for i in range(80)],'info':{}}
    truth.createIndex()
    session = ort.InferenceSession(str(onnx_file),providers=['CPUExecutionProvider'])
    reports, reference_outputs = [], []
    files = [onnx_file, *sorted(a.models.glob('*.tflite'))]
    for file in files:
        is_onnx = file.suffix == '.onnx'
        detections, coord_error, prob_error, mean_coord = [], 0., 0., []
        if not is_onnx:
            interpreter = tf.lite.Interpreter(model_content=file.read_bytes(),num_threads=1)
            interpreter.allocate_tensors()
            ins, outs = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        for image_id,(path,x,g) in enumerate(samples):
            if is_onnx:
                input_type = np.float16 if session.get_inputs()[0].type == 'tensor(float16)' else np.float32
                y = session.run(None,{session.get_inputs()[0].name:x.transpose(0,3,1,2).astype(input_type)})[0].astype(np.float32)
                reference_outputs.append(prediction_rows(y, 85 if a.family == 'v5' else 84))
            else:
                inp = x
                if ins['dtype'] in (np.int8, np.uint8):
                    scale,zero = ins['quantization']
                    limits = np.iinfo(ins['dtype'])
                    inp = np.clip(np.floor(x/scale+zero+.5),limits.min,limits.max).astype(ins['dtype'])
                interpreter.set_tensor(ins['index'],inp)
                interpreter.invoke()
                y = interpreter.get_tensor(outs['index']).astype(np.float32)
                if outs['dtype'] in (np.int8, np.uint8):
                    scale,zero = outs['quantization']; y = (y-zero)*scale
                ref = reference_outputs[image_id]
                rows = prediction_rows(y, 85 if a.family == 'v5' else 84)
                delta = np.abs(rows[:,:4]*(a.size if a.normalized_tflite else 1)-ref[:,:4])
                coord_error = max(coord_error,float(delta.max()))
                mean_coord.append(float(delta.mean()))
                prob_error = max(prob_error,float(np.abs(rows[:,4:]-ref[:,4:]).max()))
            for c,score,box in decoded_boxes(y,g,a.size,a.normalized_tflite and not is_onnx,a.family):
                detections.append({'image_id':image_id,'category_id':c,'score':score,'bbox':box})
        if not detections: raise ValueError(f'{file.name}: no detections')
        evaluator = COCOeval(truth,truth.loadRes(detections),'bbox')
        evaluator.params.imgIds = list(range(len(samples)))
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        report = {'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'bytes':file.stat().st_size,'mAP50_95':float(evaluator.stats[0]),'AP50':float(evaluator.stats[1]),
            'AP75':float(evaluator.stats[2]),'detections':len(detections),
            'max_coordinate_error_pixels':coord_error,'mean_coordinate_error_pixels':float(np.mean(mean_coord)) if mean_coord else 0.,
            'max_probability_error_including_objectness':prob_error}
        reports.append(report)
        print(json.dumps(report),flush=True)
    result = {'dataset':'COCO128; 32 images disjoint from 96 PTQ calibration images; not official COCO val2017',
        'images':[p.name for p in paths],'size':[a.size,a.size], 'family': a.family,
        'normalized_tflite_coordinates':a.normalized_tflite,'score_threshold':.001,'nms_iou':.45,
        'execution':'Host CPU reference only; device execution and latency are reported separately',
        'results':reports}
    (a.models/'accuracy-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__': main()
