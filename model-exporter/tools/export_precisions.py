"""Export chosen TFLite precisions from a verified fixed SavedModel."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import argparse
import hashlib
import json
from pathlib import Path
import tensorflow as tf
from quantize_yolo import quantize

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--size',type=int,required=True)
    p.add_argument('--formats',nargs='+',choices=['fp32','fp16','int8'],required=True)
    p.add_argument('--calibration-dir',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    for precision in dict.fromkeys(a.formats):
        suffix={'fp32':'fp32','fp16':'fp32io_fp16weights','int8':'int8'}[precision]
        destination=a.output/(a.name+'_'+suffix+'.tflite')
        if precision=='int8':
            quantize(a.saved_model,destination,a.calibration_dir,a.size,disable_per_channel=True)
        else:
            converter=tf.lite.TFLiteConverter.from_saved_model(str(a.saved_model))
            if precision=='fp16':
                converter.optimizations=[tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types=[tf.float16]
            content=converter.convert();destination.write_bytes(content)
            print(json.dumps({'file':destination.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}),flush=True)
