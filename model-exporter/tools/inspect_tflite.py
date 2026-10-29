"""Record published model tensor ranks, operator inventory and buffer sizes."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
reports=[]
for file in sorted(a.directory.glob('*.tflite')):
    interpreter=tf.lite.Interpreter(model_content=file.read_bytes(),num_threads=1)
    details=interpreter.get_tensor_details()
    rank=max(len(t['shape']) for t in details)
    if rank>4: raise ValueError(f'Rank {rank} exceeds current Neuron dispatch support: {file}')
    reports.append({'file':file.name,'max_tensor_rank':rank,
        'ops':dict(Counter(x['op_name'] for x in interpreter._get_ops_details())),
        'input_bytes':sum(int(np.prod(t['shape']))*np.dtype(t['dtype']).itemsize for t in interpreter.get_input_details()),
        'output_bytes':sum(int(np.prod(t['shape']))*np.dtype(t['dtype']).itemsize for t in interpreter.get_output_details()),
        'verification':'Host flatbuffer inspection; does not prove NPU compilation or performance'})
print(json.dumps(reports,indent=2))
(a.directory/'tensor-report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
