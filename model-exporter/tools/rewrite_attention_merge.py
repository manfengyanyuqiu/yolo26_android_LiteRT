"""Apply a shape-derived NCHW sum at structurally matched YOLO26 Attention merges."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

root = Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',default='models/yolo26n-320-mediatek')
p.add_argument('--tag',default='yolo26-nchw-add-saved')
p.add_argument('--mode',choices=['conv','flat-add','flat-conv','nchw-add'],default='nchw-add')
p.add_argument('--output', type=Path, help='Explicit destination SavedModel directory')
a=p.parse_args()
loaded = tf.saved_model.load(str(root / a.source))
frozen = convert_variables_to_constants_v2(loaded.signatures['serving_default'])
graph = frozen.graph.as_graph_def(add_shapes=True)
def unwrap(tensor):
    while tensor.op.type in ('Identity', 'Reshape', 'Transpose'):
        tensor = tensor.op.inputs[0]
    return tensor

def attention_merge(operation):
    if operation.type not in ('Add', 'AddV2') or len(operation.inputs) != 2:
        return False
    left, right = operation.inputs
    if left.shape.rank != 4 or left.shape != right.shape:
        return False
    def attention(t):
        return unwrap(t).op.type in ('BatchMatMulV2', 'BatchMatMul', 'MatMul')
    def position(t):
        op = t.op
        if op.type in ('Add', 'AddV2', 'BiasAdd'):
            return any(i.op.type in ('Conv2D', 'DepthwiseConv2dNative') for i in op.inputs)
        return op.type in ('Conv2D', 'DepthwiseConv2dNative')
    return (attention(left) and position(right)) or (attention(right) and position(left))

targets = {op.name for op in frozen.graph.get_operations() if attention_merge(op)}
softmax_count = sum(op.type == 'Softmax' for op in frozen.graph.get_operations())
if not targets or len(targets) != softmax_count:
    raise ValueError(f'Unsupported Attention topology: {len(targets)} merges / {softmax_count} Softmax nodes')
replacement, reports = [], []
for node in graph.node:
    if node.name not in targets:
        replacement.append(node); continue
    operation = frozen.graph.get_operation_by_name(node.name)
    left, right = operation.inputs
    if left.shape != right.shape or left.shape.rank != 4:
        raise ValueError('Attention merge shape mismatch')
    batch, height, width, channels = [int(d) for d in left.shape]
    if batch != 1: raise ValueError('Only fixed batch=1 Attention is supported')
    spatial = height * width
    weights = np.concatenate([np.eye(channels), np.eye(channels)], axis=0).astype(np.float32)
    weights = weights.reshape(1, 1, channels * 2, channels)
    subgraph = tf.Graph()
    with subgraph.as_default():
        x = tf.compat.v1.placeholder(tf.float32, left.shape, name='left')
        y = tf.compat.v1.placeholder(tf.float32, right.shape, name='right')
        with tf.name_scope(node.name + '/fixed_sum'):
            if a.mode == 'conv':
                merged = tf.concat([x, y], axis=3)
                kernel = tf.constant(weights)
                result = tf.nn.conv2d(merged, kernel, strides=[1,1,1,1], padding='VALID')
            elif a.mode == 'nchw-add':
                left_nchw=tf.reshape(tf.transpose(x,[0,3,1,2]),[1,channels,spatial])
                right_nchw=tf.reshape(tf.transpose(y,[0,3,1,2]),[1,channels,spatial])
                result=tf.transpose(tf.reshape(left_nchw+right_nchw,[1,channels,height,width]),[0,2,3,1])
            else:
                flat_x=tf.reshape(x,[1,1,channels*spatial,1])
                flat_y=tf.reshape(y,[1,1,channels*spatial,1])
                if a.mode == 'flat-add': result=flat_x+flat_y
                else:
                    merged=tf.concat([flat_x,flat_y],axis=3)
                    result=tf.nn.conv2d(merged,tf.ones([1,1,2,1]),strides=[1,1,1,1],padding='VALID')
        tf.reshape(result,left.shape,name=node.name)
    for added in subgraph.as_graph_def(add_shapes=True).node:
        if added.op == 'Placeholder': continue
        for index, value in enumerate(added.input):
            if value == 'left': added.input[index] = left.name
            elif value == 'right': added.input[index] = right.name
        replacement.append(added)
    reports.append({'node':node.name,'channels':channels,'nhwc_shape':left.shape.as_list()})
if len(reports) != len(targets): raise ValueError(reports)
del graph.node[:]
graph.node.extend(replacement)
wrapped = tf.compat.v1.wrap_function(lambda: tf.import_graph_def(graph, name=''), [])
input_tensor = wrapped.graph.get_tensor_by_name(frozen.inputs[0].name)
output_tensor = wrapped.graph.get_tensor_by_name(frozen.outputs[0].name)
pruned = wrapped.prune(input_tensor, output_tensor)
input_shape = frozen.inputs[0].shape.as_list()
if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[-1] != 3 or any(d is None for d in input_shape):
    raise ValueError(f'Expected fixed NHWC input: {input_shape}')
class Model(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(input_shape,tf.float32,name='images')])
    def serve(self, images):
        return {'output0':pruned(images)}
model=Model()
sample=np.random.default_rng(26).random(input_shape,dtype=np.float32)
expected=frozen(tf.constant(sample))[0].numpy()
actual=model.serve(tf.constant(sample))['output0'].numpy()
error=float(np.max(np.abs(expected-actual)))
if not np.isfinite(actual).all() or error>.001: raise ValueError(error)
destination=a.output.resolve() if a.output else root / '.deps' / a.tag
tf.saved_model.save(model,str(destination),signatures={'serving_default':model.serve})
(destination/'rewrite.json').write_text(json.dumps({'source_saved_model':a.source,'mode':a.mode,
    'merges':reports,'max_host_error':error,'device_validation':'See separate hash-bound device report'},indent=2),encoding='utf-8')
print(json.dumps({'max_host_error':error,'merges':len(reports),'input_shape':input_shape,'saved_model':str(destination)}))
