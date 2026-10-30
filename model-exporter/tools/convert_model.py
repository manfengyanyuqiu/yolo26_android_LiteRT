"""Convert user's ONNX using onnx2tf; validate output numerically against ONNX."""
import argparse
import hashlib
import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
from pathlib import Path
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx2tf import convert
from onnx import numpy_helper

def yolo26_output_adapter(model, size, normalize, rank4, classes=80):
    """Normalize official dense output and add a singleton output axis only."""
    output = model.graph.output[0]
    shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[:2] != [1, classes+4] or shape[2] <= 0:
        raise ValueError(f'Expected fixed YOLO26 COCO dense [1,84,N], got {shape}')
    name = output.name
    # Canonicalize constant - variable: onnx2tf's broadcasting path reverses
    # this subtraction. Multiply by -1 + Add preserves the ONNX operand semantics.
    model.graph.initializer.append(numpy_helper.from_array(np.array(-1., dtype=np.float32), 'fydnn_minus_one'))
    nodes = []
    for node in model.graph.node:
        if node.name == '/model.23/Sub' and node.op_type == 'Sub':
            neg = node.output[0] + '_neg_distance'
            nodes.append(onnx.helper.make_node('Mul', [node.input[1], 'fydnn_minus_one'], [neg], name='fydnn_neg_distance'))
            node.op_type = 'Add'; node.input[1] = neg
        nodes.append(node)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    if normalize:
        # Normalize boxes BEFORE class concatenation. Post-concat normalization
        # destroys probabilities in INT8 because pixel boxes share their scale.
        stride_mul = next(n for n in model.graph.node if n.name == '/model.23/Mul_2')
        stride_node = next(n for n in model.graph.node if stride_mul.input[1] in n.output)
        if stride_node.op_type != 'Constant': raise ValueError('Expected fixed YOLO26 strides')
        stride = next(a.t for a in stride_node.attribute if a.name == 'value')
        value = numpy_helper.to_array(stride).astype(np.float32) / np.float32(size)
        model.graph.initializer.append(numpy_helper.from_array(value, 'fydnn_normalized_strides'))
        stride_mul.input[1] = 'fydnn_normalized_strides'
    if rank4:
        model.graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), 'fydnn_axes'))
        model.graph.node.append(onnx.helper.make_node('Unsqueeze', [name, 'fydnn_axes'],
                                                     ['fydnn_rank4'], name='fydnn_rank4'))
        name = 'fydnn_rank4'; shape = [1, 1, *shape[1:]]
    output.CopyFrom(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    onnx.checker.check_model(model)


def yolo_v8_output_adapter(model, size, normalize, rank4, classes):
    """Adapt a simplified decoded v8 head, preserving box and score axis order."""
    shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    if shape != [1, classes+4, sum((size//s)**2 for s in (8,16,32))]:
        raise ValueError(f'Expected decoded fixed YOLOv8 output, got {shape}')
    producers = {name: n for n in model.graph.node for name in n.output}
    final = producers[model.graph.output[0].name]
    if final.op_type != 'Concat' or len(final.input) != 2:
        raise ValueError('Expected YOLOv8 final boxes/scores Concat without embedded NMS')
    boxes = producers[final.input[0]]
    if boxes.op_type != 'Mul' or producers[final.input[1]].op_type != 'Sigmoid':
        raise ValueError('Unsupported YOLOv8 decoded boxes/scores topology')
    constants = {t.name:numpy_helper.to_array(t) for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == 'Constant':
            for attr in n.attribute:
                if attr.name == 'value': constants[n.output[0]] = numpy_helper.to_array(attr.t)
    stride_names = [name for name in boxes.input if name in constants]
    if len(stride_names) != 1: raise ValueError('Expected fixed YOLOv8 stride tensor')
    stride_name = stride_names[0]
    strides = constants[stride_name].astype(np.float32)
    if not np.isin(strides,[8,16,32]).all(): raise ValueError('Unexpected YOLOv8 stride values')
    if normalize:
        model.graph.initializer.append(numpy_helper.from_array(strides/np.float32(size),'fydnn_normalized_strides'))
        boxes.input[list(boxes.input).index(stride_name)] = 'fydnn_normalized_strides'
        stride_name = 'fydnn_normalized_strides'
    # onnx2tf can reverse a constant-minus-variable broadcast in decoded boxes.
    replacement=[]
    for n in model.graph.node:
        if n.op_type=='Sub' and n.input[0] in constants and n.input[1] not in constants:
            negative=n.output[0]+'_negative'
            neg_name=n.output[0]+'_minus_one'
            model.graph.initializer.append(numpy_helper.from_array(np.array(-1,dtype=np.float32),neg_name))
            replacement.append(onnx.helper.make_node('Mul',[n.input[1],neg_name],[negative],name=n.name+'_neg'))
            n.op_type='Add'; n.input[1]=negative
        replacement.append(n)
    del model.graph.node[:]; model.graph.node.extend(replacement)
    output=model.graph.output[0]; name=output.name
    if rank4:
        model.graph.initializer.append(numpy_helper.from_array(np.array([1],dtype=np.int64),'fydnn_axes'))
        model.graph.node.append(onnx.helper.make_node('Unsqueeze',[name,'fydnn_axes'],['fydnn_rank4'],name='fydnn_rank4'))
        name='fydnn_rank4'; shape=[1,1,*shape[1:]]
    output.CopyFrom(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,shape))
    onnx.checker.check_model(model)
    return {'boxes':boxes.name,'box_output':boxes.output[0],'concat':final.name,'stride':stride_name}

def yolo_v8_layout_replacements(output, adapter):
    operations=[
        {'op_name':adapter['boxes'],'param_target':'inputs','param_name':adapter['stride'],'pre_process_transpose_perm':[1,0]},
        {'op_name':adapter['concat'],'param_target':'inputs','param_name':adapter['box_output'],'pre_process_transpose_perm':[0,2,1]},
        {'op_name':adapter['concat'],'param_target':'attributes','param_name':'axis','values':1},
    ]
    path=output/'onnx2tf-replacements.json'
    path.write_text(json.dumps({'operations':operations},indent=2),encoding='utf-8')
    return path


def yolo26_layout_replacements(output, normalize):
    # onnx2tf 1.26.3 decodes boxes to NWC but retains scores in NCW.
    # Restore NCW explicitly before the official final channel concatenation.
    operations = [
        {'op_name': '/model.23/Mul_2', 'param_target': 'inputs',
         'param_name': 'fydnn_normalized_strides' if normalize else '/model.23/Constant_15_output_0', 'pre_process_transpose_perm': [1,0]},
        {'op_name': '/model.23/Concat_3', 'param_target': 'inputs',
         'param_name': '/model.23/Mul_2_output_0', 'pre_process_transpose_perm': [0,2,1]},
        {'op_name': '/model.23/Concat_3', 'param_target': 'attributes',
         'param_name': 'axis', 'values': 1},
    ]
    path = output / 'onnx2tf-replacements.json'
    path.write_text(json.dumps({'operations': operations}, indent=2), encoding='utf-8')
    return path


def yolo_v5_mediatek_4d(model):
    """Fold only the detection head's H/W axes; keep all runtime tensors 4D.

    Neuron 9's LiteRT dispatch rejects padded-dimension queries for rank-5
    partition boundaries. This is an equivalent reshape, not CPU execution of
    the detection head. The original ONNX is still the validation reference.
    """
    for node in model.graph.node:
        if not node.name.startswith('/model.24/'):
            continue
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name != 'value':
                    continue
                value = numpy_helper.to_array(attr.t)
                if value.ndim == 1 and value.size == 5:
                    # [N,A,C,H,W] before detection-head transpose.
                    value = np.array([*value[:3], value[3]*value[4]], dtype=value.dtype)
                elif value.ndim == 1 and value.size == 3 and value[0] == 1:
                    # Preserve a rank-4 output even after flattening anchors.
                    value = np.array([1, 1, *value[1:]], dtype=value.dtype)
                elif value.ndim == 5:
                    value = value.reshape(value.shape[0], value.shape[1],
                                          value.shape[2]*value.shape[3], value.shape[4])
                attr.t.CopyFrom(numpy_helper.from_array(value))
        elif node.op_type == 'Transpose':
            for attr in node.attribute:
                if attr.name == 'perm':
                    if list(attr.ints) != [0, 1, 3, 4, 2]:
                        raise ValueError('Unexpected YOLOv5 head transpose')
                    del attr.ints[:]
                    attr.ints.extend([0, 1, 3, 2])
        elif node.op_type in ('Split', 'Concat'):
            for attr in node.attribute:
                if attr.name == 'axis':
                    attr.i = 3 if attr.i == 4 else 2 if node.output[0] == 'output0' else attr.i
    output = model.graph.output[0]
    shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError('Expected fixed [1,N,C] YOLOv5 output')
    output.CopyFrom(onnx.helper.make_tensor_value_info(
        output.name, output.type.tensor_type.elem_type, [1, 1, *shape[1:]]))
    # Existing inferred rank-5 metadata describes the old graph.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)


def yolo_v5_layout_replacements(model, output):
    """Keep YOLOv5's [batch,anchor,y,x,xy] constants in their ONNX layout.

    onnx2tf normally interprets every rank-5 constant as NCDHW and moves axis
    1 to the end. YOLOv5's decoded detection head is instead [N,A,H,W,2], so
    those constants must use the inverse permutation at their consuming op.
    """
    constant_values = {}
    for node in model.graph.node:
        if node.op_type != 'Constant':
            continue
        for attr in node.attribute:
            if attr.name == 'value' and attr.type == onnx.AttributeProto.TENSOR:
                constant_values[node.output[0]] = numpy_helper.to_array(attr.t)

    operations = []
    for node in model.graph.node:
        if node.op_type not in ('Add', 'Mul') or not node.name.startswith('/model.24/'):
            continue
        for input_name in node.input:
            value = constant_values.get(input_name)
            if value is None or value.ndim not in (4,5) or value.shape[0] != 1 \
                    or value.shape[1] != 3 or value.shape[-1] != 2:
                continue
            operations.append({
                'op_name': 'wa' + node.name,
                'param_target': 'inputs',
                'param_name': 'wa' + input_name,
                'pre_process_transpose_perm': [0, 4, 1, 2, 3] if value.ndim == 5 else [0, 3, 1, 2],
            })
    replacement_path = output / 'onnx2tf-replacements.json'
    replacement_path.write_text(json.dumps({'operations': operations}, indent=2), encoding='utf-8')
    return replacement_path if operations else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--family', choices=['v5','v8','26'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--classes', type=int, default=80)
    parser.add_argument('--test-image', type=Path, required=True)
    parser.add_argument('--mediatek-4d', action='store_true',
                        help='Fold YOLOv5 head spatial axes for Neuron rank-4 dispatch')
    parser.add_argument('--name', help='Explicit output stem, e.g. yolov5n_224')
    parser.add_argument('--normalize-coordinates', action='store_true')
    parser.add_argument('--calibration-dir', type=Path)
    parser.add_argument('--npu-int8-per-tensor', action='store_true',
                        help='Use per-tensor INT8 weight scales for MediaTek Neuron compatibility')
    parser.add_argument('--npu-int8-head-sigmoid', action='store_true',
                        help='Keep decoded class Sigmoid in NHWC heads for MediaTek Neuron INT8')
    parser.add_argument('--keep-saved-model', action='store_true')
    parser.add_argument('--fp32-intermediate-only', action='store_true',
                        help='Validate only the intermediate FP32 model; final precision exports require separate checks')
    args = parser.parse_args()
    source = args.model.resolve()
    calibration_dir = args.calibration_dir.resolve() if args.calibration_dir else None
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    graph = onnx.load(source)
    onnx.checker.check_model(graph)
    if len(graph.graph.input)!=1 or len(graph.graph.output)!=1:
        raise ValueError('Expected one input and one decoded YOLO output')
    # Conversion only reads this copy; the supplied ONNX remains unchanged.
    # The original v5s graph stores FP16 weights. Convert graph constants and
    # operators to FP32 without changing the source, then validate against ONNX.
    source_is_fp16 = any(t.data_type == onnx.TensorProto.FLOAT16 for t in graph.graph.initializer)
    if source_is_fp16:
        def promote(g):
            for collection in (g.input, g.output, g.value_info):
                for info in collection:
                    if info.type.HasField('tensor_type') and info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                        info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            for tensor in g.initializer:
                if tensor.data_type == onnx.TensorProto.FLOAT16:
                    promoted = numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name)
                    tensor.CopyFrom(promoted)
            for node in g.node:
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == onnx.TensorProto.FLOAT16:
                        attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype(np.float32)))
                    elif attr.type == onnx.AttributeProto.TENSORS:
                        for i,tensor in enumerate(attr.tensors):
                            if tensor.data_type == onnx.TensorProto.FLOAT16:
                                attr.tensors[i].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32),tensor.name))
                    elif attr.type == onnx.AttributeProto.GRAPH:
                        promote(attr.g)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        for child in attr.graphs: promote(child)
                if node.op_type == 'Cast':
                    for attr in node.attribute:
                        if attr.name == 'to' and attr.i == onnx.TensorProto.FLOAT16:
                            attr.i = onnx.TensorProto.FLOAT
        promote(graph.graph)
        onnx.checker.check_model(graph)
    # Compare graph conversion at the same arithmetic precision. The original
    # FP16 source is separately compared by validate_export and dataset evaluation.
    reference_source=source
    if source_is_fp16:
        reference_source=output/'reference-fp32.onnx'
        onnx.save(graph,reference_source)
    v8_adapter = None
    if args.family in ('26','v8'):
        from onnxsim import simplify
        graph, valid = simplify(graph)
        if not valid: raise ValueError('YOLO26 ONNX simplification parity failed')
        if args.family == '26':
            yolo26_output_adapter(graph, args.size, args.normalize_coordinates, args.mediatek_4d,args.classes)
        else:
            v8_adapter=yolo_v8_output_adapter(graph,args.size,args.normalize_coordinates,args.mediatek_4d,args.classes)
    elif args.normalize_coordinates:
        if args.family != 'v5':
            raise ValueError('Coordinate normalization currently requires v5')
        producers = {name: node for node in graph.graph.node for name in node.output}
        normalized_count = 0
        for node in graph.graph.node:
            if node.op_type != 'Mul' or not node.name.startswith('/model.24/'):
                continue
            left, right = (producers.get(name) for name in node.input)
            if left is not None and left.op_type in ('Add', 'Pow') and right is not None and right.op_type == 'Constant':
                attr = next(a for a in right.attribute if a.name == 'value')
                attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t) / np.float32(args.size)))
                normalized_count += 1
        if normalized_count != 6:
            raise ValueError(f'Unexpected YOLOv5 coordinate head: {normalized_count}')
    if args.mediatek_4d and args.family not in ('26','v8'):
        if args.family != 'v5':
            raise ValueError('--mediatek-4d currently requires YOLOv5')
        yolo_v5_mediatek_4d(graph)
    prepared = output / 'conversion-input.onnx'
    onnx.save(graph, prepared)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH','')
    image = tf.io.decode_image(args.test_image.read_bytes(), channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32) / 255.0
    source_height, source_width = int(image.shape[0]), int(image.shape[1])
    scale = min(args.size / source_width, args.size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    pad_x = (args.size - resized_width) // 2
    pad_y = (args.size - resized_height) // 2
    resized = tf.image.resize(image, [resized_height, resized_width])[None,...]
    sample_image = tf.pad(
        resized,
        [[0,0],[pad_y,args.size-resized_height-pad_y],[pad_x,args.size-resized_width-pad_x],[0,0]],
        constant_values=114.0/255.0,
    ).numpy()
    # onnx2tf uses this local file for shape checks. Real project image; no INT8 calibration.
    np.save(output/'calibration_image_sample_data_20x128x128x3_float32.npy',
            np.repeat(tf.image.resize(image,[128,128]).numpy()[None,...],20,axis=0))
    os.chdir(output)
    replacement_path = (yolo_v5_layout_replacements(graph, output) if args.family == 'v5' else
                        yolo26_layout_replacements(output, args.normalize_coordinates) if args.family == '26' else
                        yolo_v8_layout_replacements(output,v8_adapter) if args.family == 'v8' else None)
    convert(input_onnx_file_path=str(prepared), output_folder_path=str(output),
            overwrite_input_shape=[f'{graph.graph.input[0].name}:1,3,{args.size},{args.size}'],
            output_signaturedefs=True, copy_onnx_input_output_names_to_tflite=True,
            not_use_onnxsim=args.family in ('26','v8'),
            disable_strict_mode=args.family not in ('26','v8'), verbosity='error',
            param_replacement_file=str(replacement_path) if replacement_path else '')
    stem = (args.name or f'yolo{args.family}_{args.size}') + ('_4d' if args.mediatek_4d else '')
    if args.normalize_coordinates: stem += '_norm'
    if args.family in ('26','v8'): stem += '_channels'
    preferred = output / f'{stem}_fp32.tflite'
    fp32_file = output / 'conversion-input_float32.tflite'
    session = ort.InferenceSession(str(reference_source), providers=['CPUExecutionProvider'])
    input_dtype = np.float16 if session.get_inputs()[0].type == 'tensor(float16)' else np.float32
    sample = sample_image.transpose(0,3,1,2)
    reference = session.run(None, {session.get_inputs()[0].name:sample.astype(input_dtype)})[0].astype(np.float32)
    reports = []
    candidates = set(output.glob('*_float*.tflite'))
    if args.fp32_intermediate_only: candidates = {fp32_file}
    if fp32_file.is_file(): candidates.add(fp32_file)
    final_names = {
        'conversion-input_float16.tflite': f'{stem}_fp32io_fp16weights.tflite',
        'conversion-input_float32.tflite': preferred.name,
    }
    rename_after_validation = []
    for file in sorted(candidates):
        interpreter = tf.lite.Interpreter(model_path=str(file), num_threads=1)
        interpreter.allocate_tensors()
        ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
        if len(ins)!=1 or len(outs)!=1 or list(ins[0]['shape']) != [1,args.size,args.size,3]:
            raise ValueError(f'Unsupported converted tensors: {file}')
        interpreter.set_tensor(ins[0]['index'], sample.transpose(0,2,3,1).astype(ins[0]['dtype']))
        interpreter.invoke(); actual = interpreter.get_tensor(outs[0]['index']).astype(np.float32)
        if args.mediatek_4d:
            if actual.ndim != 4:
                raise ValueError(f'Neuron model output must stay rank 4: {actual.shape}')
            # onnx2tf may emit NHWC [1,N,C,1] or retain [1,1,N,C].
            actual = np.squeeze(actual, axis=1 if actual.shape[1] == 1 else 3)
        if actual.shape == reference.shape:
            reference_for_compare = reference
        elif actual.shape == (reference.shape[0], reference.shape[2], reference.shape[1]):
            reference_for_compare = reference.transpose(0,2,1)
        else:
            raise ValueError(f'Output layout changed: {actual.shape} vs {reference.shape}')
        if not np.isfinite(actual).all(): raise ValueError('Non-finite converted output')
        offset=5 if args.family=='v5' else 4
        channels=args.classes+offset
        layout='ROWS' if actual.shape[2] == channels else 'CHANNELS'
        if args.family == '26' and layout != 'CHANNELS':
            raise ValueError(f'YOLO26 delivery layout differs from filename: {actual.shape}')
        if (layout=='ROWS' and actual.shape[2] != channels) or (layout=='CHANNELS' and actual.shape[1] != channels):
            raise ValueError(f'Unsupported output channel axis: {actual.shape}')
        coords_a=actual[...,:4] if layout=='ROWS' else actual[:,:4,:]
        if args.normalize_coordinates: coords_a = coords_a * args.size
        coords_b=reference_for_compare[...,:4] if layout=='ROWS' else reference_for_compare[:,:4,:]
        scores_a=actual[...,4:] if layout=='ROWS' else actual[:,4:,:]
        scores_b=reference_for_compare[...,4:] if layout=='ROWS' else reference_for_compare[:,4:,:]
        coord_error=float(np.max(np.abs(coords_a-coords_b)))
        score_error=float(np.max(np.abs(scores_a-scores_b)))
        # FP16 source and reordered convolution may introduce small numerical error.
        coord_limit = .05 if args.fp32_intermediate_only else 8
        score_limit = .001 if args.fp32_intermediate_only else .05
        if coord_error>coord_limit or score_error>score_limit:
            raise ValueError(f'Conversion parity failed: coords={coord_error}, scores={score_error}')
        shape=[int(x) for x in actual.shape]
        report={'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'input_shape':[int(x) for x in ins[0]['shape']],
            'output_shape':[int(x) for x in outs[0]['shape']], 'decoded_output_shape':shape,
            'family':args.family,'classes':shape[2]-offset if layout=='ROWS' else shape[1]-offset,
            'layout':layout,'normalized_coordinates':args.normalize_coordinates,
            'source_fp16_weights_converted_to_fp32':source_is_fp16,
            'conversion_reference':'FP32-promoted source' if source_is_fp16 else 'original FP32 source',
            'mediatek_4d':args.mediatek_4d,
            'max_coordinate_error_pixels':coord_error,'max_probability_error':score_error,
            'validation':'one deterministic input; not dataset accuracy or NPU execution verification'}
        if file.name in final_names:
            rename_after_validation.append((file, output / final_names[file.name], report))
        reports.append(report); print(json.dumps(report),flush=True)
    if not reports: raise RuntimeError('onnx2tf produced no float TFLite output')
    for source_file, final_file, report in rename_after_validation:
        if source_file != final_file:
            source_file.replace(final_file)
        report['file'] = final_file.name
    (output/'validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    if calibration_dir:
        from quantize_yolo import quantize
        quantize(output, output / f'{stem}_int8.tflite', calibration_dir, args.size,
                 disable_per_channel=args.npu_int8_per_tensor, neuron_head_sigmoid=args.npu_int8_head_sigmoid)
    prepared.unlink(missing_ok=True)
    (output/'calibration_image_sample_data_20x128x128x3_float32.npy').unlink(missing_ok=True)
    if not args.keep_saved_model:
        (output/'saved_model.pb').unlink(missing_ok=True)
        (output/'fingerprint.pb').unlink(missing_ok=True)

if __name__=='__main__': main()
