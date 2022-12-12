import onnx
import numpy as np
from onnx.helper import make_node, make_tensor

CONSTANT_SIZE = 16
tensor_shapes={}

def get_info(onnx_model):
    input_node = []
    graph = onnx_model.graph
    value_infos = [val_info for val_info in graph.value_info]
    inputs = [input_ for input_ in graph.input]
    outputs = [output_ for output_ in graph.output]
    value_infos.extend(inputs)
    value_infos.extend(outputs)
    for val_info in value_infos:
        tensor_name = val_info.name
        shape = [dim.dim_value for dim in val_info.type.tensor_type.shape.dim]
        tensor_shapes[tensor_name] = shape
    for i, node in enumerate(onnx_model.graph.node):
        input_node.append(node)
    return input_node

def find_layernorm(onnx_model,input_node):
    count = 0
    node_num = len(input_node)
    for i, node in enumerate(input_node):
        if(i < node_num - 10 and input_node[i+1].op_type == 'ReduceMean' 
        and input_node[i+2].op_type == 'Sub' and input_node[i+3].op_type == 'Cast' 
        and input_node[i+4].op_type == 'Pow' and input_node[i+5].op_type == 'ReduceMean' 
        and input_node[i+6].op_type == 'Add' and input_node[i+7].op_type == 'Sqrt' 
        and input_node[i+8].op_type == 'Div' and input_node[i+9].op_type == 'Mul' 
        and input_node[i+10].op_type == 'Add' and node.output[0] == input_node[i+2].input[0] 
        and input_node[i+1].output[0] == input_node[i+2].input[1] and input_node[i+3].output[0] == input_node[i+4].input[0]
        and input_node[i+5].output[0] == input_node[i+6].input[0] and input_node[i+2].output[0] == input_node[i+8].input[0] 
        and input_node[i+7].output[0] == input_node[i+8].input[1] and input_node[i+8].output[0] == input_node[i+9].input[0] 
        and input_node[i+9].output[0] == input_node[i+10].input[0]):
            count = modify(onnx_model, input_node, 10, i, count)    
        elif (i < node_num - 9 and input_node[i+1].op_type == 'ReduceMean' 
        and input_node[i+2].op_type == 'Sub' and input_node[i+3].op_type == 'Pow' 
        and input_node[i+4].op_type == 'ReduceMean' and input_node[i+5].op_type == 'Add' 
        and input_node[i+6].op_type == 'Sqrt' and input_node[i+7].op_type == 'Div'
        and input_node[i+8].op_type == 'Mul' and input_node[i+9].op_type == 'Add'
        and node.output[0] == input_node[i+2].input[0] and input_node[i+1].output[0] == input_node[i+2].input[1]     
        and input_node[i+2].output[0] == input_node[i+3].input[0] and input_node[i+4].output[0] == input_node[i+5].input[0]
        and input_node[i+2].output[0] == input_node[i+7].input[0] and input_node[i+6].output[0] == input_node[i+7].input[1] 
        and input_node[i+7].output[0] == input_node[i+8].input[0] and input_node[i+8].output[0] == input_node[i+9].input[0]):
            count = modify(onnx_model, input_node, 9, i, count)    
        elif(i < node_num - 8 and input_node[i+1].op_type == 'ReduceMean' 
        and input_node[i+2].op_type == 'Sub' and input_node[i+3].op_type == 'Cast' 
        and input_node[i+4].op_type == 'Pow' and input_node[i+5].op_type == 'ReduceMean' 
        and input_node[i+6].op_type == 'Add' and input_node[i+7].op_type == 'Sqrt' 
        and input_node[i+8].op_type == 'Div' and node.output[0] == input_node[i+2].input[0] 
        and input_node[i+1].output[0] == input_node[i+2].input[1] and input_node[i+3].output[0] == input_node[i+4].input[0]
        and input_node[i+5].output[0] == input_node[i+6].input[0] and input_node[i+2].output[0] == input_node[i+8].input[0] 
        and input_node[i+7].output[0] == input_node[i+8].input[1]): 
            count = modify(onnx_model, input_node, 8, i, count)  
        elif(i < node_num - 7 and input_node[i+1].op_type == 'ReduceMean' 
        and input_node[i+2].op_type == 'Sub' and input_node[i+3].op_type == 'Pow' 
        and input_node[i+4].op_type == 'ReduceMean' and input_node[i+5].op_type == 'Add' 
        and input_node[i+6].op_type == 'Sqrt' and input_node[i+7].op_type == 'Div' 
        and node.output[0] == input_node[i+2].input[0] and input_node[i+1].output[0] == input_node[i+2].input[1]     
        and input_node[i+2].output[0] == input_node[i+3].input[0] and input_node[i+4].output[0] == input_node[i+5].input[0]
        and input_node[i+2].output[0] == input_node[i+7].input[0] and input_node[i+6].output[0] == input_node[i+7].input[1]):
            count = modify(onnx_model, input_node, 7, i, count)     

def modify(onnx_model, input_node, index, i, count):   
    node = input_node[i]
    tensor_name = node.output
    shape = tensor_shapes[tensor_name[0]]  
    length = len(shape)
    if(length >= 2):
        pad_size = [0 for x in range(2*length)]
        if(shape[-1] % CONSTANT_SIZE != 0 or shape[-2] % CONSTANT_SIZE != 0):
            if(shape[-1] % CONSTANT_SIZE != 0):
                pad_size[-1] = (int(shape[-1] / CONSTANT_SIZE) + 1 ) * CONSTANT_SIZE - shape[-1]
            if(shape[-2] % CONSTANT_SIZE != 0):
                pad_size[-2] = (int(shape[-2] / CONSTANT_SIZE) + 1 ) * CONSTANT_SIZE - shape[-2]
            new_nodes = create_pad(onnx_model, input_node[i], input_node[i+1], input_node[i+2], 0, 0, 0, pad_size, length)  
            onnx_model.graph.node.insert(i + 1 + count, new_nodes)
            count = count + 1
            div_tensor_name = input_node[i + index].output
            slice_shape = tensor_shapes[div_tensor_name[0]]
            slice_length = len(slice_shape)
            start = [0 for x in range(slice_length)]
            end = slice_shape
            slice_nodes = create_slice(onnx_model, input_node[i + index], 0, start, end, slice_length)
            onnx_model.graph.node.insert(i + index + 1 + count, slice_nodes)
            count = count + 1
    return count
            
def create_pad(onnx_model, node, rmnode, subnode, out_idx, rm_in_idx, sub_in_idx, pad_size, length):
    """Create pad and pad size node"""
    node_out_name = node.output[out_idx]
    node_out_new_name = node_out_name + "_padded"
    pad_size_node_name = node_out_name + "_pad_size"
    pad_value_node_name = node_out_name + "_pad_value"
    pad_node_name = node_out_name + "_pad"

    pads_value = make_tensor(name=pad_value_node_name,
                                     data_type=6,
                                     dims=[],
                                     vals=[0])
    pad_const = make_tensor(name=pad_size_node_name, data_type=onnx.TensorProto.INT64, dims=[2*length], vals=pad_size)
    pad_node = make_node(op_type="Pad",
                                     inputs=[node_out_name, pad_size_node_name, pad_value_node_name],
                                     outputs=[node_out_new_name],
                                     name=pad_node_name,
                                     mode="constant")
                                    
    rmnode.input[rm_in_idx] = node_out_new_name
    subnode.input[sub_in_idx] = node_out_new_name
    onnx_model.graph.initializer.append(pads_value)
    onnx_model.graph.initializer.append(pad_const)
    return pad_node

def create_slice(onnx_model, node, out_idx, start, end, len):
    node_out_name = node.output[out_idx]
    node_out_new_name = node_out_name + "_sliced"
    slice_node_name = node_out_name + "_slice"
    start_node_name = node_out_name + "_start"
    end_node_name = node_out_name + "_end"

    node.output[out_idx] = node_out_new_name 
    start_t = make_tensor(name=start_node_name, data_type=onnx.TensorProto.INT64, dims=[len],vals=start)
    end_t = make_tensor(name=end_node_name, data_type=onnx.TensorProto.INT64, dims=[len],vals=end)
    slice_node = make_node(
        "Slice",
        inputs=[node_out_new_name, start_node_name, end_node_name],
        outputs=[node_out_name],
        name=slice_node_name)
    onnx_model.graph.initializer.append(start_t)
    onnx_model.graph.initializer.append(end_t)
    return slice_node

def add_pad(model_path):
    onnx_model = onnx.load(model_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    input_node = get_info(onnx_model)
    find_layernorm(onnx_model,input_node)
    onnx.checker.check_model(onnx_model)
    return onnx_model