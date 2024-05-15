import numpy as np

import onnx
from onnx import TensorProto, ValueInfoProto
import torch
from .node import OnnxNodes
from dataclasses import dataclass
from typing import Any
# from ast_analyzer.shape_inference.shape_elem import unwrap_shape
# from ast_analyzer.shape_inference.types import tyobj_to_dtype
# from ast_analyzer.shape_inference.types import *


@dataclass
class ArgInfo:
    onnx_names: list[str]
    obj: Any

np2onnx = {
    np.single: TensorProto.FLOAT,
    np.float32: TensorProto.FLOAT,
    np.uint8: TensorProto.UINT8,
    np.int8: TensorProto.INT8,
    np.uint16: TensorProto.UINT16,
    np.int16: TensorProto.INT16,
    np.int32: TensorProto.INT32,
    np.int64: TensorProto.INT64,
    # skip TensorProto.STRING
    np.bool_: TensorProto.BOOL,
    np.float16: TensorProto.FLOAT16,
    np.double: TensorProto.FLOAT, # HACK DOUBLE
    np.float64: TensorProto.FLOAT, # HACK DOUBLE
    np.uint32: TensorProto.UINT32,
    np.uint64: TensorProto.UINT64,
    np.complex64: TensorProto.COMPLEX64,
    np.complex128: TensorProto.COMPLEX128,
}

torch2onnx = {
    torch.float32: TensorProto.FLOAT,
    torch.float64: TensorProto.DOUBLE,
    torch.float16: TensorProto.FLOAT16,
    torch.uint8: TensorProto.UINT8,
    torch.int8: TensorProto.INT8,
    torch.int16: TensorProto.INT16,
    torch.int32: TensorProto.INT32,
    torch.int64: TensorProto.INT64,
    torch.bool: TensorProto.BOOL,
    torch.complex64: TensorProto.COMPLEX64,
    torch.complex128: TensorProto.COMPLEX128,
}

scalar2onnx = {
    int: TensorProto.INT64,
    float: TensorProto.FLOAT,
    bool: TensorProto.BOOL,
}

torch2scalar = {
    torch.float32: float,
    torch.float64: float,
    torch.float16: float,
    torch.uint8: int,
    torch.int8: int,
    torch.int16: int,
    torch.int32: int,
    torch.int64: int,
    torch.bool: bool,
    torch.complex64: complex,
    torch.complex128: complex,
}

def type_np_to_onnx(np_dtype):
    # cannot use np2onnx[np_dtype], so hack it
    for np_ty in np2onnx.keys():
        if np_ty == np_dtype:
            return np2onnx[np_ty]
    raise NotImplementedError(
        np_dtype + " is not supported in type_np_to_onnx")


def type_torch_to_onnx(torch_dtype):
    return torch2onnx[torch_dtype]

def type_scalar_to_onnx(scalar):
    return scalar2onnx[type(scalar)]


def type_onnx_to_np(onnx_dtype):
    if 'float16' in onnx_dtype:
        return np.float16
    elif 'float' in onnx_dtype:
        return np.float32
    elif 'double' in onnx_dtype:
        return np.float32 # HACK DOUBLE
    elif 'int8' in onnx_dtype:
        return np.int8
    elif 'int16' in onnx_dtype:
        return np.int16
    elif 'int32' in onnx_dtype:
        return np.int32
    elif 'int64' in onnx_dtype:
        return np.int64
    elif 'uint8' in onnx_dtype:
        return np.uint8
    elif 'uint16' in onnx_dtype:
        return np.uint16
    elif 'bool' in onnx_dtype:
        return np.bool_
    else:
        raise NotImplementedError(
            onnx_dtype + " is not supported in type_onnx_to_np")
    return np.float32

# str, np.array, ExportEngine -> ValueInfoProto
def np_inst_to_value_info(name, np_inst, visitor):
    value_info = onnx.helper.make_tensor_value_info(
        name, type_np_to_onnx(np_inst.dtype), np_inst.shape)
    visitor.register_value_info(value_info)
    return value_info

def unwrap_shape(shape):
    return tuple(shape)


# str, Type, ExportEngine -> ValueInfoProto
def obj_to_value_info(name, obj, visitor):
    if isinstance(obj, torch.Tensor):
        value_info = onnx.helper.make_tensor_value_info(
            name, type_torch_to_onnx(obj.dtype), obj.shape)
    elif isinstance(obj, (int, float, bool)):
        value_info = onnx.helper.make_tensor_value_info(
            name, type_scalar_to_onnx(obj), [])
    elif isinstance(obj, tuple):
        value_info = onnx.helper.make_tensor_value_info(
            name, TensorProto.INT64, [len(obj)])
    else:
        raise NotImplementedError(
            str(type(obj)) + " is not supported in obj_to_value_info")

    visitor.register_value_info(value_info)
    return value_info


# str, onnx_type, ExportEngine -> ValueInfoProto
def scalar_to_value_info(name, onnx_ty, visitor):
    value_info = onnx.helper.make_tensor_value_info(name, onnx_ty, [])
    visitor.register_value_info(value_info)
    return value_info


def scalar_to_constant_node(name, value, visitor):  # ASTNode -> OnnxNodes
    name = visitor.gen_name('const_' + str(value))
    if isinstance(value, bool):
        dtype = np.bool_
    elif isinstance(value, int):
        dtype = np.int64
    elif isinstance(value, float):
        dtype = np.float32
    else:
        raise NotImplementedError
    onnx_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=onnx.numpy_helper.from_array(np.full((), value, dtype=dtype))
    )
    value_info = obj_to_value_info(name, value, visitor)
    return OnnxNodes(
        def_nodes=[onnx_node],
        def_value_infos={name: value_info},
        out_node=[name]
    )

# ValueInfoProto, str, ExportEngine -> ValueInfoProto
def value_info_with_name(value_info, name, visitor):
    new_info = ValueInfoProto()
    new_info.CopyFrom(value_info)
    new_info.name = name
    visitor.register_value_info(new_info)
    return new_info


def have_real_func(model):
    if len(model.graph.input) == 0:
        return False
    found = False
    op_ignore = ['Gather']
    for node in model.graph.node:
        print(node.op_type)
        if node.op_type not in op_ignore:
            found = True
            break
    return found


def unify_tensor_arg(arg, dtype, visitor):
    if isinstance(arg, ArgInfo):
        return OnnxNodes(
            def_nodes=[],
            def_value_infos={},
            out_node=[arg.onnx_names[0]]
        )
    elif isinstance(arg, int):
        return scalar_to_constant_node('const_' + str(arg), torch2scalar[dtype](arg), visitor)
    else:
        raise NotImplementedError(str(type(arg)) + " is not supported in unify_tensor_arg")

def unify_tensor_args(args, visitor):
    tensor_dtype = None
    for arg in args:
        if isinstance(arg, ArgInfo):
            if tensor_dtype is None:
                if isinstance(arg.obj, torch.Tensor):
                    tensor_dtype = arg.obj.dtype
                elif isinstance(arg.obj, int):
                    tensor_dtype = torch.int64
                else:
                    raise NotImplementedError
            else:
                assert tensor_dtype == arg.obj.dtype
    if tensor_dtype is None:
        raise NotImplementedError("No tensor in args")
    return [unify_tensor_arg(arg, tensor_dtype, visitor) for arg in args]
