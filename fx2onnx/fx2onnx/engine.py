import torch

from dataclasses import dataclass
from typing import Any
import onnx
from onnx import helper, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto, ValueInfoProto, OperatorSetIdProto
from .export_functions import pytorch_func_export, pytorch_layer_export, pytorch_tensor_export
from .utils import obj_to_value_info, type_np_to_onnx, np_inst_to_value_info, ArgInfo, scalar_to_value_info
from .pytorch_layers import pytorch_layer_initializer
from .node import OnnxNodes
from .utils import scalar_to_constant_node



class ExportEngine(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule) -> None:
        super().__init__(module, False)
        self.name_cache = {} # Map[str, int] name -> last_id to ensure SSA in onnx
        self.input_value_infos = [] # List[ValueInfoProto]
        self.value_infos = {} # Map[str, ValueInfoProto] name_in_onnx -> value_info
        self.initializers = {} # Map[obj(submodule), List[TensorProto]] obj -> onnxnode
        self.compute_nodes = []
        self.output_node_names = [] # List[str] name_in_onnx
        self.tensor_2_onnx_name = {} # Map[Any, str] object in interpreter->name_in_onnx
        
    # def fetch_args_kwargs_info(self, node):

    # def run_node(self, n):
    #     arg_info = super().run_node(n)

    def placeholder(self, target, args, kwargs):
        ret_obj = super().placeholder(target, [], {})
        if id(ret_obj) not in self.tensor_2_onnx_name:
            name_in_onnx = self.gen_name(target)
        else:
            name_in_onnx = self.tensor_2_onnx_name[id(ret_obj)]
        if name_in_onnx not in self.value_infos:
            arg_value_info = obj_to_value_info(name_in_onnx, ret_obj, self)
            self.input_value_infos.append(arg_value_info)
        return ArgInfo([name_in_onnx], ret_obj)
    
    def get_obj_from_arginfo(self, arg):
        if isinstance(arg, ArgInfo):
            return arg.obj
        if isinstance(arg, tuple):
            return tuple([self.get_obj_from_arginfo(a) for a in arg])
        if isinstance(arg, list):
            return [self.get_obj_from_arginfo(a) for a in arg]
        return arg

    def get_obj_from_args_kwargs(self, args, kwargs):
        args_obj = [self.get_obj_from_arginfo(arg) for arg in args ]
        kwargs_obj = {k: self.get_obj_from_arginfo(v) for k, v in kwargs.items()}
        return args_obj, kwargs_obj

    def call_module(self, target, args, kwargs):
        args_obj, kwargs_obj = self.get_obj_from_args_kwargs(args, kwargs)
        return_value = super().call_module(target, args_obj, kwargs_obj)
        submod = self.fetch_attr(target)
        if type(submod) in pytorch_layer_export:
            onnx_node = pytorch_layer_export[type(submod)](args, kwargs, return_value, target, submod, self)
        else:
            from frontend.control_flow import CondModule, LoopModule
            if isinstance(submod, CondModule):
                onnx_node =  self.call_cond(submod, args, kwargs, return_value)
            elif isinstance(submod, LoopModule):
                onnx_node = self.call_loop(submod, args, kwargs, return_value)
            else:
                # self.compute_nodes += onnx_node.def_nodes
                # self.value_infos.update(onnx_node.def_value_infos)
                raise NotImplementedError('module not found', submod)
        self.compute_nodes += onnx_node.def_nodes
        self.value_infos.update(onnx_node.def_value_infos)
        return ArgInfo(onnx_node.out_node, return_value)
    
    def get_attr(self, target, args, kwargs):
        return_value = super().get_attr(target, args, kwargs)
        onnx_node = self.export_layer(target, return_value)
        self.compute_nodes += onnx_node.def_nodes
        self.value_infos.update(onnx_node.def_value_infos)
        return ArgInfo(onnx_node.out_node, return_value)
    
    
    def call_function(self, target, args, kwargs):
        args_obj, kwargs_obj = self.get_obj_from_args_kwargs(args, kwargs)
        return_value = super().call_function(target, args_obj, kwargs_obj)
        out_name = target.__name__ if hasattr(target, '__name__') else 'func'
        if target in pytorch_func_export:
            onnx_node = pytorch_func_export[target](args, kwargs, return_value, out_name, target, self)
            self.compute_nodes += onnx_node.def_nodes
            self.value_infos.update(onnx_node.def_value_infos)
        else:
            raise NotImplementedError('function not found', target)
        ret = ArgInfo(onnx_node.out_node, return_value)
        return ret
    
    def call_method(self, target, args, kwargs):
        args_obj, kwargs_obj = self.get_obj_from_args_kwargs(args, kwargs)
        return_value = super().call_method(target, args_obj, kwargs_obj)
        out_name = target if isinstance(target, str) else 'method'
        if target in pytorch_tensor_export:
            onnx_node = pytorch_tensor_export[target](args, kwargs, return_value, out_name, target, self)
            self.compute_nodes += onnx_node.def_nodes
            self.value_infos.update(onnx_node.def_value_infos)
        else:
            raise NotImplementedError('method not found', target)
        return ArgInfo(onnx_node.out_node, return_value)

    def call_cond(self, cond_module, args, kwargs, return_value):
        assert len(kwargs) == 0
        args_obj, _ = self.get_obj_from_args_kwargs(args, kwargs)
        body_graphs = []
        for body_module in (cond_module.true_body, cond_module.false_body):
            # TODO: what if nested control flow?
            sub_engine = ExportEngine(body_module)
            sub_engine.name_cache = self.name_cache # NOTE: reuse name_cache to avoid name conflict
            sub_engine.tensor_2_onnx_name = {id(a.obj): a.onnx_names[0] for a in args}
            out = sub_engine.run(*args_obj)
            graph_value_info_in = []
            graph_value_info_out = []
            for onnx_name in sub_engine.output_node_names:
                graph_value_info_out.append(sub_engine.value_infos[onnx_name])
            body_graph = onnx.helper.make_graph(
                sub_engine.compute_nodes,
                self.gen_name('IF_NODE'),
                graph_value_info_in,
                graph_value_info_out
            )
            body_graph.value_info.extend(sub_engine.value_infos.values())
            for submod, ini in sub_engine.initializers.items():
                self.add_initializer(submod, ini)
            body_graphs.append(body_graph)
        out_names = [self.gen_name('IF_OUT') for _ in range(len(return_value))]
        out_value_infos = [obj_to_value_info(name, value, self) for name, value in zip(out_names, return_value)]
        onnx_node = onnx.helper.make_node(
            'If',
            inputs=[args[0].onnx_names[0]],
            outputs=out_names,
            then_branch=body_graphs[0],
            else_branch=body_graphs[1]
        )
        onnx_node = OnnxNodes(def_nodes=[onnx_node], def_value_infos={v.name: v for v in out_value_infos}, out_node=out_names)
        return onnx_node
    
    def call_loop(self, loop_module, args, kwargs, return_value):
        print("start call_loop")
        assert len(kwargs) == 0
        num_iter = loop_module.num_iter
        num_iter_node = scalar_to_constant_node('num_iter', num_iter, self)
        cond_in_name = self.gen_name('cond_in')
        cond_in = onnx.helper.make_tensor_value_info(cond_in_name, onnx.TensorProto.BOOL, [])
        cond_in_onnx_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[cond_in_name],
            value=onnx.helper.make_tensor(
                name=self.gen_name(),
                data_type=onnx.TensorProto.BOOL,
                dims=[],
                vals=[1], # numpy does not support zero-dimention tensor
            )
        )
        cond_in_node = OnnxNodes([cond_in_onnx_node], [cond_in_name], {cond_in_name: cond_in})
        ret_node = num_iter_node + cond_in_node
        # construct inner graph
        print("body_fx_graph", loop_module.body.graph)
        iter_count = scalar_to_value_info(self.gen_name('iter_count'), onnx.TensorProto.INT64, self)
        val_nodes = OnnxNodes([], [iter_count.name], {})

        args_obj, _ = self.get_obj_from_args_kwargs(args, kwargs)
        body_module = loop_module.body
        sub_engine = ExportEngine(body_module)
        sub_engine.name_cache = self.name_cache
        sub_engine.tensor_2_onnx_name = {id(a.obj): a.onnx_names[0] for a in args[:loop_module.num_read_only_param]}
        print("tensor2onnx", sub_engine.tensor_2_onnx_name)
        out = sub_engine.run(0, *args_obj)

        cond_out_name = self.gen_name('cond_out')
        cond_out = onnx.helper.make_tensor_value_info(cond_out_name, onnx.TensorProto.BOOL, [])
        cond_out_onnx_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[cond_out_name],
            value=onnx.helper.make_tensor(
                name=self.gen_name(),
                data_type=onnx.TensorProto.BOOL,
                dims=[],
                vals=[1], # numpy does not support zero-dimention tensor
            )
        )
        sub_engine.compute_nodes.append(cond_out_onnx_node)
        graph_value_info_out = [cond_out]
        for onnx_name in sub_engine.output_node_names:
            graph_value_info_out.append(sub_engine.value_infos[onnx_name])
        graph_value_info_in = [sub_engine.input_value_infos[0], cond_in] + sub_engine.input_value_infos[loop_module.num_read_only_param + 1:]
        body_graph = onnx.helper.make_graph(
            sub_engine.compute_nodes,
            self.gen_name('FOR_LOOP'),
            graph_value_info_in,
            graph_value_info_out
        )
        body_graph.value_info.extend(sub_engine.value_infos.values())
        body_graph.value_info.append(cond_out)
        print("body_value_info", [v.name for v in body_graph.value_info])
        for submod, ini in sub_engine.initializers.items():
            self.add_initializer(submod, ini)
        print("[loop subgraph]")
        print(onnx.helper.printable_graph(body_graph))

        # construct loop node
        in_names = [num_iter_node.out_node[0], cond_in_node.out_node[0]]
        for arg in args[loop_module.num_read_only_param:]:
            assert isinstance(arg, ArgInfo)
            assert len(arg.onnx_names) == 1
            in_names.append(arg.onnx_names[0])
        out_names = [self.gen_name('IF_OUT') for _ in range(len(return_value))]
        out_value_infos = [obj_to_value_info(name, value, self) for name, value in zip(out_names, return_value)]
        onnx_node = onnx.helper.make_node(
            'Loop',
            inputs=in_names,
            outputs=out_names,
            body=body_graph
        )
        ret_node.set_outputs([onnx_node], out_names, {v.name: v for v in out_value_infos})
        return ret_node
        

    def add_graph_output(self, obj, onnx_name):
        input_names = [v.name for v in self.input_value_infos]
        if onnx_name not in input_names:
            self.output_node_names.append(onnx_name)
        else:
            identity_name = self.gen_name('identity')
            identity_node = onnx.helper.make_node(
                'Identity',
                inputs=[onnx_name],
                outputs=[identity_name]
            )
            self.compute_nodes.append(identity_node)
            identity_value_info = obj_to_value_info(identity_name, obj, self)
            self.output_node_names.append(identity_name)
            self.value_infos[identity_name] = identity_value_info

    def output(self, target, args, kwargs):
        args_obj, kwargs_obj = self.get_obj_from_args_kwargs(args, kwargs)
        return_value = super().output(target, args, kwargs)
        assert len(args) == 1
        args = args[0]
        if isinstance(args, tuple):
            for arg in args:
                if isinstance(arg, ArgInfo):
                    self.add_graph_output(arg.obj, arg.onnx_names[0])
                else:
                    raise NotImplementedError(str(type(arg)))
        elif isinstance(args, ArgInfo):
            self.add_graph_output(args.obj, args.onnx_names[0])
        else:
            raise NotImplementedError(str(type(args)))
        return return_value

    def gen_name(self, name=None):  # str -> str
        if isinstance(name, str):
            pass
        elif name is None:
            name = "@tmp"
        else:
            raise NotImplementedError(str(type(name)))

        if name not in self.name_cache:
            self.name_cache[name] = 0
        else:
            self.name_cache[name] += 1
        return name + f"_{self.name_cache[name]}"
    
    def register_value_info(self, value_info):  # ValueInfoProto -> None
        assert(value_info.name not in self.value_infos)
        self.value_infos[value_info.name] = value_info

    def add_initializer(self, submod, initializers):
        if submod not in self.initializers:
            assert not isinstance(submod, str)  # make sure not passing a "name"
            assert isinstance(initializers, list)
            self.initializers[submod] = initializers


    def export_layer(self, name, func_inst):
        if func_inst in self.initializers:
            return OnnxNodes(out_node=[ini.name for ini in self.initializers[func_inst]], def_value_infos={})
        if type(func_inst) in pytorch_layer_initializer:
            initializers, value_infos, = pytorch_layer_initializer[type(func_inst)](
                func_inst, name, self)
            self.add_initializer(func_inst, initializers)
            return OnnxNodes(out_node=[ini.name for ini in self.initializers[func_inst]], def_value_infos=value_infos)
        else:
            print(func_inst)
            raise NotImplementedError()
    

    def get_onnx_graph(self):
        output_nodes = [self.value_infos[x] for x in self.output_node_names]
        graph_def = onnx.helper.make_graph(
            self.compute_nodes,
            "onnx_model",
            self.input_value_infos,
            output_nodes
        )
        graph_def.value_info.extend(self.value_infos.values())
        for inis in self.initializers.values():
            graph_def.initializer.extend(inis)
        
        opset = OperatorSetIdProto()
        opset.version = 11
        model = helper.make_model(graph_def, producer_name='autogen', opset_imports=[opset])
        print("[generated graph]")
        print(onnx.helper.printable_graph(model.graph))
        onnx.checker.check_model(model)
        print("onnx check passed")
        return model

    def get_onnx_node_by_name(self, name):
        raise NotImplementedError

