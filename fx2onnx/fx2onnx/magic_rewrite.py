import torch
import operator
import itertools

from frontend.control_flow import LoopModule
from frontend.fx_graph import fetch_attr

def loop_unroll(gm: torch.fx.GraphModule):
    unroll_factor = 8 # a magic number
    for node in gm.graph.nodes:
        if node.op == 'call_module' and isinstance(fetch_attr(gm, node.target), LoopModule):
            loop_module = fetch_attr(gm, node.target)
            if loop_module.num_iter % unroll_factor != 0:
                raise NotImplementedError
            loop_module.num_iter //= unroll_factor
            
            old_body = loop_module.body.graph
            new_body = torch.fx.Graph()

            replacement_mapping = {}
            def replacement_fn(node: torch.fx.Node) -> torch.fx.Node:
                return replacement_mapping[node]

            output_nodes = []
            iter_node = None
            for i in range(unroll_factor):
                placeholder_id = 0
                for node in old_body.nodes:
                    if node.op == "placeholder":
                        if placeholder_id == 0: # iter_count
                            if iter_node is None:
                                assert i == 0
                                iter_node = new_body.placeholder(node.name, node.type)
                            new_node = new_body.call_function(operator.mul, args=(iter_node, unroll_factor))
                            new_node = new_body.call_function(operator.add, args=(new_node, i))
                            replacement_mapping[node] = new_node
                        elif i == 0:
                            new_node = new_body.placeholder(node.name, node.type)
                            replacement_mapping[node] = new_node
                        else:
                            if placeholder_id >= loop_module.num_read_only_param + 1:
                                replacement_mapping[node] = output_nodes[placeholder_id - loop_module.num_read_only_param - 1]
                        placeholder_id += 1
                    elif node.op == "output":
                        output_nodes = [replacement_mapping[x] for x in node.args[0]]
                        if i == unroll_factor - 1:
                            new_node = new_body.output(tuple(output_nodes))
                            replacement_mapping[node] = new_node
                    else:
                        new_node = new_body.node_copy(node, replacement_fn)
                        replacement_mapping[node] = new_node
            loop_module.body.graph = new_body


def move_inner_constant(gm):
    for node in gm.graph.nodes:
        if node.op == 'call_module' and isinstance(fetch_attr(gm, node.target), LoopModule):
            constant_nodes: list[str, torch.fx.Node] = []
            loop_node = node
            loop_module = fetch_attr(gm, node.target)
            loop_body = loop_module.body
            loop_graph = loop_body.graph
            for node in loop_graph.nodes:
                if node.op == 'get_attr':
                    constant_nodes.append(node)
                elif node.op == 'call_module' or node.op == 'call_function' or node.op == 'call_method':
                    is_constant = True
                    for arg in node.args:
                        if arg not in constant_nodes:
                            is_constant = False
                            break
                    if is_constant:
                        constant_nodes.append(node)
            placeholder_nodes = [x for x in loop_graph.nodes if x.op == 'placeholder']
            # print("constant_nodes", constant_nodes)
            # print("======================old outer graph==================\n", gm.graph)
            # print("======================old inner graph==================\n", loop_graph, flush=True)
            outer_new_nodes = []
            inner_new_nodes = []
            with gm.graph.inserting_before(loop_node):
                # add the constant nodes to outer graph
                outer_graph_mapping = {}
                def outer_graph_replacement_fn(node: torch.fx.Node) -> torch.fx.Node:
                    return outer_graph_mapping[node]
                for node in constant_nodes:
                    new_node = gm.graph.node_copy(node, outer_graph_replacement_fn)
                    outer_graph_mapping[node] = new_node
                    outer_new_nodes.append(new_node)
                    if node.op == 'get_attr':
                        new_node.target = f"{loop_node.name}.body.{node.target}"
            with loop_graph.inserting_before(placeholder_nodes[loop_module.num_read_only_param + 1]):
                loop_node.args = tuple(itertools.chain(loop_node.args[:loop_module.num_read_only_param], outer_new_nodes, loop_node.args[loop_module.num_read_only_param:]))
                # replace the nodes in inner graph as placeholder
                for node in constant_nodes:
                    new_node = loop_graph.placeholder(node.name+'_move_out', node.type)
                    inner_new_nodes.append(new_node)
                    node.replace_all_uses_with(new_node)
                    loop_graph.erase_node(node)
            loop_module.num_read_only_param += len(outer_new_nodes)
            # print("======================new outer graph==================\n", gm.graph)
            # print("======================new inner graph==================\n", loop_graph, flush=True)


def magic_rewrite(gm: torch.fx.GraphModule):
    # the loop unroll of cocktailer is performed in python, reproduce the process here
    move_inner_constant(gm)
    loop_unroll(gm)
    for node in gm.graph.nodes:
        if node.op == 'call_module' and isinstance(fetch_attr(gm, node.target), LoopModule):
            loop_module = fetch_attr(gm, node.target)
            loop_module.body.recompile()
    gm.recompile()
    return gm