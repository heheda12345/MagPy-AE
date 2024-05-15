from .engine import ExportEngine
from .check import check_onnx

def to_onnx(fx_graph, *args):
    # fx_graph = magic_rewrite(fx_graph)
    engine = ExportEngine(fx_graph)
    engine.run(*args)
    onnx_graph = engine.get_onnx_graph()
    check_onnx(fx_graph, onnx_graph, args)
    return onnx_graph
    