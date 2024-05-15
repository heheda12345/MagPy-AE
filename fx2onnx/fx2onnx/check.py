import onnx
import onnxruntime as ort
import torch
import os
def run_fx_graph_module(graph_module, input_tensors):
    # Run the torch.fx graph module
    return graph_module(*input_tensors)

def run_onnx_with_ort(onnx_graph_proto, input_tensors):
    # Create ONNX Runtime session directly from the ONNX model proto
    ort_session = ort.InferenceSession(onnx_graph_proto.SerializeToString())

    # Get input names and shapes from the ONNX model
    input_names = [inp.name for inp in ort_session.get_inputs()]
    input_shapes = [inp.shape for inp in ort_session.get_inputs()]

    # Ensure input tensors match expected shapes
    for i, (inp, inp_shape) in enumerate(zip(input_tensors, input_shapes)):
        if tuple(inp.shape) != tuple(inp_shape):
            raise ValueError(f"Input tensor at index {i} has shape {inp.shape}, expected {inp_shape}")

    # Prepare inputs for ONNX Runtime
    ort_inputs = {name: tensor.cpu().detach().numpy() for name, tensor in zip(input_names, input_tensors)}

    # Run the ONNX model using ONNX Runtime
    ort_result = ort_session.run(None, ort_inputs)

    # Convert ONNX Runtime results to torch.Tensor
    ort_tensors = [torch.tensor(arr) for arr in ort_result]

    os.makedirs('tmp/bin', exist_ok=True)
    for i, input_name in enumerate(input_names):
        x = ort_inputs[input_name]
        with open(f'tmp/bin/input_ref_{i}.bin', 'wb') as f:
            x.tofile(f)
    for i, x in enumerate(ort_result):
         with open(f'tmp/bin/output_ref_{i}.bin', 'wb') as f:
            x.tofile(f)
    return ort_tensors

def check_onnx(graph_module, onnx_graph_proto, input_tensors):
    # Run the torch.fx graph module
    fx_result = run_fx_graph_module(graph_module, input_tensors)
    
    # Run the ONNX model using ONNX Runtime
    ort_result = run_onnx_with_ort(onnx_graph_proto, input_tensors)
    if isinstance(fx_result, torch.Tensor):
        fx_result = [fx_result,]

    # Compare results
    results_match = all(torch.allclose(fx_res.cpu(), ort_res.cpu(), rtol=1e-05, atol=1e-05) for fx_res, ort_res in zip(fx_result, ort_result))

    return results_match

# Example usage:
# graph_module = your_torch_fx_graph_module
# onnx_graph_proto = your_onnx_graph_proto
# input_tensors = [input_tensor1, input_tensor2, ...]
# result_match = compare_fx_and_onnx(graph_module, onnx_graph_proto, input_tensors)
# print("Results match:", result_match)
