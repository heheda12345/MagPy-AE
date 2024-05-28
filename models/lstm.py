import torch
import torch.nn as nn
import random
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Tuple

# https://github.com/pytorch/pytorch/blob/95a86ed9ca107329151e0dc172386d50dd3471c6/benchmarks/fastrnns/custom_lstms.py#L121
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)


    def forward(
            self, input: Tensor,
            state: Tuple[Tensor,
                         Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(nn.Module):

    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_size, hidden_size))
        for i in range(num_layers):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, cur_input, state_c, state_h): 
        state_c_new = [] 
        state_h_new = []
        for j in range(self.num_layers):
            c = state_c[j]
            h = state_h[j]
            _, (h, c) = self.layers[j](cur_input, (h, c))
            state_c_new.append(c)
            state_h_new.append(h)
            cur_input = h
        return state_c_new, state_h_new


class LSTM(nn.Module):

    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_size, hidden_size))
        for i in range(num_layers):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, inputs):  # seq_len, batch, input_size
        state_c = [
            torch.zeros(self.batch_size, self.hidden_size, device='cuda')
            for _ in range(self.num_layers)
        ]
        state_h = [
            torch.zeros(self.batch_size, self.hidden_size, device='cuda')
            for _ in range(self.num_layers)
        ]
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j in range(self.num_layers):
                c = state_c[j]
                h = state_h[j]
                _, (h, c) = self.layers[j](cur_input, (h, c))
                state_c[j].copy_(c)
                state_h[j].copy_(h)
                cur_input = h
        return state_h[self.num_layers - 1]


def forward_seq(model, inputs):
    state_c = [
        torch.zeros(model.batch_size, model.hidden_size, device='cuda')
        for _ in range(model.num_layers)
    ]
    state_h = [
        torch.zeros(model.batch_size, model.hidden_size, device='cuda')
        for _ in range(model.num_layers)
    ]
    for i in range(inputs.size()[0]):
        state_c, state_h = model.forward(inputs[i], state_c, state_h)

num_layers = 10
input_size = 256
hidden_size = 256
seq_len = 64

def get_layer_with_bs(batch_size):
    model = LSTMLayer(batch_size, input_size, hidden_size, num_layers).cuda()
    return model


def get_model_with_bs(batch_size):
    model = LSTM(batch_size, input_size, hidden_size, num_layers).cuda()
    return model


def get_input(batch_size):
    inputs = torch.randn(seq_len, batch_size, input_size).cuda()
    return (inputs,), {}


def get_dynamic_inputs(batch_size, num_inputs):
    inputs = [(torch.randn(seq_len, batch_size, input_size).cuda(),) for i in range(num_inputs)]
    random.shuffle(inputs)
    return inputs, [{} for _ in range(num_inputs)]


def perf_test(batch_size):
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from utils import perf_test_run, custom_backend, assert_equal, nnf_backend
    model = get_layer_with_bs(batch_size).eval()
    compiled = torch.compile(model, backend=nnf_backend)
    # compiled = torch.compile(model)
    input_args, input_kwargs = get_input(batch_size)
    ref = forward_seq(model, input_args[0])
    out = forward_seq(compiled, input_args[0])
    assert_equal(ref, out)
    perf_test_run(ref, forward_seq, "lstm+cellcompile", 100, (compiled,) + input_args, input_kwargs)
    # perf_test_run(forward_seq, "lstm+cellcompile", 100, (model,) + input_args, input_kwargs)


if __name__ == '__main__':
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import argparse
    from frontend.no_preload import NO_LD_PRELOAD_CTX
    from frontend import config
    
    with NO_LD_PRELOAD_CTX():
        with torch.no_grad():
            parser = argparse.ArgumentParser()
            parser.add_argument("--bs", type=int, default=1)
            args = parser.parse_args()

            config.set_config('model_name', f'lstm_bs{args.bs}')
            perf_test(args.bs)

